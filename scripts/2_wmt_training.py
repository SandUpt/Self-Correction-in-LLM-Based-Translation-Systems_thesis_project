#!/usr/bin/env python3

import os
import torch
import pandas as pd
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


class Config:
    MODEL_TYPE = "llama3"  # qwen, mistral, llama2, llama3
    LANG_PAIR = "de_en"  # zh_en, de_en
    
    MODEL_PATHS = {
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
        "llama2": "meta-llama/Llama-2-7b-hf",
        "llama3": "meta-llama/Meta-Llama-3-8B"
    }
    
    NEEDS_HF_TOKEN = {
        "qwen": False,
        "mistral": False,
        "llama2": True,
        "llama3": True
    }
    
    TRUST_REMOTE_CODE = {
        "qwen": True,
        "mistral": False,
        "llama2": False,
        "llama3": False
    }
    
    # llama3 benefits from more lora targets
    LORA_TARGET_MODULES = {
        "qwen": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "llama2": ["q_proj", "k_proj", "v_proj", "o_proj"],
        #"llama3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        "llama3": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    
    PROMPT_TEMPLATES = {
        "zh_en": (
            "Task: Translate the Chinese sentence into English.\n"
            "Chinese: {src}\n"
            "English:"
        ),
        "de_en": (
            "Task: Translate the German sentence into English.\n"
            "German: {src}\n"
            "English:"
        )
    }
    
    WMT_DATA_PATHS = {
        "zh_en": {
            "train": "../../data/processed/zh_en/train_news_un_balanced_30000.tsv",
            "val": "../../data/processed/zh_en/mix2k_dev.tsv"
        },
        "de_en": {
            "train": "../../data/processed/de_en/train_europarl_newstest_balanced_26000.tsv",
            "val": "../../data/processed/de_en/mix2k_dev.tsv"
        }
    }
    
    # auto-configured
    BASE_MODEL = MODEL_PATHS[MODEL_TYPE]
    USE_HF_TOKEN = NEEDS_HF_TOKEN[MODEL_TYPE]
    USE_REMOTE_CODE = TRUST_REMOTE_CODE[MODEL_TYPE]
    TARGET_MODULES = LORA_TARGET_MODULES[MODEL_TYPE]
    PROMPT_TEMPLATE = PROMPT_TEMPLATES[LANG_PAIR]
    TRAIN_DATA = WMT_DATA_PATHS[LANG_PAIR]["train"]
    VAL_DATA = WMT_DATA_PATHS[LANG_PAIR]["val"]
    OUTPUT_DIR = f"../models/wmt_adapters/{MODEL_TYPE}_{LANG_PAIR}_4M_Lora"
    
    # lora config
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    
    # training config
    # effective batch 16, fits A100 40GB
    BATCH_SIZE = 2
    GRAD_ACCUM = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 3
    MAX_LENGTH = 512
    WARMUP_RATIO = 0.05
    SEED = 42
    
    END_TOKEN = "|||END|||"


def load_data(train_path, val_path):    
    def read_tsv(path):
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # normalize column names
        if "source_zh" in df.columns:
            df = df.rename(columns={"source_zh": "source", "target_en": "target"})
        elif "source_de" in df.columns:
            df = df.rename(columns={"source_de": "source", "target_en": "target"})
        elif "src" in df.columns:
            df = df.rename(columns={"src": "source", "ref": "target"})
        elif "source" not in df.columns:
            # first two columns are source and target
            df.columns = ["source", "target"] + list(df.columns[2:])
        
        return df[["source", "target"]].dropna()
    
    train_df = read_tsv(train_path)
    val_df = read_tsv(val_path)
    
    return train_df, val_df


def create_dataset(df, tokenizer, prompt_template, max_length, end_token):
    """Creating tokenized dataset for training"""
    
    def tokenize(examples):
        prompts = [prompt_template.format(src=src) for src in examples["source"]]
        # add space before target and end token after
        completions = [f" {tgt}{end_token}" for tgt in examples["target"]]
        
        # tokenize prompt and completion together
        full_texts = [p + c for p, c in zip(prompts, completions)]
        
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding=False
        )
        
        # create labels - mask prompt tokens with -100
        labels = []
        for i, (prompt, input_ids) in enumerate(zip(prompts, tokenized["input_ids"])):
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_tokens)
            
            # -100 for prompt tokens (don't compute loss), keep completion tokens
            label = [-100] * prompt_len + input_ids[prompt_len:]
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    return dataset


def main():
    config = Config()
    
    print(f"{'='*60}")
    print(f"WMT Fine-tuning")
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Language Pair: {config.LANG_PAIR}")
    print(f"{'='*60}")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(config.SEED)
    
    # load data
    print(f"\nLoading training data")
    print(f"  Train: {config.TRAIN_DATA}")
    print(f"  Val: {config.VAL_DATA}")
    train_df, val_df = load_data(config.TRAIN_DATA, config.VAL_DATA)
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    
    # load model and tokenizer
    print(f"\nLoading model: {config.BASE_MODEL}")
    hf_token = os.getenv("HF_TOKEN") if config.USE_HF_TOKEN else None
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL,
        token=hf_token,
        trust_remote_code=config.USE_REMOTE_CODE
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=config.USE_REMOTE_CODE
    )
    
    # enable gradient checkpointing before wrapping with LoRA
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # setup lora
    print(f"\nSetting up LoRA")
    print(f"  Rank: {config.LORA_R}, Alpha: {config.LORA_ALPHA}")
    print(f"  Target modules: {config.TARGET_MODULES}")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.TARGET_MODULES,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # create datasets
    print(f"\nPreparing datasets")
    train_dataset = create_dataset(
        train_df, tokenizer, config.PROMPT_TEMPLATE, 
        config.MAX_LENGTH, config.END_TOKEN
    )
    val_dataset = create_dataset(
        val_df, tokenizer, config.PROMPT_TEMPLATE,
        config.MAX_LENGTH, config.END_TOKEN
    )
    
    # training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=config.LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        seed=config.SEED
    )
    
    # data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    # train
    print(f"\nStarting training")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Effective batch size: {config.BATCH_SIZE * config.GRAD_ACCUM}")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    
    trainer.train()
    
    # save final adapter
    print(f"\nSaving adapter to {config.OUTPUT_DIR}")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    print(f"\nTraining complete.")
    print(f"Adapter saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()