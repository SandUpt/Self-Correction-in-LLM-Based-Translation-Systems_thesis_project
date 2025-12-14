#!/usr/bin/env python3

import os
import torch
import pandas as pd
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
    USE_END_TOKEN = False  # True =  data with |||END|||, False = original data
    
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
    
    LORA_TARGET_MODULES = {
        "qwen": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "llama2": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "llama3": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }
    
    # original data (no end token)
    SC_DATA_ORIGINAL = {
        "zh_en": {
            "train": "../../data/processed/zh_en/self_correction_zh_en/train.tsv",
            "val": "../../data/processed/zh_en/self_correction_zh_en/val.tsv"
        },
        "de_en": {
            "train": "../../data/processed/de_en/self_correction_de_en/train.tsv",
            "val": "../../data/processed/de_en/self_correction_de_en/val.tsv"
        }
    }
    
    # data with |||END||| token
    SC_DATA_WITH_END = {
        "zh_en": {
            "train": "../../data/processed/zh_en/self_correction_zh_en_end/train.tsv",
            "val": "../../data/processed/zh_en/self_correction_zh_en_end/val.tsv"
        },
        "de_en": {
            "train": "../../data/processed/de_en/self_correction_de_en_end/train.tsv",
            "val": "../../data/processed/de_en/self_correction_de_en_end/val.tsv"
        }
    }
    
    # auto-configured
    BASE_MODEL = MODEL_PATHS[MODEL_TYPE]
    USE_HF_TOKEN = NEEDS_HF_TOKEN[MODEL_TYPE]
    USE_REMOTE_CODE = TRUST_REMOTE_CODE[MODEL_TYPE]
    TARGET_MODULES = LORA_TARGET_MODULES[MODEL_TYPE]
    
    SC_DATA = SC_DATA_WITH_END if USE_END_TOKEN else SC_DATA_ORIGINAL
    TRAIN_DATA = SC_DATA[LANG_PAIR]["train"]
    VAL_DATA = SC_DATA[LANG_PAIR]["val"]
    
    SUFFIX = "_end" if USE_END_TOKEN else ""
    OUTPUT_DIR = f"../models/sc_direct/{MODEL_TYPE}_{LANG_PAIR}{SUFFIX}"
    
    # lora config
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    
    # training config
    BATCH_SIZE = 2
    GRAD_ACCUM = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 3
    MAX_LENGTH = 768
    WARMUP_RATIO = 0.05
    SEED = 42


def load_data(train_path, val_path):
    def read_tsv(path):
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        df.columns = [c.strip().lower() for c in df.columns]
        if "prompt" not in df.columns or "completion" not in df.columns:
            cols = df.columns.tolist()
            if len(cols) >= 2:
                df = df.rename(columns={cols[0]: "prompt", cols[1]: "completion"})
        return df[["prompt", "completion"]].dropna()
    
    return read_tsv(train_path), read_tsv(val_path)


def create_dataset(df, tokenizer, max_length):
    def tokenize(examples):
        prompts = examples["prompt"]
        completions = examples["completion"]
        full_texts = [p + c for p, c in zip(prompts, completions)]
        
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding=False
        )
        
        labels = []
        for i, (prompt, input_ids) in enumerate(zip(prompts, tokenized["input_ids"])):
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_tokens)
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
    
    print(f"Direct Self-Correction Training (Base - SC)")
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Language Pair: {config.LANG_PAIR}")
    print(f"Using END token: {config.USE_END_TOKEN}")
   
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(config.SEED)
    
    # load data
    print(f"\nLoading training data")
    print(f"  Train: {config.TRAIN_DATA}")
    print(f"  Val: {config.VAL_DATA}")
    train_df, val_df = load_data(config.TRAIN_DATA, config.VAL_DATA)
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    
    # load base model
    print(f"\nLoading base model: {config.BASE_MODEL}")
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
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # setup lora
    print(f"\nSetting up LoRA")
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
    train_dataset = create_dataset(train_df, tokenizer, config.MAX_LENGTH)
    val_dataset = create_dataset(val_df, tokenizer, config.MAX_LENGTH)
    
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
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="paged_adamw_32bit",
        report_to="none",
        seed=config.SEED
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    print(f"\nStarting training")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Effective batch size: {config.BATCH_SIZE * config.GRAD_ACCUM}")
    
    trainer.train()
    
    print(f"\nSaving adapter to {config.OUTPUT_DIR}")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    print(f"\nTraining complete.")


if __name__ == "__main__":
    main()