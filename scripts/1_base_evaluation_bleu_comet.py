#!/usr/bin/env python3

import os
import time
import torch
import pandas as pd
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER
from comet import download_model, load_from_checkpoint


class Config:
    MODEL_TYPE = "llama2"  # qwen, mistral, llama2, llama3
    LANG_PAIR = "de_en"  # zh_en, de_en
    
    MODEL_PATHS = {
        "qwen": "Qwen/Qwen2.5-7B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
        "llama2": "meta-llama/Llama-2-7b-hf",
        "llama3": "meta-llama/Meta-Llama-3-8B"
    }
    
    # qwen needs trust_remote_code for custom arch
    # llama models are gated, need HF token
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
    
    TEST_DATA_PATHS = {
        "zh_en": "../../data/evaluation_sets/zh_en/test_5000_clean.tsv",
        "de_en": "../../data/evaluation_sets/de_en/test_5000_clean.tsv"
    }
    
    # auto-configured
    BASE_MODEL = MODEL_PATHS[MODEL_TYPE]
    USE_HF_TOKEN = NEEDS_HF_TOKEN[MODEL_TYPE]
    USE_REMOTE_CODE = TRUST_REMOTE_CODE[MODEL_TYPE]
    PROMPT_TEMPLATE = PROMPT_TEMPLATES[LANG_PAIR]
    TEST_DATA = TEST_DATA_PATHS[LANG_PAIR]
    OUTPUT_DIR = f"../evaluations/base/{MODEL_TYPE}_{LANG_PAIR}"
    
    # evaluation settings
    SAMPLE_SIZE = None  # int for quick test
    MAX_NEW_TOKENS = 256
    # batch 32 OOMs on A100 40GB with llama3
    BATCH_SIZE = 48
    SEED = 42


def load_test_data(path, sample_size=None):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # normalize column names 
    if "source_zh" in df.columns:
        df = df.rename(columns={"source_zh": "source", "target_en": "reference"})
    elif "source_de" in df.columns:
        df = df.rename(columns={"source_de": "source", "target_en": "reference"})
    elif "src" in df.columns:
        df = df.rename(columns={"src": "source", "ref": "reference"})
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)
    
    return df


def generate_translations(model, tokenizer, sources, prompt_template):
    model.eval()
    translations = []
    
    for i in tqdm(range(0, len(sources), Config.BATCH_SIZE), desc="Generating"):
        batch_sources = sources[i:i + Config.BATCH_SIZE]
        prompts = [prompt_template.format(src=src) for src in batch_sources]
        
        # print(f"First prompt: {prompts[0]}")  # debug
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # decode only the generated part (skip input tokens)
        for j, output in enumerate(outputs):
            generated = tokenizer.decode(
                output[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            # clean up the output - base models can be messy
            generated = clean_output(generated)
            translations.append(generated)
        
        del inputs, outputs
        torch.cuda.empty_cache()
    
    return translations


def clean_output(text):
    """Extract just the translation from model output"""
    if not text:
        return ""
    
    text = text.strip()
    
    # take first line only - base models sometimes keep generating
    text = text.split("\n")[0].strip()
    
    # remove common artifacts
    # sometimes model outputs "English: <translation>" or continues with more tasks
    if text.lower().startswith("english:"):
        text = text[8:].strip()
    
    # stop at common continuation markers
    stop_markers = ["Chinese:", "German:", "Task:", "Translation:", "\n\n"]
    for marker in stop_markers:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx].strip()
    
    return text


def compute_metrics(predictions, references, sources, comet_model):
    bleu = BLEU()
    chrf = CHRF()
    ter = TER()
    
    bleu_score = bleu.corpus_score(predictions, [references]).score
    chrf_score = chrf.corpus_score(predictions, [references]).score
    ter_score = ter.corpus_score(predictions, [references]).score
    
    comet_data = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(sources, predictions, references)
    ]
    comet_output = comet_model.predict(comet_data, batch_size=64, gpus=1, progress_bar=False)
    comet_score = comet_output.system_score
    
    return {
        "BLEU": round(float(bleu_score), 2),
        "chrF": round(float(chrf_score), 2),
        "TER": round(float(ter_score), 2),
        "COMET": round(float(comet_score), 3)
    }


def compute_per_example_metrics(predictions, references, sources, comet_model):
    bleu = BLEU(effective_order=True)
    
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        if pred and ref:
            score = bleu.sentence_score(pred, [ref]).score
        else:
            score = 0.0
        bleu_scores.append(score)
    
    comet_data = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(sources, predictions, references)
    ]
    comet_output = comet_model.predict(comet_data, batch_size=64, gpus=1, progress_bar=False)
    comet_scores = comet_output.scores
    
    return bleu_scores, comet_scores


def main():
    config = Config()
    
    print(f"Base Model Evaluation")
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Language Pair: {config.LANG_PAIR}")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(config.SEED)
    
    # load test data
    print(f"\nLoading test data from {config.TEST_DATA}")
    test_df = load_test_data(config.TEST_DATA, config.SAMPLE_SIZE)
    print(f"Test examples: {len(test_df)}")
    
    # load model
    print(f"\nLoading model: {config.BASE_MODEL}")
    hf_token = os.getenv("HF_TOKEN") if config.USE_HF_TOKEN else None
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL,
        token=hf_token,
        trust_remote_code=config.USE_REMOTE_CODE
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # left padding for batch generation with decoder-only models
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=config.USE_REMOTE_CODE
    )
    
    # generate translations
    print("\nGenerating translations")
    sources = test_df["source"].tolist()
    references = test_df["reference"].tolist()
    
    start_time = time.time()
    predictions = generate_translations(model, tokenizer, sources, config.PROMPT_TEMPLATE)
    gen_time = time.time() - start_time
    print(f"Generation took {gen_time:.1f}s ({len(sources)/gen_time:.1f} examples/s)")
    
    #  few samples
    print("\nSample outputs:")
    for i in range(min(3, len(predictions))):
        print(f"  [{i}] src: {sources[i][:50]}...")
        print(f"      pred: {predictions[i][:50]}..." if predictions[i] else "      pred: (empty)")
        print()
    
    # compute metrics
    print("\nLoading COMET model")
    comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    
    print("Computing corpus-level metrics")
    metrics = compute_metrics(predictions, references, sources, comet_model)
    
    print("Computing per-example metrics")
    bleu_scores, comet_scores = compute_per_example_metrics(predictions, references, sources, comet_model)
    
    # print results
    print(f"\nResults:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # save results
    results_df = pd.DataFrame({
        "source": sources,
        "reference": references,
        "prediction": predictions,
        "bleu": [round(s, 2) for s in bleu_scores],
        "comet": [round(s, 4) for s in comet_scores]
    })
    
    suffix = f"_sample{config.SAMPLE_SIZE}" if config.SAMPLE_SIZE else ""
    
    results_path = Path(config.OUTPUT_DIR) / f"results{suffix}.csv"
    results_df.to_csv(results_path, index=False)
    
    summary = {
        "config": {
            "model_type": config.MODEL_TYPE,
            "model_path": config.BASE_MODEL,
            "lang_pair": config.LANG_PAIR,
            "test_data": config.TEST_DATA,
            "n_samples": len(test_df),
            "batch_size": config.BATCH_SIZE,
            "max_new_tokens": config.MAX_NEW_TOKENS
        },
        "metrics": metrics,
        "timing": {
            "generation_seconds": round(gen_time, 1),
            "examples_per_second": round(len(sources)/gen_time, 1)
        }
    }
    
    summary_path = Path(config.OUTPUT_DIR) / f"summary{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()