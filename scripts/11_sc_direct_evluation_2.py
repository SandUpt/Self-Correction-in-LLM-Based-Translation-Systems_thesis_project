#!/usr/bin/env python3

import os
import re
import time
import torch
import pandas as pd
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF, TER
from comet import download_model, load_from_checkpoint


class Config:
    MODEL_TYPE = "qwen"  # qwen, mistral, llama2, llama3
    LANG_PAIR = "de_en"  # zh_en, de_en
    USE_END_TOKEN = False  
    
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
        "zh_en": "Translate the following Chinese to English:\n{src}",
        "de_en": "Translate the following German to English:\n{src}"
    }
    
    TEST_DATA_PATHS = {
        "zh_en": "../../data/evaluation_sets/zh_en/test_5000_clean.tsv",
        "de_en": "../../data/evaluation_sets/de_en/test_5000_clean.tsv"
    }
    
    # auto-configured
    USE_HF_TOKEN = NEEDS_HF_TOKEN[MODEL_TYPE]
    USE_REMOTE_CODE = TRUST_REMOTE_CODE[MODEL_TYPE]
    PROMPT_TEMPLATE = PROMPT_TEMPLATES[LANG_PAIR]
    TEST_DATA = TEST_DATA_PATHS[LANG_PAIR]
    
    SUFFIX = "_end" if USE_END_TOKEN else ""
    MODEL_PATH = f"../models/sc_direct_merged/{MODEL_TYPE}_{LANG_PAIR}{SUFFIX}"
    OUTPUT_DIR = f"../evaluations/sc_direct/{MODEL_TYPE}_{LANG_PAIR}{SUFFIX}_eval_2"
    
    SAMPLE_SIZE = None
    MAX_NEW_TOKENS = 400
    BATCH_SIZE = 32
    SEED = 42


class StopOnEnd(StoppingCriteria):
    END_VARIANTS = (
        "|||END|||", " |||END|||", "\n|||END|||",
        "||| END |||", " ||| END |||", "\n||| END |||"
    )
    
    def __init__(self, tokenizer):
        self.seqs = [tokenizer.encode(v, add_special_tokens=False) for v in self.END_VARIANTS]
        self.max_len = max(len(s) for s in self.seqs)
    
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < self.max_len:
            return False
        tail = input_ids[0, -self.max_len:].tolist()
        for seq in self.seqs:
            if len(seq) <= len(tail) and tail[-len(seq):] == seq:
                return True
        return False


def load_test_data(path, sample_size=None):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if "source_zh" in df.columns:
        df = df.rename(columns={"source_zh": "source", "target_en": "reference"})
    elif "source_de" in df.columns:
        df = df.rename(columns={"source_de": "source", "target_en": "reference"})
    elif "src" in df.columns:
        df = df.rename(columns={"src": "source", "ref": "reference"})
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=Config.SEED).reset_index(drop=True)
    
    return df


def extract_translations(text):
    result = {
        "initial": None,
        "analysis": None,
        "corrected": None,
        "raw": text
    }
    
    if not text:
        return result
    
    # cut at continuation patterns
    stop_patterns = ["Human:", "You are an AI assistant", "User:", "Assistant:", "<|im_start|>", "<|im_end|>"]
    for pattern in stop_patterns:
        idx = text.find(pattern)
        if idx > 0:
            text = text[:idx].strip()
    
    first_initial = text.find("Initial translation:")
    first_analysis = text.find("Analysis:")
    first_corrected = text.find("Corrected translation:")
    
    if first_initial == -1:
        return result
    
    # repetition detection
    second_initial = text.find("Initial translation:", first_initial + 1)
    second_analysis = text.find("Analysis:", first_analysis + 1) if first_analysis != -1 else -1
    second_corrected = text.find("Corrected translation:", first_corrected + 1) if first_corrected != -1 else -1
    
    repetition_points = [p for p in [second_initial, second_analysis, second_corrected] if p != -1]
    if repetition_points:
        text = text[:min(repetition_points)].strip()
    
    # extract parts
    if "Initial translation:" in text:
        start = text.index("Initial translation:") + len("Initial translation:")
        end = text.find("Analysis:", start)
        if end == -1:
            end = text.find("Corrected translation:", start)
        if end != -1:
            result["initial"] = text[start:end].strip()
        else:
            result["initial"] = text[start:].strip()
    
    if "Analysis:" in text:
        start = text.index("Analysis:") + len("Analysis:")
        end = text.find("Corrected translation:", start)
        if end != -1:
            result["analysis"] = text[start:end].strip()
        else:
            result["analysis"] = text[start:].strip()
    
    if "Corrected translation:" in text:
        start = text.index("Corrected translation:") + len("Corrected translation:")
        result["corrected"] = text[start:].strip()
    
    return result


def clean_translation(text):
    if not text:
        return ""
    
    text = text.strip()
    
    pipe_idx = text.find("|||")
    if pipe_idx > 0:
        text = text[:pipe_idx].strip()
    
    pipe_idx = text.find("||")
    if pipe_idx > 0:
        text = text[:pipe_idx].strip()
    
    text = text.rstrip("|").strip()
    text = re.sub(r'<\|.*?\|>', '', text)
    
    # sentence repetition detection
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) > 1:
        seen = set()
        unique = []
        for sent in sentences:
            norm = sent.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(sent.strip())
            elif norm in seen:
                break
        text = ' '.join(unique)
    
    # phrase repetition detection
    words = text.split()
    if len(words) > 12:
        for phrase_len in range(6, min(15, len(words)//2)):
            for i in range(len(words) - phrase_len * 2):
                phrase = ' '.join(words[i:i+phrase_len])
                rest = ' '.join(words[i+phrase_len:])
                if phrase in rest:
                    text = ' '.join(words[:i+phrase_len])
                    break
            else:
                continue
            break
    
    text = text.split("\n")[0].strip()
    return text


def generate_outputs(model, tokenizer, sources, prompt_template):
    model.eval()
    outputs_list = []
    stop_criteria = StoppingCriteriaList([StopOnEnd(tokenizer)])
    
    for i in tqdm(range(0, len(sources), Config.BATCH_SIZE), desc="Generating"):
        batch_sources = sources[i:i + Config.BATCH_SIZE]
        prompts = [prompt_template.format(src=src) for src in batch_sources]
        
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
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stop_criteria
            )
        
        for output in outputs:
            generated = tokenizer.decode(
                output[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            outputs_list.append(generated)
        
        del inputs, outputs
        torch.cuda.empty_cache()
    
    return outputs_list


def compute_metrics(predictions, references, sources, comet_model):
    bleu = BLEU()
    chrf = CHRF()
    ter = TER()
    
    bleu_score = bleu.corpus_score(predictions, [references]).score
    chrf_score = chrf.corpus_score(predictions, [references]).score
    ter_score = ter.corpus_score(predictions, [references]).score
    
    comet_data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
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
    
    comet_data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
    comet_output = comet_model.predict(comet_data, batch_size=64, gpus=1, progress_bar=False)
    comet_scores = comet_output.scores
    
    return bleu_scores, comet_scores


def main():
    config = Config()
    
    print(f"Direct SC Model Evaluation Base to SC")
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Language Pair: {config.LANG_PAIR}")
    print(f"END token version: {config.USE_END_TOKEN}")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(config.SEED)
    
    print(f"\nLoading test data from {config.TEST_DATA}")
    test_df = load_test_data(config.TEST_DATA, config.SAMPLE_SIZE)
    print(f"Test examples: {len(test_df)}")
    
    print(f"\nLoading model: {config.MODEL_PATH}")
    hf_token = os.getenv("HF_TOKEN") if config.USE_HF_TOKEN else None
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_PATH,
        token=hf_token,
        trust_remote_code=config.USE_REMOTE_CODE
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=config.USE_REMOTE_CODE
    )
    
    print("\nGenerating")
    sources = test_df["source"].tolist()
    references = test_df["reference"].tolist()
    
    start_time = time.time()
    raw_outputs = generate_outputs(model, tokenizer, sources, config.PROMPT_TEMPLATE)
    gen_time = time.time() - start_time
    print(f"Generation took {gen_time:.1f}s")
    
    print("\nExtracting translations")
    results = []
    for output in raw_outputs:
        extracted = extract_translations(output)
        results.append({
            "initial": clean_translation(extracted["initial"]),
            "analysis": extracted["analysis"],
            "corrected": clean_translation(extracted["corrected"]),
            "raw": output
        })
    
    initial_translations = [r["initial"] or "" for r in results]
    final_translations = [r["corrected"] or r["initial"] or "" for r in results]
    
    # stats
    failed = sum(1 for r in results if not r["initial"])
    corrections = sum(1 for r in results if r["corrected"] and r["initial"] and r["corrected"] != r["initial"])
    print(f"Failed extractions: {failed}/{len(results)}")
    print(f"Made corrections: {corrections}/{len(results)}")
    
    print("\nSample outputs:")
    for i in range(min(3, len(results))):
        print(f"  [{i}] initial: {initial_translations[i][:60]}...")
        print(f"      final:   {final_translations[i][:60]}...")
        print()
    
    print("Loading COMET")
    comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    
    print("Computing metrics")
    initial_metrics = compute_metrics(initial_translations, references, sources, comet_model)
    final_metrics = compute_metrics(final_translations, references, sources, comet_model)
    
    initial_bleu, initial_comet = compute_per_example_metrics(initial_translations, references, sources, comet_model)
    final_bleu, final_comet = compute_per_example_metrics(final_translations, references, sources, comet_model)
    
    # improvement analysis
    bleu_improved = sum(1 for i, f in zip(initial_bleu, final_bleu) if f > i + 0.1)
    bleu_degraded = sum(1 for i, f in zip(initial_bleu, final_bleu) if f < i - 0.1)
    comet_improved = sum(1 for i, f in zip(initial_comet, final_comet) if f > i + 0.001)
    comet_degraded = sum(1 for i, f in zip(initial_comet, final_comet) if f < i - 0.001)
    
    
    print("Initial Translation Metrics")
    
    for k, v in initial_metrics.items():
        print(f"  {k}: {v}")
    
    
    print("Final Translation Metrics")
    
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")
    
    
    print("Self-Correction Analysis")
    
    print(f"  BLEU improved:  {bleu_improved}/{len(results)} ({100*bleu_improved/len(results):.1f}%)")
    print(f"  BLEU degraded:  {bleu_degraded}/{len(results)} ({100*bleu_degraded/len(results):.1f}%)")
    print(f"  COMET improved: {comet_improved}/{len(results)} ({100*comet_improved/len(results):.1f}%)")
    print(f"  COMET degraded: {comet_degraded}/{len(results)} ({100*comet_degraded/len(results):.1f}%)")
    
    # save
    results_df = pd.DataFrame({
        "source": sources,
        "reference": references,
        "initial_translation": initial_translations,
        "corrected_translation": [r["corrected"] or "" for r in results],
        "initial_bleu": [round(s, 2) for s in initial_bleu],
        "final_bleu": [round(s, 2) for s in final_bleu],
        "bleu_change": [round(f - i, 2) for i, f in zip(initial_bleu, final_bleu)],
        "initial_comet": [round(s, 4) for s in initial_comet],
        "final_comet": [round(s, 4) for s in final_comet],
        "comet_change": [round(f - i, 4) for i, f in zip(initial_comet, final_comet)],
        "raw_output": [r["raw"] for r in results]
    })
    
    results_df.to_csv(Path(config.OUTPUT_DIR) / "results.csv", index=False)
    
    summary = {
        "config": {
            "model_type": config.MODEL_TYPE,
            "lang_pair": config.LANG_PAIR,
            "training_path": "Base to SC direct",
            "n_samples": len(test_df)
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "analysis": {
            "failed_extractions": failed,
            "corrections_made": corrections,
            "bleu_improved": bleu_improved,
            "bleu_degraded": bleu_degraded,
            "comet_improved": comet_improved,
            "comet_degraded": comet_degraded
        }
    }
    
    with open(Path(config.OUTPUT_DIR) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()