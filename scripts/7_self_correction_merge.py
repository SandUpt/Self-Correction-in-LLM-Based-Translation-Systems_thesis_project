#!/usr/bin/env python3

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class Config:
    MODEL_TYPE = "llama3"  # qwen, mistral, llama2, llama3
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
    
    # auto-configured
    USE_HF_TOKEN = NEEDS_HF_TOKEN[MODEL_TYPE]
    USE_REMOTE_CODE = TRUST_REMOTE_CODE[MODEL_TYPE]
    
    SUFFIX = "_end" if USE_END_TOKEN else ""
    WMT_MODEL_PATH = f"../models/wmt_merged/{MODEL_TYPE}_{LANG_PAIR}"
    ADAPTER_PATH = f"../models/self_correction/{MODEL_TYPE}_{LANG_PAIR}{SUFFIX}_70_30"
    OUTPUT_PATH = f"../models/self_correction_merged/{MODEL_TYPE}_{LANG_PAIR}{SUFFIX}_70_30"


def main():
    config = Config()

    print(f"Merging Self-Correction Adapter")
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Language Pair: {config.LANG_PAIR}")
    print(f"END token version: {config.USE_END_TOKEN}")
    
    hf_token = os.getenv("HF_TOKEN") if config.USE_HF_TOKEN else None
    
    # load WMT model (base for self-correction)
    print(f"\nLoading WMT model: {config.WMT_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.WMT_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=config.USE_REMOTE_CODE
    )
    
    print(f"Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        config.WMT_MODEL_PATH,
        token=hf_token,
        trust_remote_code=config.USE_REMOTE_CODE
    )
    
    # load adapter
    print(f"Loading adapter from: {config.ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, config.ADAPTER_PATH)
    
    # merge
    print(f"Merging adapter weights")
    merged_model = model.merge_and_unload()
    
    # save
    print(f"Saving merged model to: {config.OUTPUT_PATH}")
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    merged_model.save_pretrained(config.OUTPUT_PATH)
    tokenizer.save_pretrained(config.OUTPUT_PATH)
    
    print(f"\nMerge complete.")
    print(f"Merged model saved to: {config.OUTPUT_PATH}")


if __name__ == "__main__":
    main()