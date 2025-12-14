#!/usr/bin/env python3

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


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
    
    # auto-configured
    BASE_MODEL = MODEL_PATHS[MODEL_TYPE]
    USE_HF_TOKEN = NEEDS_HF_TOKEN[MODEL_TYPE]
    USE_REMOTE_CODE = TRUST_REMOTE_CODE[MODEL_TYPE]
    ADAPTER_PATH = f"../models/wmt_adapters/{MODEL_TYPE}_{LANG_PAIR}_4M_Lora"
    OUTPUT_PATH = f"../models/wmt_merged/{MODEL_TYPE}_{LANG_PAIR}_4M_Lora"


def main():
    config = Config()
    
    print(f"Merging LoRA Adapter")
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Language Pair: {config.LANG_PAIR}")
    
    hf_token = os.getenv("HF_TOKEN") if config.USE_HF_TOKEN else None
    
    # load base model
    print(f"\nLoading base model: {config.BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=config.USE_REMOTE_CODE
    )
    
    # load tokenizer
    print(f"Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL,
        token=hf_token,
        trust_remote_code=config.USE_REMOTE_CODE
    )
    
    # load adapter
    print(f"Loading adapter from: {config.ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, config.ADAPTER_PATH)
    
    # merge weights
    print(f"Merging adapter weights")
    merged_model = model.merge_and_unload()
    
    # save merged model
    print(f"Saving merged model to: {config.OUTPUT_PATH}")
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)
    merged_model.save_pretrained(config.OUTPUT_PATH)
    tokenizer.save_pretrained(config.OUTPUT_PATH)
    
    print(f"\nMerge complete.")
    print(f"Merged model saved to: {config.OUTPUT_PATH}")


if __name__ == "__main__":
    main()