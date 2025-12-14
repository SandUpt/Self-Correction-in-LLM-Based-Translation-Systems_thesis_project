#!/usr/bin/env python3

import pandas as pd
from pathlib import Path


class Config:
    LANG_PAIR = "de_en"  # zh_en, de_en
    
    INPUT_PATHS = {
        "zh_en": {
            "train": "../../data/processed/zh_en/self_correction_zh_en/train.tsv",
            "val": "../../data/processed/zh_en/self_correction_zh_en/val.tsv"
        },
        "de_en": {
            "train": "../../data/processed/de_en/self_correction_de_en/train.tsv",
            "val": "../../data/processed/de_en/self_correction_de_en/val.tsv"
        }
    }
    
    OUTPUT_PATHS = {
        "zh_en": {
            "train": "../../data/processed/zh_en/self_correction_zh_en_end/train.tsv",
            "val": "../../data/processed/zh_en/self_correction_zh_en_end/val.tsv"
        },
        "de_en": {
            "train": "../../data/processed/de_en/self_correction_de_en_end/train.tsv",
            "val": "../../data/processed/de_en/self_correction_de_en_end/val.tsv"
        }
    }
    
    END_TOKEN = "|||END|||"


def add_end_token(input_path, output_path, end_token):
    """Add end token to completion column"""
    df = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
    
    # completion column ('completion' or 'output')
    completion_col = None
    for col in df.columns:
        if col.lower() in ["completion", "output"]:
            completion_col = col
            break
    
    if completion_col is None:
        # assume second column is completion
        completion_col = df.columns[1]
    
    # add end token if not already present
    def add_end(text):
        text = text.strip()
        if not text.endswith(end_token):
            return text + end_token
        return text
    
    df[completion_col] = df[completion_col].apply(add_end)
    
    # save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    
    return len(df)


def main():
    config = Config()
    
    print(f"Adding |||END||| to Self-Correction Data")
    print(f"Language Pair: {config.LANG_PAIR}")
    
    input_paths = config.INPUT_PATHS[config.LANG_PAIR]
    output_paths = config.OUTPUT_PATHS[config.LANG_PAIR]
    
    # process train
    print(f"\nProcessing train")
    print(f"  Input:  {input_paths['train']}")
    print(f"  Output: {output_paths['train']}")
    n_train = add_end_token(input_paths["train"], output_paths["train"], config.END_TOKEN)
    print(f"  Processed {n_train} examples")
    
    # process val
    print(f"\nProcessing val")
    print(f"  Input:  {input_paths['val']}")
    print(f"  Output: {output_paths['val']}")
    n_val = add_end_token(input_paths["val"], output_paths["val"], config.END_TOKEN)
    print(f"  Processed {n_val} examples")
    
    print(f"\nDone.")


if __name__ == "__main__":
    main()