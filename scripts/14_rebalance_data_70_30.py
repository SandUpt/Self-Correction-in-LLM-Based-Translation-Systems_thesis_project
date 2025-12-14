#!/usr/bin/env python3

import pandas as pd
from pathlib import Path


class Config:
    LANG_PAIR = "de_en"  # zh_en, de_en
    
    INPUT_TRAIN = f"../../data/processed/{LANG_PAIR}/self_correction_{LANG_PAIR}/train.tsv"
    INPUT_VAL = f"../../data/processed/{LANG_PAIR}/self_correction_{LANG_PAIR}/val.tsv"
    
    OUTPUT_TRAIN = f"../../data/processed/{LANG_PAIR}/self_correction_{LANG_PAIR}_70-30/train.tsv"
    OUTPUT_VAL = f"../../data/processed/{LANG_PAIR}/self_correction_{LANG_PAIR}_70-30/val.tsv"
    
    SEED = 42


def check_overlap(train_df, val_df):
    train_prompts = set(train_df["prompt"].values)
    val_prompts = set(val_df["prompt"].values)
    overlap = train_prompts & val_prompts
    
    print(f"Train/val overlap check:")
    print(f"  Train: {len(train_prompts)}, Val: {len(val_prompts)}, Overlap: {len(overlap)}")
    
    if len(overlap) > 0:
        print(f"  problem: {len(overlap)} examples in both train and val")
    
    return len(overlap)


def rebalance_train(train_df, target_error_pct=0.7):
    if "data_type" not in train_df.columns:
        print("No data_type column, skipping rebalance")
        return train_df
    
    error_df = train_df[train_df["data_type"] == "error_correction"].copy()
    clean_df = train_df[train_df["data_type"] == "clean_translation"].copy()
    
    n_error = len(error_df)
    n_clean = len(clean_df)
    total = len(train_df)
    
    print(f"Current: {n_error} error ({100*n_error/total:.1f}%), {n_clean} clean ({100*n_clean/total:.1f}%)")
    
    # keep error, downsample clean for 70-30
    needed_clean = int(n_error / target_error_pct * (1 - target_error_pct))
    
    if needed_clean > n_clean:
        print(f"Warning: need {needed_clean} clean but only have {n_clean}, using all")
        clean_sampled = clean_df
    else:
        clean_sampled = clean_df.sample(n=needed_clean, random_state=Config.SEED)
    
    error_sampled = error_df
    
    rebalanced = pd.concat([error_sampled, clean_sampled], ignore_index=True)
    rebalanced = rebalanced.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
    
    n_error_final = len(error_sampled)
    n_clean_final = len(clean_sampled)
    total_final = len(rebalanced)
    
    print(f"Final: {n_error_final} error ({100*n_error_final/total_final:.1f}%), {n_clean_final} clean ({100*n_clean_final/total_final:.1f}%), total {total_final}")
    
    return rebalanced


def main():
    config = Config()
    
    print(f"Rebalancing SC data for {config.LANG_PAIR}")
    
    # load
    train_df = pd.read_csv(config.INPUT_TRAIN, sep="\t", dtype=str, keep_default_na=False)
    val_df = pd.read_csv(config.INPUT_VAL, sep="\t", dtype=str, keep_default_na=False)
    print(f"Loaded {len(train_df)} train, {len(val_df)} val")
    
    # check overlap
    overlap_count = check_overlap(train_df, val_df)
    
    # rebalance train
    train_rebalanced = rebalance_train(train_df, target_error_pct=0.7)
    
    # save
    Path(config.OUTPUT_TRAIN).parent.mkdir(parents=True, exist_ok=True)
    train_rebalanced.to_csv(config.OUTPUT_TRAIN, sep="\t", index=False)
    val_df.to_csv(config.OUTPUT_VAL, sep="\t", index=False)
    
    print(f"Saved to {Path(config.OUTPUT_TRAIN).parent}")


if __name__ == "__main__":
    main()