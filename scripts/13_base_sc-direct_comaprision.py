#!/usr/bin/env python3

import pandas as pd
import json
from pathlib import Path


class Config:
    MODEL_TYPE = "llama2"  # qwen, mistral, llama2, llama3
    LANG_PAIR = "de_en"  # zh_en, de_en
    USE_END_TOKEN = False
    
    SUFFIX = "_end" if USE_END_TOKEN else ""
    
    BASE_RESULTS = f"../evaluations/base/{MODEL_TYPE}_{LANG_PAIR}/results.csv"
    SC_RESULTS = f"../evaluations/sc_direct/{MODEL_TYPE}_{LANG_PAIR}{SUFFIX}/results.csv"
    OUTPUT_DIR = f"../evaluations/comparison_direct/{MODEL_TYPE}_{LANG_PAIR}{SUFFIX}"


def main():
    config = Config()
    
    print(f"Base vs Direct SC comparison")
    print(f"Model: {config.MODEL_TYPE}, Lang: {config.LANG_PAIR}")
    
    # load results
    print(f"\nLoading base results from {config.BASE_RESULTS}")
    base_df = pd.read_csv(config.BASE_RESULTS)
    
    print(f"Loading SC results from {config.SC_RESULTS}")
    sc_df = pd.read_csv(config.SC_RESULTS)
    
    # sanity check
    if len(base_df) != len(sc_df):
        print(f"problem: row count mismatch, base={len(base_df)}, SC={len(sc_df)}")
        min_rows = min(len(base_df), len(sc_df))
        base_df = base_df.head(min_rows)
        sc_df = sc_df.head(min_rows)
    
    # check if sources match
    if "source" in base_df.columns and "source" in sc_df.columns:
        mismatches = (base_df["source"] != sc_df["source"]).sum()
        if mismatches > 0:
            print(f"problem: {mismatches} source mismatches")
    
    print(f"Base columns: {list(base_df.columns)}")
    print(f"SC columns: {list(sc_df.columns)}")
    
    # extract data
    sources = sc_df["source"].tolist()
    references = sc_df["reference"].tolist()
    
    base_predictions = base_df["prediction"].fillna("").tolist()
    sc_initial = sc_df["initial_translation"].fillna("").tolist()
    sc_corrected = sc_df["corrected_translation"].fillna("").tolist()
    
    sc_final = []
    for init, corr in zip(sc_initial, sc_corrected):
        sc_final.append(corr if corr else init)
    
    # get scores
    base_bleu_scores = base_df["bleu"].tolist()
    base_comet_scores = base_df["comet"].tolist()
    sc_initial_bleu = sc_df["initial_bleu"].tolist()
    sc_final_bleu = sc_df["final_bleu"].tolist()
    sc_initial_comet = sc_df["initial_comet"].tolist()
    sc_final_comet = sc_df["final_comet"].tolist()
    
    # compare base vs sc final belu
    sc_better_bleu = 0
    sc_worse_bleu = 0
    similar_bleu = 0
    bleu_diffs = []
    
    for i in range(len(base_df)):
        base_b = base_bleu_scores[i] if base_bleu_scores[i] else 0
        sc_b = sc_final_bleu[i] if sc_final_bleu[i] else 0
        
        diff = sc_b - base_b
        bleu_diffs.append(diff)
        
        if diff > 0.1:
            sc_better_bleu += 1
        elif diff < -0.1:
            sc_worse_bleu += 1
        else:
            similar_bleu += 1
    
    # compare base vs sc final comet
    sc_better_comet = 0
    sc_worse_comet = 0
    similar_comet = 0
    comet_diffs = []
    
    for i in range(len(base_df)):
        base_c = base_comet_scores[i] if base_comet_scores[i] else 0
        sc_c = sc_final_comet[i] if sc_final_comet[i] else 0
        
        diff = sc_c - base_c
        comet_diffs.append(diff)
        
        if diff > 0.001:
            sc_better_comet += 1
        elif diff < -0.001:
            sc_worse_comet += 1
        else:
            similar_comet += 1
    
    n = len(base_df)
    print(f"\nPer-example BLEU comparison (SC final vs Base):")
    print(f"  SC better: {sc_better_bleu}/{n} ({100*sc_better_bleu/n:.1f}%)")
    print(f"  SC worse: {sc_worse_bleu}/{n} ({100*sc_worse_bleu/n:.1f}%)")
    print(f"  Similar: {similar_bleu}/{n} ({100*similar_bleu/n:.1f}%)")
    print(f"  Avg BLEU diff: {sum(bleu_diffs)/len(bleu_diffs):.2f}")
    
    print(f"\nPer-example COMET comparison (SC final vs Base):")
    print(f"  SC better: {sc_better_comet}/{n} ({100*sc_better_comet/n:.1f}%)")
    print(f"  SC worse: {sc_worse_comet}/{n} ({100*sc_worse_comet/n:.1f}%)")
    print(f"  Similar: {similar_comet}/{n} ({100*similar_comet/n:.1f}%)")
    print(f"  Avg COMET diff: {sum(comet_diffs)/len(comet_diffs):.4f}")
    
    # corpus level from summaries
    base_summary_path = Path(config.BASE_RESULTS).parent / "summary.json"
    sc_summary_path = Path(config.SC_RESULTS).parent / "summary.json"
    
    if base_summary_path.exists() and sc_summary_path.exists():
        with open(base_summary_path) as f:
            base_summary = json.load(f)
        with open(sc_summary_path) as f:
            sc_summary = json.load(f)
        
        base_metrics = base_summary.get("metrics", {})
        sc_init_metrics = sc_summary.get("initial_metrics", {})
        sc_final_metrics = sc_summary.get("final_metrics", {})
        
        print(f"\nCorpus level metrics:")
        print(f"{'Metric':<8} {'Base':>8} {'SC Init':>10} {'SC Final':>10} {'Diff':>8}")
        
        for metric in ["BLEU", "chrF", "TER", "COMET"]:
            base_val = base_metrics.get(metric, "-")
            sc_init = sc_init_metrics.get(metric, "-")
            sc_final = sc_final_metrics.get(metric, "-")
            
            if isinstance(base_val, (int, float)) and isinstance(sc_final, (int, float)):
                diff = f"{sc_final - base_val:+.2f}"
            else:
                diff = "-"
            
            print(f"{metric:<8} {str(base_val):>8} {str(sc_init):>10} {str(sc_final):>10} {diff:>8}")
    
    # save 
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    comparison_df = pd.DataFrame({
        "source": sources,
        "reference": references,
        "base_prediction": base_predictions,
        "sc_initial": sc_initial,
        "sc_corrected": sc_corrected,
        "sc_final": sc_final,
        "base_bleu": base_bleu_scores,
        "sc_initial_bleu": sc_initial_bleu,
        "sc_final_bleu": sc_final_bleu,
        "bleu_diff": bleu_diffs,
        "base_comet": base_comet_scores,
        "sc_initial_comet": sc_initial_comet,
        "sc_final_comet": sc_final_comet,
        "comet_diff": comet_diffs
    })
    
    comparison_df.to_csv(Path(config.OUTPUT_DIR) / "comparison.csv", index=False)
    
    summary = {
        "config": {
            "model_type": config.MODEL_TYPE,
            "lang_pair": config.LANG_PAIR,
            "use_end_token": config.USE_END_TOKEN
        },
        "bleu_comparison": {
            "sc_better": sc_better_bleu,
            "sc_worse": sc_worse_bleu,
            "similar": similar_bleu,
            "sc_better_pct": round(100 * sc_better_bleu / n, 1),
            "sc_worse_pct": round(100 * sc_worse_bleu / n, 1),
            "avg_diff": round(sum(bleu_diffs) / len(bleu_diffs), 2)
        },
        "comet_comparison": {
            "sc_better": sc_better_comet,
            "sc_worse": sc_worse_comet,
            "similar": similar_comet,
            "sc_better_pct": round(100 * sc_better_comet / n, 1),
            "sc_worse_pct": round(100 * sc_worse_comet / n, 1),
            "avg_diff": round(sum(comet_diffs) / len(comet_diffs), 4)
        }
    }
    
    with open(Path(config.OUTPUT_DIR) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()