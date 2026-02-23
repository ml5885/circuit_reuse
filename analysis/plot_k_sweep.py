import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from circuit_reuse.dataset import get_task_display_name, get_model_display_name

def discover_metrics(results_dir):
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("metrics.json"))

def load_metrics_json(path):
    with path.open("r") as f:
        return json.load(f)

def _expand_v2(r):
    rows = []
    base = {
        "model_name": r["model_name"],
        "hf_revision": r["hf_revision"],
        "task": r["task"],
        "method": r["method"],
    }
    
    by_k = r["by_k"]
    for k_str, block in by_k.items():
        K = int(k_str)
            
        thresholds = block["thresholds"]
        for p_str, tblock in thresholds.items():
            P = int(p_str)
                
            tr = tblock["train"]
            va = tblock["val"]
            
            def compute_lift_val(ablation, control, baseline):
                if baseline == 0: return float('nan')
                return (ablation - control) / baseline

            lift_train = compute_lift_val(
                tr["ablation_accuracy"],
                tr["control_accuracy"],
                r["baseline_train_accuracy"]
            )
            lift_val = compute_lift_val(
                va["ablation_accuracy"],
                va["control_accuracy"],
                r["baseline_val_accuracy"]
            )

            row = dict(base)
            row.update({
                "top_k": K,
                "reuse_threshold": P,
                "reuse_percent": tblock["reuse_percent"],
                "lift_train": lift_train,
                "lift_val": lift_val,
            })
            rows.append(row)
    return rows

def aggregate(paths):
    expanded = []
    for p in paths:
        r = load_metrics_json(p)
        if "by_k" in r:
            expanded.extend(_expand_v2(r))
    return pd.DataFrame(expanded) if expanded else pd.DataFrame()

def plot_k_sweep(df, out_dir, threshold=100):
    df = df[df["reuse_threshold"] == threshold].copy()
    if df.empty:
        print(f"[WARN] No data for threshold {threshold}")
        return

    df["task_display"] = df["task"].apply(get_task_display_name)
    df["model_display"] = df["model_name"].apply(get_model_display_name)
    
    sns.set_theme(style="whitegrid")
    
    tasks = df["task_display"].unique()
    for task in tasks:
        task_df = df[df["task_display"] == task]
        
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=task_df, x="top_k", y="lift_val", hue="model_display", marker="o")
        plt.title(f"Lift vs Top-K Components ({task}, reuse@{threshold})")
        plt.xlabel("Top-K Components")
        plt.ylabel("Lift (Val)")
        plt.axhline(0, color='gray', linestyle='--')
        
        out_path = out_dir / f"k_sweep_lift_{task.replace(' ', '_')}_reuse{threshold}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Saved {out_path}")

    for task in tasks:
        task_df = df[df["task_display"] == task]
        
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=task_df, x="top_k", y="reuse_percent", hue="model_display", marker="o")
        plt.title(f"Reuse % vs Top-K Components ({task}, reuse@{threshold})")
        plt.xlabel("Top-K Components")
        plt.ylabel("Reuse %")
        
        out_path = out_dir / f"k_sweep_reuse_{task.replace(' ', '_')}_reuse{threshold}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="plots_k_sweep")
    parser.add_argument("--threshold", type=int, default=100)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_metrics(results_dir)
    df = aggregate(paths)
    
    if df.empty:
        print("No data found.")
        return

    plot_k_sweep(df, out_dir, args.threshold)

if __name__ == "__main__":
    main()
