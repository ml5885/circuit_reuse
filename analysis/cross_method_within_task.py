"""Cross-method comparison of within-task reuse and lift.

For each model and K, compute the mean reuse@P and mean lift across the six
tasks, separately for EAP, EAP-IG, and RelP. Plot one panel per model, with
three colored lines (one per method).

Two figures:
  * cross_method_reuse.png  -- mean reuse percentage vs K
  * cross_method_lift.png   -- mean lift vs K
"""

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.use("Agg")
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

REPO = Path("/Users/michaelli/Desktop/Research/circuit_reuse")
OUT_DIR = REPO / "results2" / "cross_method"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["addition", "arc_challenge", "arc_easy", "boolean", "ioi", "mcqa"]

MODELS = [
    "google_gemma-2-2b",
    "google_gemma-2-2b-it",
    "meta-llama_Llama-3.2-3B",
    "meta-llama_Llama-3.2-3B-Instruct",
    "qwen3-4b",
    "qwen3-8b",
]
MODEL_DISPLAY = {
    "google_gemma-2-2b": "Gemma 2 2B",
    "google_gemma-2-2b-it": "Gemma 2 2B IT",
    "meta-llama_Llama-3.2-3B": "Llama-3.2-3B",
    "meta-llama_Llama-3.2-3B-Instruct": "Llama-3.2-3B Inst",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
}

METHODS = {
    "EAP":    REPO / "results" / "cross_task",
    "EAP-IG": REPO / "results_eap_ig" / "cross_task_eap_ig",
    "RelP":   REPO / "results_relp_new" / "cross_task_relp",
}
METHOD_COLOR = {
    "EAP":    "#E76F51",
    "EAP-IG": "#2A9D8F",
    "RelP":   "#5DA9C9",
}

KS = [1, 5, 10, 20, 30]
P_TARGET = 95


def find_metrics(method_dir: Path, model: str, task: str):
    """Locate the metrics.json for one (method, model, task)."""
    pattern = re.compile(rf"^{re.escape(model)}__main__{re.escape(task)}__")
    for sub in method_dir.iterdir():
        if not sub.is_dir():
            continue
        if pattern.match(sub.name):
            mp = sub / "metrics.json"
            if mp.exists():
                return mp
    return None


def get_metric_at(metrics_path: Path, K: int, P: int, what: str):
    """Return (reuse, lift) for given K and P. lift uses val split."""
    with open(metrics_path) as f:
        d = json.load(f)
    by_k = d.get("by_k", {}).get(str(K))
    if by_k is None:
        return None
    thr = by_k.get("thresholds", {}).get(str(P))
    if thr is None:
        return None
    if what == "reuse":
        return float(thr["reuse_percent"])
    if what == "lift":
        base = d.get("baseline_val_accuracy")
        ab = thr["val"].get("ablation_accuracy")
        ct = thr["val"].get("control_accuracy")
        if base is None or ab is None or ct is None or base == 0:
            return None
        return (ct - ab) / base
    raise ValueError(what)


def collect(method_dir: Path, model: str, what: str, K: int, P: int):
    """Return list of per-task values for (model, K, P)."""
    vals = []
    for task in TASKS:
        mp = find_metrics(method_dir, model, task)
        if mp is None:
            continue
        v = get_metric_at(mp, K, P, what)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        vals.append(v)
    return vals


def t_ci95(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return float("nan")
    return stats.t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr))


def make_plot(what: str, ylabel: str, title: str | None, out_path: Path,
              ylim=None, yticks=None):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharey=True)
    axes = axes.reshape(2, 3)

    methods = list(METHODS.keys())
    n_methods = len(methods)
    w = 0.27

    for idx, model in enumerate(MODELS):
        r, c = idx // 3, idx % 3
        ax = axes[r, c]

        # Compute per-method means and CIs for every K
        all_means = {m: [] for m in methods}
        all_errs = {m: [] for m in methods}
        for method, method_dir in METHODS.items():
            for K in KS:
                vals = collect(method_dir, model, what, K, P_TARGET)
                if not vals:
                    all_means[method].append(np.nan)
                    all_errs[method].append(np.nan)
                    continue
                all_means[method].append(float(np.mean(vals)))
                all_errs[method].append(t_ci95(vals))

        x = np.arange(len(KS))
        for i, method in enumerate(methods):
            means = np.asarray(all_means[method], dtype=float)
            errs = np.asarray(all_errs[method], dtype=float)
            errs = np.where(np.isnan(errs), 0.0, errs)
            offset = (i - (n_methods - 1) / 2) * w
            lo = ylim[0] if ylim is not None else -np.inf
            hi = ylim[1] if ylim is not None else np.inf
            room_below = np.maximum(means - lo, 0.0)
            room_above = np.maximum(hi - means, 0.0)
            lower = np.minimum(errs, room_below)
            upper = np.minimum(errs, room_above)
            yerr = np.vstack([lower, upper])
            ax.bar(
                x + offset, means, w,
                yerr=yerr, capsize=2.5,
                color=METHOD_COLOR[method], label=method,
                edgecolor="white", linewidth=0.5,
                error_kw={"elinewidth": 1.0, "ecolor": "#444"},
                zorder=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{k}%" for k in KS])
        ax.set_title(MODEL_DISPLAY[model], fontsize=12)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":")
        ax.set_axisbelow(True)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
        if c == 0:
            ax.set_ylabel(ylabel)
        if r == 1:
            ax.set_xlabel("Circuit size $K$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.04),
        ncol=3, fontsize=12,
    )
    if title:
        fig.suptitle(title, fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    make_plot(
        what="reuse",
        ylabel=f"Reuse @ $P={P_TARGET}\\%$",
        title=None,
        out_path=OUT_DIR / "cross_method_reuse.png",
        ylim=(0, 105),
        yticks=np.arange(0, 101, 20),
    )

    make_plot(
        what="lift",
        ylabel=f"Lift @ $P={P_TARGET}\\%$",
        title=None,
        out_path=OUT_DIR / "cross_method_lift.png",
        ylim=(-0.05, 1.05),
        yticks=np.arange(0, 1.01, 0.2),
    )


if __name__ == "__main__":
    main()
