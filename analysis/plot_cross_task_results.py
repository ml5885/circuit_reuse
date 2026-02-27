import argparse, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from circuit_reuse.dataset import get_task_display_name, get_model_display_name


def load_cross_task_jsons(results_dir):
    paths = sorted(results_dir.glob("cross_task_*.json"))
    data = []
    for p in paths:
        with p.open() as f:
            data.append(json.load(f))
    return data


def print_matrix(matrix, tasks, title, fmt=".1f"):
    task_labels = [get_task_display_name(t) for t in tasks]
    col_w = max(len(l) for l in task_labels) + 2
    header = "Source \\ Target".ljust(col_w) + "".join(l.rjust(col_w) for l in task_labels)
    print(f"\n{title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for src in tasks:
        row = get_task_display_name(src).ljust(col_w)
        for tgt in tasks:
            val = matrix[src][tgt]
            row += f"{val:{fmt}}".rjust(col_w)
        print(row)
    print()


def render_heatmap(
    matrix,
    tasks,
    title,
    ax,
    vmin=None,
    vmax=None,
    cmap="RdYlGn_r",
    fmt=".1f",
):
    n = len(tasks)
    arr = np.array([[matrix[src][tgt] for tgt in tasks] for src in tasks])
    labels = [get_task_display_name(t) for t in tasks]

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel("Target task")
    ax.set_ylabel("Source task (circuit ablated)")
    ax.set_title(title, fontsize=10)

    vmin_eff = float(np.nanmin(arr) if vmin is None else vmin)
    vmax_eff = float(np.nanmax(arr) if vmax is None else vmax)
    denom = (vmax_eff - vmin_eff) if (vmax_eff - vmin_eff) != 0 else 1.0

    for i in range(n):
        for j in range(n):
            norm = (arr[i, j] - vmin_eff) / denom
            color = "white" if norm > 0.6 else "black"
            ax.text(j, i, f"{arr[i, j]:{fmt}}", ha="center", va="center", fontsize=7, color=color)

    ax.grid(False)
    ax.grid(which="major", visible=False)

    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)

    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)

    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    return im


def plot_heatmaps(data, metric, metric_label, out_dir):
    all_vals = [d[metric][s][t] for d in data for s in d["tasks"] for t in d["tasks"]]
    if metric == "ablated_accuracy_pct":
        vmin, vmax, cmap = 0, 100, "viridis"
    else:
        vmin = min(0, min(all_vals))
        vmax = max(all_vals)
        cmap = "coolwarm"

    def safe_model_tag(model_name):
        return model_name.replace("/", "_").replace(" ", "_")

    out_subdir = out_dir / "heatmaps" if out_dir else None
    if out_subdir:
        out_subdir.mkdir(parents=True, exist_ok=True)

    for d in data:
        model_label = get_model_display_name(d["model_name"])
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = render_heatmap(d[metric], d["tasks"], model_label, ax, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label=metric_label)
        fig.tight_layout()

        if out_subdir:
            path = out_subdir / f"cross_task_heatmap_{metric}_{safe_model_tag(d['model_name'])}.png"
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Saved {path}")
        plt.close(fig)


def plot_heatmaps_multiplot(data, metric, metric_label, out_dir):
    if not out_dir:
        return

    all_vals = [d[metric][s][t] for d in data for s in d["tasks"] for t in d["tasks"]]
    if metric == "ablated_accuracy_pct":
        vmin, vmax, cmap = 0, 100, "viridis"
    else:
        vmin = min(0, min(all_vals))
        vmax = max(all_vals)
        cmap = "coolwarm"

    out_subdir = out_dir / "heatmaps"
    out_subdir.mkdir(parents=True, exist_ok=True)

    chunk_size = 6
    for chunk_idx in range(0, len(data), chunk_size):
        chunk = data[chunk_idx : chunk_idx + chunk_size]

        fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), squeeze=False)
        axes_flat = axes.flatten()

        im = None
        for i, d in enumerate(chunk):
            ax = axes_flat[i]
            model_label = get_model_display_name(d["model_name"])

            im = render_heatmap(d[metric], d["tasks"], model_label, ax, vmin=vmin, vmax=vmax, cmap=cmap)

            ax.set_title(model_label, fontsize=16, pad=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", which="major", labelsize=10)

            for lbl in ax.get_xticklabels():
                lbl.set_rotation(35)
                lbl.set_ha("right")

            row, col = divmod(i, 3)
            if row == 0:
                ax.tick_params(labelbottom=False)
            if col != 0:
                ax.tick_params(labelleft=False)

        for j in range(len(chunk), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.subplots_adjust(left=0.14, right=0.86, bottom=0.20, top=0.92, wspace=0.10, hspace=0.18)
        fig.supxlabel("Target task", fontsize=13, y=0.04)
        fig.supylabel("Source task (circuit ablated)", fontsize=13, x=0.02)

        if im is not None:
            cax = fig.add_axes([0.885, 0.22, 0.02, 0.58])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label(metric_label, fontsize=11)
            cb.ax.tick_params(labelsize=10)

        suffix = "" if len(data) <= chunk_size else f"_part{chunk_idx // chunk_size + 1}"
        path = out_subdir / f"cross_task_heatmaps_{metric}{suffix}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
        plt.close(fig)
        

def plot_diagonal_vs_offdiag(data, out_dir):
    def safe_model_tag(model_name):
        return model_name.replace("/", "_").replace(" ", "_")

    out_subdir = out_dir / "diagonal_vs_offdiag" if out_dir else None
    if out_subdir:
        out_subdir.mkdir(parents=True, exist_ok=True)

    for d in data:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        tasks = d["tasks"]
        drop = d["accuracy_drop_pp"]
        labels = [get_task_display_name(t) for t in tasks]

        diag = [drop[t][t] for t in tasks]
        offdiag = []
        for tgt in tasks:
            vals = [drop[src][tgt] for src in tasks if src != tgt]
            offdiag.append(np.mean(vals))

        x = np.arange(len(tasks))
        w = 0.35
        ax.bar(x - w / 2, diag, w, label="Own circuit", color="#4c78a8", edgecolor="none")
        ax.bar(x + w / 2, offdiag, w, label="Other circuits (mean)", color="#9ecae9", edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy drop (pp)")
        ax.set_title(get_model_display_name(d["model_name"]), fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
        ax.legend(fontsize=7, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.suptitle("Task-specificity: own-circuit vs. cross-circuit ablation damage", fontsize=12, y=1.02)
        fig.tight_layout()

        if out_subdir:
            path = out_subdir / f"cross_task_diagonal_vs_offdiag_{safe_model_tag(d['model_name'])}.png"
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Saved {path}")
        plt.close(fig)


def plot_diagonal_vs_offdiag_multiplot(data, out_dir):
    if not out_dir:
        return

    out_subdir = out_dir / "diagonal_vs_offdiag"
    out_subdir.mkdir(parents=True, exist_ok=True)

    chunk_size = 6
    for chunk_idx in range(0, len(data), chunk_size):
        chunk = data[chunk_idx : chunk_idx + chunk_size]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False, sharey=True)
        axes_flat = axes.flatten()

        for i, d in enumerate(chunk):
            ax = axes_flat[i]
            tasks = d["tasks"]
            drop = d["accuracy_drop_pp"]
            labels = [get_task_display_name(t) for t in tasks]

            diag = [drop[t][t] for t in tasks]
            offdiag = []
            for tgt in tasks:
                vals = [drop[src][tgt] for src in tasks if src != tgt]
                offdiag.append(np.mean(vals))

            x = np.arange(len(tasks))
            w = 0.35
            ax.bar(x - w / 2, diag, w, label="Own circuit", color="#4c78a8", edgecolor="none")
            ax.bar(x + w / 2, offdiag, w, label="Other circuits (mean)", color="#9ecae9", edgecolor="none")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(get_model_display_name(d["model_name"]), fontsize=10)
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            row, col = divmod(i, 3)
            if row == 0:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if col != 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel("Accuracy drop (pp)")

        for j in range(len(chunk), len(axes_flat)):
            axes_flat[j].set_visible(False)

        handles, labels = axes_flat[0].get_legend_handles_labels() if chunk else ([], [])
        if handles:
            fig.legend(handles, labels, fontsize=8, loc="upper right")
        fig.suptitle("Task-specificity: own-circuit vs. cross-circuit ablation damage", fontsize=12, y=1.02)
        fig.tight_layout()

        suffix = "" if len(data) <= chunk_size else f"_part{chunk_idx // chunk_size + 1}"
        path = out_subdir / f"cross_task_diagonal_vs_offdiag{suffix}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
        plt.close(fig)


def plot_normalized_heatmaps(data, out_dir):
    def safe_model_tag(model_name):
        return model_name.replace("/", "_").replace(" ", "_")

    out_subdir = out_dir / "normalized_heatmaps" if out_dir else None
    if out_subdir:
        out_subdir.mkdir(parents=True, exist_ok=True)

    for d in data:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        tasks = d["tasks"]
        drop = d["accuracy_drop_pp"]
        n = len(tasks)
        arr = np.array([[drop[src][tgt] for tgt in tasks] for src in tasks])

        diag = np.diag(arr).copy()
        diag[diag == 0] = np.nan
        norm = arr / diag[:, None]

        labels = [get_task_display_name(t) for t in tasks]
        im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1.0, aspect="equal", interpolation="nearest")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Target task")
        ax.set_ylabel("Source task (circuit ablated)")
        ax.set_title(get_model_display_name(d["model_name"]), fontsize=10)

        for i in range(n):
            for j in range(n):
                v = norm[i, j]
                if np.isnan(v):
                    txt = "--"
                else:
                    txt = f"{v:.2f}"
                weight = "bold" if i == j else "normal"
                color = "white" if (not np.isnan(v) and v > 0.6) else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color, fontweight=weight)

        ax.grid(False)
        ax.grid(which="major", visible=False)
        ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)

        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Drop relative to own-circuit drop")
        fig.tight_layout()

        if out_subdir:
            path = out_subdir / f"cross_task_normalized_heatmap_{safe_model_tag(d['model_name'])}.png"
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Saved {path}")
        plt.close(fig)


def plot_normalized_heatmaps_multiplot(data, out_dir):
    if not out_dir:
        return

    out_subdir = out_dir / "normalized_heatmaps"
    out_subdir.mkdir(parents=True, exist_ok=True)

    chunk_size = 6
    for chunk_idx in range(0, len(data), chunk_size):
        chunk = data[chunk_idx : chunk_idx + chunk_size]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), squeeze=False)
        axes_flat = axes.flatten()
        im = None

        for i, d in enumerate(chunk):
            ax = axes_flat[i]
            tasks = d["tasks"]
            drop = d["accuracy_drop_pp"]
            n = len(tasks)
            arr = np.array([[drop[src][tgt] for tgt in tasks] for src in tasks])

            diag = np.diag(arr).copy()
            diag[diag == 0] = np.nan
            norm = arr / diag[:, None]

            labels = [get_task_display_name(t) for t in tasks]
            im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1.0, aspect="equal", interpolation="nearest")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(get_model_display_name(d["model_name"]), fontsize=10)

            for r in range(n):
                for c in range(n):
                    v = norm[r, c]
                    txt = "--" if np.isnan(v) else f"{v:.2f}"
                    weight = "bold" if r == c else "normal"
                    color = "white" if (not np.isnan(v) and v > 0.6) else "black"
                    ax.text(c, r, txt, ha="center", va="center", fontsize=7, color=color, fontweight=weight)

            ax.grid(False)
            ax.grid(which="major", visible=False)
            ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(n - 0.5, -0.5)

            row, col = divmod(i, 3)
            if row == 0:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if col != 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

        for j in range(len(chunk), len(axes_flat)):
            axes_flat[j].set_visible(False)

        if im is not None:
            fig.colorbar(
                im,
                ax=axes_flat[: len(chunk)],
                shrink=0.7,
                pad=0.02,
                label="Drop relative to own-circuit drop",
            )
        fig.tight_layout(rect=[0, 0, 0.93, 1.0])

        suffix = "" if len(data) <= chunk_size else f"_part{chunk_idx // chunk_size + 1}"
        path = out_subdir / f"cross_task_normalized_heatmaps{suffix}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
        plt.close(fig)


def plot_specificity_summary(data, out_dir):
    def safe_model_tag(model_name):
        return model_name.replace("/", "_").replace(" ", "_")

    out_subdir = out_dir / "specificity_summary" if out_dir else None
    if out_subdir:
        out_subdir.mkdir(parents=True, exist_ok=True)

    for d in data:
        tasks = d["tasks"]
        labels = [get_task_display_name(t) for t in tasks]
        drop = d["accuracy_drop_pp"]
        ratios = []
        for t in tasks:
            diag_val = drop[t][t]
            if diag_val == 0:
                ratios.append(np.nan)
                continue
            off_vals = [drop[src][t] for src in tasks if src != t]
            ratios.append(np.mean(off_vals) / diag_val)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(tasks))
        ax.bar(x, ratios, width=0.6, color="#4c78a8", edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Cross-task / same-task drop ratio")
        ax.set_title(f"Circuit specificity by task — {get_model_display_name(d['model_name'])}")
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        if out_subdir:
            path = out_subdir / f"cross_task_specificity_summary_{safe_model_tag(d['model_name'])}.png"
            fig.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Saved {path}")
        plt.close(fig)


def plot_specificity_summary_multiplot(data, out_dir):
    if not out_dir:
        return

    out_subdir = out_dir / "specificity_summary"
    out_subdir.mkdir(parents=True, exist_ok=True)

    chunk_size = 6
    for chunk_idx in range(0, len(data), chunk_size):
        chunk = data[chunk_idx : chunk_idx + chunk_size]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False, sharey=True)
        axes_flat = axes.flatten()

        for i, d in enumerate(chunk):
            ax = axes_flat[i]
            tasks = d["tasks"]
            labels = [get_task_display_name(t) for t in tasks]
            drop = d["accuracy_drop_pp"]
            ratios = []
            for t in tasks:
                diag_val = drop[t][t]
                if diag_val == 0:
                    ratios.append(np.nan)
                    continue
                off_vals = [drop[src][t] for src in tasks if src != t]
                ratios.append(np.mean(off_vals) / diag_val)

            x = np.arange(len(tasks))
            ax.bar(x, ratios, width=0.6, color="#4c78a8", edgecolor="none")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_title(get_model_display_name(d["model_name"]), fontsize=10)
            ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))

            row, col = divmod(i, 3)
            if row == 0:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            if col != 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel("Cross-task / same-task drop ratio")

        for j in range(len(chunk), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("Circuit specificity by task", fontsize=12, y=1.02)
        fig.tight_layout()

        suffix = "" if len(data) <= chunk_size else f"_part{chunk_idx // chunk_size + 1}"
        path = out_subdir / f"cross_task_specificity_summary{suffix}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Print and plot cross-task ablation matrices.")
    p.add_argument("--results-dir", type=str, default="results2/cross_task_ablation")
    p.add_argument("--output-dir", type=str, default=None, help="Save figures here (optional).")
    p.add_argument(
        "--metric",
        type=str,
        default="accuracy_drop_pp",
        choices=["accuracy_drop_pp", "relative_drop_pct", "ablated_accuracy_pct"],
        help="Which matrix to display in heatmaps.",
    )
    return p.parse_args()


def main():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    args = parse_args()
    results_dir = Path(args.results_dir)
    data = load_cross_task_jsons(results_dir)
    if not data:
        print(f"No cross_task_*.json files found in {results_dir}")
        return

    metric = args.metric
    metric_labels = {
        "accuracy_drop_pp": "Accuracy drop (pp)",
        "relative_drop_pct": "Relative drop (%)",
        "ablated_accuracy_pct": "Ablated accuracy (%)",
    }

    for d in data:
        model = get_model_display_name(d["model_name"])
        tasks = d["tasks"]
        title = f"{model} — {metric_labels[metric]} (K={d['K']}, threshold={d['threshold']})"
        print_matrix(d[metric], tasks, title)
        bl = d["baseline_accuracy"]
        cs = d["circuit_sizes"]
        print("  Baselines:", {get_task_display_name(t): f"{bl[t]*100:.0f}%" for t in tasks})
        print("  Circuit sizes:", {get_task_display_name(t): cs[t] for t in tasks})

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if out_dir:
        plot_heatmaps(data, metric, metric_labels[metric], out_dir)
        plot_diagonal_vs_offdiag(data, out_dir)
        plot_normalized_heatmaps(data, out_dir)
        plot_specificity_summary(data, out_dir)
        plot_heatmaps_multiplot(data, metric, metric_labels[metric], out_dir)
        plot_diagonal_vs_offdiag_multiplot(data, out_dir)
        plot_normalized_heatmaps_multiplot(data, out_dir)
        plot_specificity_summary_multiplot(data, out_dir)


if __name__ == "__main__":
    main()