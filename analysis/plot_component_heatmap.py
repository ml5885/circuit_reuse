"""Component membership heatmap across top-K thresholds.

For each (model, task) pair, produces a heatmap where:
  - X-axis: top-K% values
  - Y-axis: every component in the model (ordered by layer, heads then MLP)
  - Color: fraction of examples where that component is in the top-K% circuit
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from circuit_reuse.dataset import get_model_display_name, get_task_display_name

matplotlib.use("Agg")


def parse_component_label(c: dict) -> str:
    if c["kind"] == "mlp":
        return f"L{c['layer']} MLP"
    return f"L{c['layer']} H{c['index']}"


def component_sort_key(c: dict) -> tuple:
    """Sort by layer, then MLPs after heads within the same layer."""
    return (c["layer"], 0 if c["kind"] == "head" else 1, c["index"])


def load_attributions(path: Path) -> list[list[dict]]:
    """Load per-example attribution scores from JSONL."""
    examples = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            examples.append(row["components"])
    return examples


def compute_frequency_matrix(
    examples: list[list[dict]], k_percents: list[float]
) -> tuple[np.ndarray, list[str]]:
    """Compute per-component membership frequency across examples at each K%.

    Returns (matrix of shape [n_components, len(k_percents)], component_labels).
    """
    n_total = len(examples[0])
    # Build canonical component ordering from the first example's full list
    all_components = sorted(examples[0], key=component_sort_key)
    comp_to_idx = {}
    labels = []
    for i, c in enumerate(all_components):
        key = (c["layer"], c["kind"], c["index"])
        comp_to_idx[key] = i
        labels.append(parse_component_label(c))

    n_components = len(all_components)
    n_examples = len(examples)
    freq = np.zeros((n_components, len(k_percents)))

    for ex_components in examples:
        for ki, k_pct in enumerate(k_percents):
            top_n = max(1, math.ceil(n_total * k_pct / 100))
            # Components are already sorted by score descending
            for c in ex_components[:top_n]:
                key = (c["layer"], c["kind"], c["index"])
                idx = comp_to_idx[key]
                freq[idx, ki] += 1

    freq /= n_examples
    return freq, labels


def plot_heatmap(
    freq: np.ndarray,
    labels: list[str],
    k_percents: list[float],
    model_name: str,
    task: str,
    output_path: Path,
):
    # Split into MLP and head indices
    mlp_idx = [i for i, l in enumerate(labels) if "MLP" in l]
    head_idx = [i for i, l in enumerate(labels) if "MLP" not in l]

    # Flip so layer 0 is at bottom, last layer at top
    freq_mlp = freq[mlp_idx][::-1]
    freq_head = freq[head_idx][::-1]

    n_mlp = len(mlp_idx)
    n_head = len(head_idx)
    n_k = len(k_percents)

    model_disp = get_model_display_name(model_name)
    task_disp = get_task_display_name(task)

    # Side-by-side: MLP on left, heads on right, colorbar on far right
    fig = plt.figure(figsize=(18, max(5, n_mlp * 0.22)))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 3, 0.08], wspace=0.3)
    ax_mlp = fig.add_subplot(gs[0, 0])
    ax_head = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

    cmap = LinearSegmentedColormap.from_list(
        "circuit", ["#f7f7f7", "#2166ac"], N=256
    )

    # --- MLP panel ---
    ax_mlp.imshow(freq_mlp, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax_mlp.set_xticks(range(n_k))
    ax_mlp.set_xticklabels([f"{k:g}%" for k in k_percents], fontsize=8, rotation=45, ha="right")
    ax_mlp.set_yticks(range(n_mlp))
    ax_mlp.set_yticklabels(list(range(n_mlp - 1, -1, -1)), fontsize=7)
    ax_mlp.set_xlabel("Top-K%", fontsize=9)
    ax_mlp.set_ylabel("Layer", fontsize=10)
    ax_mlp.set_title("MLP", fontsize=11)

    # --- Head panel ---
    n_heads_per_layer = sum(1 for l in labels if l.startswith("L0 H"))
    im = ax_head.imshow(freq_head, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax_head.set_xticks(range(n_k))
    ax_head.set_xticklabels([f"{k:g}%" for k in k_percents], fontsize=8, rotation=45, ha="right")
    ax_head.set_xlabel("Top-K%", fontsize=9)
    ax_head.set_title("Attention Heads", fontsize=11)

    # Y-axis: tick at center of each layer block (reversed: last layer at top)
    n_layers = n_mlp
    tick_positions = []
    tick_labels_y = []
    for layer in range(n_layers):
        # In reversed order, layer (n_layers-1-layer) maps to row position layer
        row_start = layer * n_heads_per_layer
        mid = row_start + (n_heads_per_layer - 1) / 2
        tick_positions.append(mid)
        tick_labels_y.append(str(n_layers - 1 - layer))
        if layer > 0:
            ax_head.axhline(y=row_start - 0.5, color="gray", linewidth=0.3, alpha=0.5)

    ax_head.set_yticks(tick_positions)
    ax_head.set_yticklabels(tick_labels_y, fontsize=7)
    ax_head.set_ylabel("Layer", fontsize=10)

    # Head index labels on the right y-axis
    ax_head_r = ax_head.twinx()
    ax_head_r.set_ylim(ax_head.get_ylim())
    ax_head_r.set_yticks(range(n_head))
    ax_head_r.set_yticklabels(
        [f"H{(n_heads_per_layer - 1) - (i % n_heads_per_layer)}" for i in range(n_head)],
        fontsize=max(2, min(5, 200 / n_head)),
        family="monospace",
    )
    ax_head_r.tick_params(length=0)

    fig.colorbar(im, cax=ax_cbar, label="Fraction of examples")

    fig.suptitle(f"{model_disp} — {task_disp}", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


def _split_mlp_head(freq: np.ndarray, labels: list[str]):
    """Split frequency matrix into MLP and head parts, flipped so layer 0 is at bottom."""
    mlp_idx = [i for i, l in enumerate(labels) if "MLP" in l]
    head_idx = [i for i, l in enumerate(labels) if "MLP" not in l]
    n_heads_per_layer = sum(1 for l in labels if l.startswith("L0 H"))
    return (
        freq[mlp_idx][::-1],
        freq[head_idx][::-1],
        len(mlp_idx),
        len(head_idx),
        n_heads_per_layer,
    )


def _setup_mlp_ax(ax, freq_mlp, n_mlp, k_percents, cmap, model_disp):
    """Configure a single MLP heatmap axis."""
    n_k = len(k_percents)
    ax.imshow(freq_mlp, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f"{k:g}%" for k in k_percents], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n_mlp))
    ax.set_yticklabels(list(range(n_mlp - 1, -1, -1)), fontsize=6)
    ax.set_title(model_disp, fontsize=9)


def _setup_head_ax(ax, freq_head, n_head, n_layers, n_heads_per_layer, k_percents, cmap, model_disp):
    """Configure a single attention head heatmap axis."""
    n_k = len(k_percents)
    ax.imshow(freq_head, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n_k))
    ax.set_xticklabels([f"{k:g}%" for k in k_percents], fontsize=7, rotation=45, ha="right")

    tick_positions = []
    tick_labels_y = []
    for layer in range(n_layers):
        row_start = layer * n_heads_per_layer
        mid = row_start + (n_heads_per_layer - 1) / 2
        tick_positions.append(mid)
        tick_labels_y.append(str(n_layers - 1 - layer))
        if layer > 0:
            ax.axhline(y=row_start - 0.5, color="gray", linewidth=0.3, alpha=0.5)

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels_y, fontsize=6)
    ax.set_title(model_disp, fontsize=9)


def plot_multimodel(
    all_data: dict[str, tuple[np.ndarray, list[str]]],
    k_percents: list[float],
    task: str,
    output_dir: Path,
):
    """Plot 2x3 grids of MLP and head heatmaps for all models on one task."""
    models = sorted(all_data.keys())
    if len(models) > 6:
        models = models[:6]

    nrows, ncols = 2, 3
    task_disp = get_task_display_name(task)
    cmap = LinearSegmentedColormap.from_list("circuit", ["#f7f7f7", "#2166ac"], N=256)

    # --- MLP multiplot ---
    fig_mlp, axes_mlp = plt.subplots(nrows, ncols, figsize=(16, 10))
    for idx, model_name in enumerate(models):
        r, c = divmod(idx, ncols)
        ax = axes_mlp[r, c]
        freq, labels = all_data[model_name]
        freq_m, _, n_mlp, _, _ = _split_mlp_head(freq, labels)
        model_disp = get_model_display_name(model_name)
        _setup_mlp_ax(ax, freq_m, n_mlp, k_percents, cmap, model_disp)
        if c == 0:
            ax.set_ylabel("Layer", fontsize=9)

    # Hide unused axes
    for idx in range(len(models), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes_mlp[r, c].set_visible(False)

    fig_mlp.suptitle(f"{task_disp} — MLP Component Membership", fontsize=14)
    fig_mlp.tight_layout(rect=[0, 0, 0.93, 0.95])
    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig_mlp.colorbar(sm, ax=axes_mlp.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Fraction of examples", fontsize=9)

    out = output_dir / "multiplots" / f"{task}_mlp.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_mlp.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig_mlp)
    print(f"[SAVED] {out}")

    # --- Head multiplot ---
    fig_head, axes_head = plt.subplots(nrows, ncols, figsize=(18, 10))
    for idx, model_name in enumerate(models):
        r, c = divmod(idx, ncols)
        ax = axes_head[r, c]
        freq, labels = all_data[model_name]
        _, freq_h, n_mlp, n_head, n_hpl = _split_mlp_head(freq, labels)
        model_disp = get_model_display_name(model_name)
        _setup_head_ax(ax, freq_h, n_head, n_mlp, n_hpl, k_percents, cmap, model_disp)
        if c == 0:
            ax.set_ylabel("Layer", fontsize=9)

    for idx in range(len(models), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes_head[r, c].set_visible(False)

    fig_head.suptitle(f"{task_disp} — Attention Head Component Membership", fontsize=14)
    fig_head.tight_layout(rect=[0, 0, 0.93, 0.95])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cbar = fig_head.colorbar(sm, ax=axes_head.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Fraction of examples", fontsize=9)

    out = output_dir / "multiplots" / f"{task}_heads.png"
    fig_head.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig_head)
    print(f"[SAVED] {out}")


def find_attribution_files(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("*.jsonl"))


def parse_attrib_filename(path: Path) -> dict:
    """Parse model/task/method from attribution filename.

    Format: {model}__{revision}__{task}__{method}__n{N}__d{digits}__s{seed}.jsonl
    """
    stem = path.stem
    parts = stem.split("__")
    return {
        "model_name": parts[0].replace("_", "/", 1) if "_" in parts[0] else parts[0],
        "revision": parts[1],
        "task": parts[2],
        "method": parts[3],
        "num_examples": parts[4],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-dir", type=str, default="cache",
        help="Directory containing per-example attribution JSONL files",
    )
    p.add_argument("--output-dir", type=str, default="results2/component_heatmaps")
    p.add_argument(
        "--k-percents", type=str, default="1,2,3,5,10,15,20,25,30",
        help="Comma-separated top-K%% values to plot",
    )
    p.add_argument("--models", type=str, default=None, help="Filter to these models (comma-sep)")
    p.add_argument("--tasks", type=str, default=None, help="Filter to these tasks (comma-sep)")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    k_percents = [float(x) for x in args.k_percents.split(",")]
    model_filter = set(args.models.split(",")) if args.models else None
    task_filter = set(args.tasks.split(",")) if args.tasks else None

    attrib_files = find_attribution_files(cache_dir)
    if not attrib_files:
        print(f"No attribution JSONL files found in {cache_dir}")
        return

    # First pass: load all data, grouped by task
    by_task: dict[str, dict[str, tuple[np.ndarray, list[str]]]] = {}

    for path in attrib_files:
        info = parse_attrib_filename(path)
        model_name = info["model_name"]
        task = info["task"]

        if model_filter and model_name not in model_filter:
            continue
        if task_filter and task not in task_filter:
            continue

        print(f"[LOAD] {model_name} / {task} from {path.name}")
        examples = load_attributions(path)
        if not examples:
            print(f"[SKIP] {path.name} — empty file")
            continue
        freq, labels = compute_frequency_matrix(examples, k_percents)

        # Individual plot
        safe_model = model_name.replace("/", "_")
        out_path = output_dir / task / f"{safe_model}.png"
        plot_heatmap(freq, labels, k_percents, model_name, task, out_path)

        by_task.setdefault(task, {})[model_name] = (freq, labels)

    # Second pass: multimodel plots per task
    for task, all_data in by_task.items():
        if len(all_data) >= 2:
            plot_multimodel(all_data, k_percents, task, output_dir)


if __name__ == "__main__":
    main()
