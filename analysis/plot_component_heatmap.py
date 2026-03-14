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
    n_components = freq.shape[0]
    fig_height = max(6, n_components * 0.055)
    fig, ax = plt.subplots(figsize=(3 + 0.8 * len(k_percents), fig_height))

    cmap = LinearSegmentedColormap.from_list(
        "circuit", ["#f7f7f7", "#2166ac"], N=256
    )
    im = ax.imshow(freq, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(k_percents)))
    ax.set_xticklabels([f"{k}%" for k in k_percents], fontsize=9)
    ax.set_xlabel("Top-K%")

    # Y-axis: show only layer-boundary labels to reduce clutter
    # Find indices where a new layer starts, and MLP positions
    layer_starts = []
    mlp_positions = []
    for i, lbl in enumerate(labels):
        if "H0" in lbl:
            layer_starts.append(i)
        if "MLP" in lbl:
            mlp_positions.append(i)

    # Draw horizontal lines at layer boundaries
    for pos in layer_starts[1:]:
        ax.axhline(y=pos - 0.5, color="gray", linewidth=0.3, alpha=0.5)

    # Label only layer boundaries (center of each layer's block)
    tick_positions = []
    tick_labels = []
    for li, start in enumerate(layer_starts):
        end = layer_starts[li + 1] if li + 1 < len(layer_starts) else n_components
        mid = (start + end - 1) / 2
        layer_num = labels[start].split(" ")[0]  # e.g. "L0"
        tick_positions.append(mid)
        tick_labels.append(layer_num)

    ax.set_yticks(tick_positions)
    fontsize = max(4, min(7, 500 / len(layer_starts)))
    ax.set_yticklabels(tick_labels, fontsize=fontsize)
    ax.set_ylabel("Layer")

    # Mark MLP rows with a subtle left-side indicator
    for pos in mlp_positions:
        ax.plot(-0.7, pos, marker="s", markersize=2, color="#d62728",
                clip_on=False, zorder=5)

    model_disp = get_model_display_name(model_name)
    task_disp = get_task_display_name(task)
    ax.set_title(f"{model_disp} — {task_disp}\nComponent membership frequency", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Fraction of examples", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


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
        freq, labels = compute_frequency_matrix(examples, k_percents)

        safe_model = model_name.replace("/", "_")
        out_path = output_dir / safe_model / f"{task}.png"
        plot_heatmap(freq, labels, k_percents, model_name, task, out_path)


if __name__ == "__main__":
    main()
