import argparse, json, re
from pathlib import Path

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from circuit_reuse.dataset import get_task_display_name, get_model_display_name


def _extract_topk_from_filename(filename):
    match = re.search(r"_K(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None


def load_cross_task_jsons(results_dir):
    paths = sorted(results_dir.glob("cross_task_*.json"))
    data = []
    for p in paths:
        with p.open() as f:
            payload = json.load(f)
        payload["_source_filename"] = p.name
        payload["_topk"] = _extract_topk_from_filename(p.name)
        data.append(payload)
    return data


def exclude_tasks(data, excluded_task_names=None):
    if excluded_task_names is None:
        excluded_task_names = {"mmlu"}
    excluded = {t.lower() for t in excluded_task_names}

    matrix_keys = ["accuracy_drop_pp", "relative_drop_pct", "ablated_accuracy_pct"]
    vector_keys = ["baseline_accuracy", "circuit_sizes"]

    filtered = []
    for d in data:
        keep_tasks = [t for t in d["tasks"] if t.lower() not in excluded]
        if not keep_tasks:
            continue

        new_d = dict(d)
        new_d["tasks"] = keep_tasks

        for k in matrix_keys:
            if k in d:
                new_d[k] = {
                    src: {tgt: d[k][src][tgt] for tgt in keep_tasks}
                    for src in keep_tasks
                }

        for k in vector_keys:
            if k in d:
                new_d[k] = {t: d[k][t] for t in keep_tasks}

        filtered.append(new_d)

    return filtered


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


def _format_model_title(d):
    model_label = get_model_display_name(d["model_name"])
    return model_label


def _format_topk_suffix(d, prefix="K="):
    topk = d.get("_topk") or d.get("K")
    if topk is None:
        return ""
    return f"{prefix}{topk}"


def _multiplot_k_filename_suffix(chunk):
    ks = sorted({d.get("_topk") or d.get("K") for d in chunk if (d.get("_topk") or d.get("K")) is not None})
    if not ks:
        return ""
    if len(ks) == 1:
        return f"_K{ks[0]}"
    return "_Kmulti"


def _two_color_diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "teal_grey_coral",
        ["#2A9D8F", "#F2F2F2", "#E76F51"],
        N=256,
    )


def render_heatmap(
    matrix,
    tasks,
    title,
    ax,
    vmin=None,
    vmax=None,
    vcenter=None,
    cmap="RdYlGn_r",
    fmt=".1f",
    source_label="Source task (circuit ablated)",
):
    n = len(tasks)
    arr = np.array([[matrix[src][tgt] for tgt in tasks] for src in tasks])
    labels = [get_task_display_name(t) for t in tasks]

    if vcenter is not None and vmin is not None and vmax is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = None
    imshow_kwargs = {
        "cmap": cmap,
        "aspect": "equal",
        "interpolation": "nearest",
    }
    if norm is not None:
        imshow_kwargs["norm"] = norm
    else:
        imshow_kwargs["vmin"] = vmin
        imshow_kwargs["vmax"] = vmax

    im = ax.imshow(arr, **imshow_kwargs)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    ax.set_xlabel("Target task")
    ax.set_ylabel(source_label)
    ax.set_title("", fontsize=10)

    vmin_eff = float(np.nanmin(arr) if vmin is None else vmin)
    vmax_eff = float(np.nanmax(arr) if vmax is None else vmax)
    denom = (vmax_eff - vmin_eff) if (vmax_eff - vmin_eff) != 0 else 1.0

    for i in range(n):
        for j in range(n):
            norm = (arr[i, j] - vmin_eff) / denom
            color = "white" if norm > 0.6 else "black"
            ax.text(j, i, f"{arr[i, j]:{fmt}}", ha="center", va="center", fontsize=9, color=color)

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
        vcenter = None
    else:
        min_val = min(all_vals)
        max_val = max(all_vals)
        max_abs = max(abs(min_val), abs(max_val))
        vmin = -max_abs
        vmax = max_abs
        vcenter = 0
        cmap = _two_color_diverging_cmap()

    def safe_model_tag(model_name):
        return model_name.replace("/", "_").replace(" ", "_")

    out_subdir = out_dir / "heatmaps" if out_dir else None
    if out_subdir:
        out_subdir.mkdir(parents=True, exist_ok=True)

    for d in data:
        model_label = _format_model_title(d)
        k_suffix = _format_topk_suffix(d)
        source_label = "Source task (circuit ablated)" + (f"  {k_suffix}" if k_suffix else "")
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = render_heatmap(
            d[metric],
            d["tasks"],
            model_label,
            ax,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            cmap=cmap,
            source_label=source_label,
        )
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
        vcenter = None
    else:
        min_val = min(all_vals)
        max_val = max(all_vals)
        max_abs = max(abs(min_val), abs(max_val))
        vmin = -max_abs
        vmax = max_abs
        vcenter = 0
        cmap = _two_color_diverging_cmap()

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
            model_label = _format_model_title(d)
            k_suffix = _format_topk_suffix(d)
            source_label = "Source task (circuit ablated)" + (f"  {k_suffix}" if k_suffix else "")

            im = render_heatmap(
                d[metric],
                d["tasks"],
                model_label,
                ax,
                vmin=vmin,
                vmax=vmax,
                vcenter=vcenter,
                cmap=cmap,
                source_label=source_label,
            )

            ax.set_title(model_label, fontsize=15, pad=4)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", which="major", labelsize=12)

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

        fig.subplots_adjust(left=0.14, right=0.86, bottom=0.20, top=0.96, wspace=0.10, hspace=0.18)
        fig.supxlabel("Target task", fontsize=14, y=0.04)
        any_k = _format_topk_suffix(chunk[0]) if chunk else ""
        fig.supylabel(
            "Source task (circuit ablated)" + (f"  {any_k}" if any_k else ""),
            fontsize=14,
            x=0.0,
        )

        if im is not None:
            cax = fig.add_axes([0.885, 0.22, 0.02, 0.58])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label(metric_label, fontsize=13)
            cb.ax.tick_params(labelsize=12)

        k_suffix = _multiplot_k_filename_suffix(chunk)
        suffix = "" if len(data) <= chunk_size else f"_part{chunk_idx // chunk_size + 1}"
        path = out_subdir / f"cross_task_heatmaps_{metric}{k_suffix}{suffix}.png"
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
        offdiag_ci = []
        for tgt in tasks:
            vals = [drop[src][tgt] for src in tasks if src != tgt]
            offdiag.append(np.mean(vals))
            n = len(vals)
            se = np.std(vals, ddof=1) / np.sqrt(n)
            offdiag_ci.append(scipy.stats.t.ppf(0.975, df=n - 1) * se)

        x = np.arange(len(tasks))
        w = 0.35
        ax.bar(x - w / 2, diag, w, label="Own circuit", color="#6A0572", edgecolor="none")
        ax.bar(x + w / 2, offdiag, w, yerr=offdiag_ci, capsize=3, error_kw={"linewidth": 1.2}, label="Other circuits (mean)", color="#AB83A1", edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
        k_suffix = _format_topk_suffix(d)
        ax.set_ylabel("Accuracy drop (pp)" + (f"  {k_suffix}" if k_suffix else ""))
        ax.set_title("", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
        ax.grid(False)
        ax.grid(True, axis="y", which="major", linewidth=0.7, alpha=0.4)
        ax.grid(False, axis="x", which="both")
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=True, edgecolor="grey", facecolor="white", fancybox=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.suptitle("")
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

        k_val = chunk[0].get("_topk") or chunk[0].get("K") or ""

        for i, d in enumerate(chunk):
            ax = axes_flat[i]
            tasks = d["tasks"]
            drop = d["accuracy_drop_pp"]
            labels = [get_task_display_name(t) for t in tasks]

            diag = [drop[t][t] for t in tasks]
            offdiag = []
            offdiag_ci = []
            for tgt in tasks:
                vals = [drop[src][tgt] for src in tasks if src != tgt]
                offdiag.append(np.mean(vals))
                n = len(vals)
                se = np.std(vals, ddof=1) / np.sqrt(n)
                offdiag_ci.append(scipy.stats.t.ppf(0.975, df=n - 1) * se)

            x = np.arange(len(tasks))
            w = 0.35
            ax.bar(x - w / 2, diag, w, label="Own circuit", color="#6A0572", edgecolor="none")
            ax.bar(x + w / 2, offdiag, w, yerr=offdiag_ci, capsize=3, error_kw={"linewidth": 1.2}, label="Other circuits (mean)", color="#AB83A1", edgecolor="none")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=13)
            ax.set_title(_format_model_title(d), fontsize=18, pad=4)
            ax.axhline(0, color="black", linewidth=0.6, alpha=0.6)
            ax.grid(False)
            ax.grid(True, axis="y", which="major", linewidth=0.6, alpha=0.3)
            ax.grid(False, axis="x", which="both")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=13)

            row, col = divmod(i, 3)
            if row == 0:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

        for j in range(len(chunk), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.supylabel(f"Accuracy Drop (pp)  K={k_val}%", fontsize=19, x=0.06, y=0.55)

        handles, labels = axes_flat[0].get_legend_handles_labels() if chunk else ([], [])
        if handles:
            fig.legend(
                handles,
                labels,
                fontsize=17,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.06),
                ncol=2,
                frameon=True,
                edgecolor="grey",
                facecolor="white",
                fancybox=True,
                borderpad=0.4,
            )
        fig.suptitle("")
        fig.tight_layout(rect=[0.04, 0, 1, 1.0])

        k_suffix = _multiplot_k_filename_suffix(chunk)
        suffix = "" if len(data) <= chunk_size else f"_part{chunk_idx // chunk_size + 1}"
        path = out_subdir / f"cross_task_diagonal_vs_offdiag{k_suffix}{suffix}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
        plt.close(fig)


def _compute_normalized_drop_matrix(d):
    tasks = d["tasks"]
    drop = d["accuracy_drop_pp"]
    arr = np.array([[drop[src][tgt] for tgt in tasks] for src in tasks], dtype=float)

    diag = np.diag(arr).copy()
    diag[diag == 0] = np.nan
    norm = arr / diag[:, None]

    matrix = {
        src: {tgt: float(norm[i, j]) for j, tgt in enumerate(tasks)}
        for i, src in enumerate(tasks)
    }
    return matrix


def _normalized_scale(data):
    vals = []
    for d in data:
        matrix = _compute_normalized_drop_matrix(d)
        tasks = d["tasks"]
        vals.extend(
            matrix[src][tgt]
            for src in tasks
            for tgt in tasks
            if np.isfinite(matrix[src][tgt])
        )

    if not vals:
        return 0.0, 2.0, 1.0

    min_val = min(vals)
    max_val = max(vals)
    max_dev = max(abs(min_val - 1.0), abs(max_val - 1.0))
    return 1.0 - max_dev, 1.0 + max_dev, 1.0


def plot_normalized_heatmaps(data, out_dir):
    def safe_model_tag(model_name):
        return model_name.replace("/", "_").replace(" ", "_")

    out_subdir = out_dir / "normalized_heatmaps" if out_dir else None
    if out_subdir:
        out_subdir.mkdir(parents=True, exist_ok=True)

    vmin, vmax, vcenter = _normalized_scale(data)
    cmap = _two_color_diverging_cmap()

    for d in data:
        matrix = _compute_normalized_drop_matrix(d)
        fig, ax = plt.subplots(figsize=(6, 5.5))
        k_suffix = _format_topk_suffix(d)
        source_label = "Source task (circuit ablated)" + (f"  {k_suffix}" if k_suffix else "")
        im = render_heatmap(
            matrix,
            d["tasks"],
            _format_model_title(d),
            ax,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            cmap=cmap,
            fmt=".2f",
            source_label=source_label,
        )

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

    vmin, vmax, vcenter = _normalized_scale(data)
    cmap = _two_color_diverging_cmap()

    chunk_size = 6
    for chunk_idx in range(0, len(data), chunk_size):
        chunk = data[chunk_idx : chunk_idx + chunk_size]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), squeeze=False)
        axes_flat = axes.flatten()
        im = None

        for i, d in enumerate(chunk):
            ax = axes_flat[i]
            matrix = _compute_normalized_drop_matrix(d)
            k_suffix = _format_topk_suffix(d)
            source_label = "Source task (circuit ablated)" + (f"  {k_suffix}" if k_suffix else "")

            im = render_heatmap(
                matrix,
                d["tasks"],
                _format_model_title(d),
                ax,
                vmin=vmin,
                vmax=vmax,
                vcenter=vcenter,
                cmap=cmap,
                fmt=".2f",
                source_label=source_label,
            )

            ax.set_title(_format_model_title(d), fontsize=15, pad=4)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", which="major", labelsize=12)

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

        if im is not None:
            cax = fig.add_axes([0.885, 0.22, 0.02, 0.58])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("Drop relative to own-circuit drop", fontsize=13)
            cb.ax.tick_params(labelsize=12)

        fig.subplots_adjust(left=0.14, right=0.86, bottom=0.20, top=0.96, wspace=0.10, hspace=0.18)
        fig.supxlabel("Target task", fontsize=14, y=0.04)
        any_k = _format_topk_suffix(chunk[0]) if chunk else ""
        fig.supylabel(
            "Source task (circuit ablated)" + (f"  {any_k}" if any_k else ""),
            fontsize=14,
            x=0.0,
        )

        k_suffix = _multiplot_k_filename_suffix(chunk)
        suffix = "" if len(data) <= chunk_size else f"_part{chunk_idx // chunk_size + 1}"
        path = out_subdir / f"cross_task_normalized_heatmaps{k_suffix}{suffix}.png"
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
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
        k_suffix = _format_topk_suffix(d)
        ax.set_ylabel("Cross-task / same-task drop ratio" + (f"  {k_suffix}" if k_suffix else ""))
        ax.set_title("")
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
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=12)
            ax.set_title(_format_model_title(d), fontsize=15, pad=4)
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
                k_suffix = _format_topk_suffix(d)
                ax.set_ylabel("Cross-task / same-task drop ratio" + (f"  {k_suffix}" if k_suffix else ""))

        for j in range(len(chunk), len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle("")
        fig.tight_layout()

        k_suffix = _multiplot_k_filename_suffix(chunk)
        suffix = "" if len(data) <= chunk_size else f"_part{chunk_idx // chunk_size + 1}"
        path = out_subdir / f"cross_task_specificity_summary{k_suffix}{suffix}.png"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
        plt.close(fig)


def export_diagonal_vs_offdiag_latex(data, out_dir):
    """Export a LaTeX table of own-circuit vs other-circuits accuracy drops."""
    if not out_dir:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by K
    by_k = {}
    for d in data:
        k = d.get("_topk") or d.get("K")
        by_k.setdefault(k, []).append(d)

    for k, entries in sorted(by_k.items()):
        # Collect all tasks from first entry
        tasks = entries[0]["tasks"]
        task_labels = [get_task_display_name(t) for t in tasks]
        n_tasks = len(tasks)

        col_spec = "l" + "r@{\\,/\\,}r" * n_tasks
        lines = [
            f"% Auto-generated: diagonal vs off-diagonal drops at K={k}%",
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
        ]
        # Header row
        header_parts = ["Model"]
        for tl in task_labels:
            header_parts.append(f"\\multicolumn{{2}}{{c}}{{{tl}}}")
        lines.append(" & ".join(header_parts) + " \\\\")
        # Sub-header
        sub = [""]
        for _ in tasks:
            sub.append("Own")
            sub.append("Other")
        lines.append(" & ".join(sub) + " \\\\")
        lines.append("\\midrule")

        for d in entries:
            model = get_model_display_name(d["model_name"])
            drop = d["accuracy_drop_pp"]
            row = [model]
            for t in tasks:
                diag = drop[t][t]
                offdiag_vals = [drop[src][t] for src in tasks if src != t]
                offdiag = np.mean(offdiag_vals)
                row.append(f"{diag:.0f}")
                row.append(f"{offdiag:.0f}")
            lines.append(" & ".join(row) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\caption{{\\textbf{{Own-circuit vs.\\ other-circuits accuracy drop (pp) at $K$={k}\\%.}} "
                     f"For each task, we compare the drop from ablating that task's own circuit against the mean drop "
                     f"from ablating other tasks' circuits. Similar values indicate low task-specificity.}}")
        lines.append(f"\\label{{tab:diagonal_vs_offdiag}}")
        lines.append("\\end{table}")

        path = out_dir / f"diagonal_vs_offdiag_K{k}.tex"
        path.write_text("\n".join(lines) + "\n")
        print(f"[SAVED] {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Print and plot cross-task ablation matrices.")
    p.add_argument("--results-dir", type=str, default="results2/cross_task_ablation")
    p.add_argument("--output-dir", type=str, default=None, help="Save figures here (optional).")
    p.add_argument(
        "--multiplots-dir",
        type=str,
        default=None,
        help="Save multiplots here (optional). If omitted, multiplots are saved to --output-dir.",
    )
    p.add_argument(
        "--metric",
        type=str,
        default="accuracy_drop_pp",
        choices=["accuracy_drop_pp", "relative_drop_pct", "ablated_accuracy_pct"],
        help="Which matrix to display in heatmaps.",
    )
    p.add_argument(
        "--exclude-tasks",
        type=str,
        default="mmlu",
        help="Comma-separated task names to exclude (default: mmlu). Use 'none' to include all.",
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
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    args = parse_args()
    results_dir = Path(args.results_dir)
    data = load_cross_task_jsons(results_dir)
    if not data:
        print(f"No cross_task_*.json files found in {results_dir}")
        return

    if args.exclude_tasks.lower() == "none":
        excluded = set()
    else:
        excluded = {t.strip() for t in args.exclude_tasks.split(",")}
    data = exclude_tasks(data, excluded_task_names=excluded)

    metric = args.metric
    metric_labels = {
        "accuracy_drop_pp": "Accuracy drop (pp)",
        "relative_drop_pct": "Relative drop (%)",
        "ablated_accuracy_pct": "Ablated accuracy (%)",
    }

    for d in data:
        model = get_model_display_name(d["model_name"])
        tasks = d["tasks"]
        title = f"{model}  {metric_labels[metric]} (K={d['K']}, threshold={d['threshold']})"
        print_matrix(d[metric], tasks, title)
        bl = d["baseline_accuracy"]
        cs = d["circuit_sizes"]
        print("  Baselines:", {get_task_display_name(t): f"{bl[t]*100:.0f}%" for t in tasks})
        print("  Circuit sizes:", {get_task_display_name(t): cs[t] for t in tasks})

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    multiplots_dir = Path(args.multiplots_dir) if args.multiplots_dir else out_dir
    if multiplots_dir:
        multiplots_dir.mkdir(parents=True, exist_ok=True)

    if out_dir:
        plot_heatmaps(data, metric, metric_labels[metric], out_dir)
        plot_diagonal_vs_offdiag(data, out_dir)
        plot_normalized_heatmaps(data, out_dir)
        plot_specificity_summary(data, out_dir)

    if multiplots_dir:
        plot_heatmaps_multiplot(data, metric, metric_labels[metric], multiplots_dir)
        plot_diagonal_vs_offdiag_multiplot(data, multiplots_dir)
        plot_normalized_heatmaps_multiplot(data, multiplots_dir)
        plot_specificity_summary_multiplot(data, multiplots_dir)
        export_diagonal_vs_offdiag_latex(data, multiplots_dir / "diagonal_vs_offdiag")


if __name__ == "__main__":
    main()