"""
Ad-hoc analysis of shared attention heads (L7H1, L8H17) in Llama-3.2-3B
across all six tasks. For each task we sample a few prompts, capture
attention weights at the final query position, and aggregate by template
role to see whether the head does the same thing across tasks.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.granularity_parity import _ramp, _task_label
from circuit_reuse.dataset import (
    AdditionDataset,
    ARCDataset,
    BooleanDataset,
    IOIDataset,
    MCQADataset,
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

MODEL_NAME = "meta-llama/Llama-3.2-3B"
N_PER_TASK = 16


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", default="L7H1,L8H17",
                   help="comma-separated heads, e.g. L8H17,L9H21,L13H12")
    p.add_argument("--out", type=Path, default=Path("new_plots/shared_head_attention"))
    p.add_argument("--replot", action="store_true",
                   help="redraw the summary panels from the cached role aggregates")
    return p.parse_args()


def parse_heads(spec: str) -> list[tuple[int, int]]:
    return [tuple(int(n) for n in h.strip().lstrip("Ll").split("H")) for h in spec.split(",")]


ARGS = parse_args()
HEADS = parse_heads(ARGS.heads)
OUT_DIR = ARGS.out
OUT_DIR.mkdir(parents=True, exist_ok=True)


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_examples(task: str, n: int):
    random.seed(0)
    if task == "addition":
        return list(AdditionDataset(num_examples=n))
    if task == "boolean":
        return list(BooleanDataset(num_examples=n, min_ops=2, max_ops=4))
    if task == "ioi":
        return list(IOIDataset(num_examples=n))
    if task == "mcqa":
        return list(MCQADataset(num_examples=n, n=4))
    if task == "arc_easy":
        return list(ARCDataset("arc_easy", num_examples=n))
    if task == "arc_challenge":
        return list(ARCDataset("arc_challenge", num_examples=n))
    raise ValueError(task)


def classify_token(task: str, tok: str, pos: int, n_tok: int, prompt_lower: str, tokens: list[str]) -> str:
    """Return a coarse template role for token `tok` at position `pos` of a prompt."""
    raw = tok.replace("Ġ", " ").replace("▁", " ")
    stripped = raw.strip().lower()

    if pos == 0:
        return "BOS"
    if pos == n_tok - 1:
        return "FINAL"
    if pos >= n_tok - 3:
        return "RECENT"

    if task == "addition":
        if stripped in {"compute", ":"}:
            return "INSTR"
        if stripped in {"+", "="}:
            return "OP"
        if any(c.isdigit() for c in stripped):
            return "DIGIT"
        return "OTHER"

    if task == "boolean":
        if stripped in {"evaluate", ":"}:
            return "INSTR"
        if stripped in {"true", "false"}:
            return "LITERAL"
        if stripped in {"and", "or", "not", "(", ")", "="}:
            return "OP"
        return "OTHER"

    if task == "ioi":
        # heuristic name detection: capitalized token in the original prompt
        if raw.strip() and raw.strip()[0].isupper() and raw.strip().isalpha():
            return "NAME"
        if stripped in {"and", "to", "the", "a"}:
            return "FUNCTION"
        return "CONTEXT"

    if task in {"mcqa", "arc_easy", "arc_challenge"}:
        if stripped in {"a", "b", "c", "d"} and pos > 0:
            # only label tokens immediately followed by "."
            if pos + 1 < n_tok and tokens[pos + 1].replace("Ġ", " ").replace("▁", " ").strip() == ".":
                return "LABEL"
        if stripped == "answer":
            return "ANSWER_KW"
        if stripped in {":", "."}:
            return "PUNCT"
        if "\n" in raw:
            return "NEWLINE"
        return "QUESTION"

    return "OTHER"


def capture_attention(model, tokenizer, dev, prompt: str, layer: int, head: int) -> tuple[np.ndarray, list[str]]:
    enc = tokenizer(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**enc, output_attentions=True, use_cache=False)
    attn = out.attentions[layer][0, head].float().cpu().numpy()  # (n_tok, n_tok)
    tokens = tokenizer.convert_ids_to_tokens(enc.input_ids[0])
    return attn, tokens


def main():
    dev = device()
    print(f"device: {dev}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32 if dev == "cpu" else torch.float16,
        attn_implementation="eager",
    ).to(dev)
    model.eval()

    tasks = ["addition", "boolean", "ioi", "mcqa", "arc_easy", "arc_challenge"]
    results = {}  # task -> head_key -> {role: [weights]}

    for task in tasks:
        print(f"\n[{task}]")
        examples = load_examples(task, N_PER_TASK)
        results[task] = {f"L{L}H{H}": {} for L, H in HEADS}
        first_examples = {f"L{L}H{H}": [] for L, H in HEADS}

        for ex_idx, ex in enumerate(examples):
            prompt = ex.prompt
            for L, H in HEADS:
                key = f"L{L}H{H}"
                attn, tokens = capture_attention(model, tokenizer, dev, prompt, L, H)
                n_tok = len(tokens)
                # attention FROM final query position TO each key position
                attn_from_final = attn[-1]  # (n_tok,)
                prompt_lower = prompt.lower()
                for pos, tok in enumerate(tokens):
                    role = classify_token(task, tok, pos, n_tok, prompt_lower, tokens)
                    results[task][key].setdefault(role, []).append(float(attn_from_final[pos]))

                if ex_idx < 2:
                    first_examples[key].append((attn_from_final.tolist(), tokens))

            if ex_idx == 0:
                # save full heatmap for first example
                for L, H in HEADS:
                    key = f"L{L}H{H}"
                    attn, tokens = capture_attention(model, tokenizer, dev, prompt, L, H)
                    save_heatmap(attn, tokens, task, key)

        # save per-example final-query attention for first 2 examples
        for key, exs in first_examples.items():
            save_final_query_bars(exs, task, key)

    # aggregate per role
    summary = {}
    for task, by_head in results.items():
        summary[task] = {}
        for key, by_role in by_head.items():
            summary[task][key] = {role: float(np.mean(w)) for role, w in by_role.items()}

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_summary(summary)
    print(f"\nsaved to {OUT_DIR}")


def save_heatmap(attn, tokens, task, key):
    fig, ax = plt.subplots(figsize=(0.18 * len(tokens) + 2, 0.18 * len(tokens) + 2))
    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    pretty = [t.replace("Ġ", "·").replace("▁", "·") for t in tokens]
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(pretty, rotation=90, fontsize=6)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(pretty, fontsize=6)
    ax.set_title(f"{task} | {key} | first example", fontsize=10)
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    plt.colorbar(im, ax=ax, shrink=0.5)
    plt.tight_layout()
    p = OUT_DIR / f"heatmap_{task}_{key}.png"
    plt.savefig(p, dpi=150)
    plt.close()


def save_final_query_bars(exs, task, key):
    fig, axes = plt.subplots(len(exs), 1, figsize=(max(6, 0.16 * max(len(t) for _, t in exs)), 2.4 * len(exs)))
    if len(exs) == 1:
        axes = [axes]
    for ax, (weights, tokens) in zip(axes, exs):
        pretty = [t.replace("Ġ", "·").replace("▁", "·") for t in tokens]
        ax.bar(range(len(tokens)), weights, color="#3a6ea5")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(pretty, rotation=90, fontsize=6)
        ax.set_ylabel("attn from final")
        ax.set_title(f"{task} | {key}", fontsize=9)
    plt.tight_layout()
    p = OUT_DIR / f"final_q_{task}_{key}.png"
    plt.savefig(p, dpi=150)
    plt.close()


def _role_matrix(summary: dict, key: str, tasks: list[str], roles: list[str]) -> np.ndarray:
    return np.array([[summary[t][key].get(r, np.nan) for r in roles] for t in tasks])


def _draw_roles(ax, mat, tasks, roles, title=None):
    # Blank cells render white and a genuine zero renders dark, so an absent
    # role can no longer be confused with a role that gets no attention.
    im = ax.imshow(mat, aspect="auto", cmap=_ramp(), vmin=0, vmax=1)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels([r.capitalize() for r in roles], rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([_task_label(t) for t in tasks], fontsize=9)
    if title:
        ax.set_title(title, fontsize=11)
    for (i, j), v in np.ndenumerate(mat):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                    fontsize=7, color="white" if v > 0.55 else "0.15")
    return im


def plot_summary(summary: dict):
    tasks = list(summary.keys())
    head_keys = [f"L{L}H{H}" for L, H in HEADS]
    roles = sorted({r for t in tasks for k in head_keys for r in summary[t][k]})

    for key in head_keys:
        fig, ax = plt.subplots(figsize=(0.55 * len(roles) + 2, 0.5 * len(tasks) + 2))
        im = _draw_roles(ax, _role_matrix(summary, key, tasks, roles), tasks, roles)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Mean attention weight")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"summary_{key}.png", dpi=150)
        plt.close()

    # One panel per head under a shared colour scale: this is the paper figure.
    fig, axes = plt.subplots(1, len(head_keys), figsize=(
        (0.5 * len(roles) + 1.5) * len(head_keys) + 1.5, 0.5 * len(tasks) + 2.5))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, head_keys):
        im = _draw_roles(ax, _role_matrix(summary, key, tasks, roles), tasks, roles, title=key)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.colorbar(im, ax=axes, shrink=0.7, label="Mean attention weight")
    plt.savefig(OUT_DIR / "summary_combined.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Redrawing the summary panels does not need the model, only the cached
    # role aggregates, so restyling them stays a second-long job.
    if ARGS.replot:
        plot_summary(json.loads((OUT_DIR / "summary.json").read_text()))
    else:
        main()
