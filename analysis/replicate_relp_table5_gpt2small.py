#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr
from transformer_lens import HookedTransformer, patching, utils

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from circuit_reuse.lrp_patch import disable_lrp, enable_lrp


PATCH_TYPES = ("resid_pre", "attn_out", "mlp_out")
PATCH_LABELS = {
    "resid_pre": "Residual Stream",
    "attn_out": "Attention Outputs",
    "mlp_out": "MLP Outputs",
}
REFERENCE_LRP_RULES = ["LN-rule", "Identity-rule"]
PAPER_TABLE5_GPT2_SMALL = {
    "resid_pre": {"atp": 0.753, "relp": 0.968},
    "attn_out": {"atp": 0.979, "relp": 0.962},
    "mlp_out": {"atp": 0.195, "relp": 0.992},
}


@dataclass
class PromptPair:
    source_row: int
    clean_prompt: str
    corrupted_prompt: str
    correct_name: str
    incorrect_name: str
    correct_token_id: int
    incorrect_token_id: int


TABLE2_PATTERNS = (
    re.compile(
        r"^Then, ([A-Z][a-z]+) and ([A-Z][a-z]+) went to the ([a-z]+)\. "
        r"([A-Z][a-z]+) gave an? ([a-z]+) to$"
    ),
    re.compile(
        r"^When ([A-Z][a-z]+) and ([A-Z][a-z]+) went to the ([a-z]+), "
        r"([A-Z][a-z]+) gave an? ([a-z]+) to$"
    ),
    re.compile(
        r"^After ([A-Z][a-z]+) and ([A-Z][a-z]+) went to the ([a-z]+), "
        r"([A-Z][a-z]+) gave an? ([a-z]+) to$"
    ),
)
TABLE2_TEMPLATE_BUILDERS = (
    lambda first_name, second_name, giver_name, place, obj, article: (
        f"Then, {first_name} and {second_name} went to the {place}. "
        f"{giver_name} gave {article} {obj} to"
    ),
    lambda first_name, second_name, giver_name, place, obj, article: (
        f"When {first_name} and {second_name} went to the {place}, "
        f"{giver_name} gave {article} {obj} to"
    ),
    lambda first_name, second_name, giver_name, place, obj, article: (
        f"After {first_name} and {second_name} went to the {place}, "
        f"{giver_name} gave {article} {obj} to"
    ),
)
RELP_DEMO_NAMES = ("John", "Mary", "Tom", "James", "Dan", "Sid", "Martin", "Amy")
RELP_DEMO_PLACES = ("shops", "park")
RELP_DEMO_OBJECTS = (
    ("bag", "the"),
    ("ball", "the"),
    ("apple", "an"),
    ("drink", "a"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replicate the RelP paper's GPT-2 Small IOI PCC comparison "
            "(Table 5 / Figure 1 subset) on reference IOI prompt pairs."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt2",
        help="TransformerLens model name. Default: gpt2",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("reference_code/EAP-IG/ioi_llama.csv"),
        help="Reference IOI prompt CSV. Default: reference_code/EAP-IG/ioi_llama.csv",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of valid prompt pairs to use. Default: 100",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. Default: cuda if available else cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bf16", "float16"],
        help="Model dtype hint. Default: float32",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/relp_table5_gpt2small"),
        help="Output directory for JSON, plot, and selected prompts.",
    )
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="synthetic_table2_from_csv",
        choices=[
            "csv_all_valid",
            "csv_table2_subset",
            "synthetic_table2_from_csv",
            "relp_demo_vocab",
        ],
        help=(
            "Prompt construction mode. "
            "'csv_all_valid' reproduces the earlier run over all parseable rows; "
            "'csv_table2_subset' restricts to paper-style IOI templates already present in the CSV; "
            "'synthetic_table2_from_csv' builds fresh prompt pairs from the paper's Table 2 templates "
            "using GPT-2 single-token names/places/objects extracted from the reference CSV; "
            "'relp_demo_vocab' uses the exact name/place/object vocabulary shown in the RelP demo notebooks."
        ),
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="normalized_logit_diff",
        choices=["normalized_logit_diff", "raw_logit_diff"],
        help=(
            "Scoring metric. 'normalized_logit_diff' matches the reference notebooks; "
            "'raw_logit_diff' reproduces the earlier run."
        ),
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="sum_then_correlate",
        choices=["sum_then_correlate", "per_prompt_concat"],
        help=(
            "How to aggregate across prompts before PCC. "
            "'sum_then_correlate' matches the notebook-style batch reduction; "
            "'per_prompt_concat' reproduces the earlier run."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help=(
            "Prompt batch size for 'sum_then_correlate'. "
            "Chunked batches are accumulated exactly by prompt count. Default: 20"
        ),
    )
    return parser.parse_args()


def _dtype_from_arg(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[dtype_name]


def _parse_ioi_targets(clean_prompt: str) -> tuple[str, str]:
    names = re.match(r"^(?:Then, |After |When )([A-Z][a-z]+) and ([A-Z][a-z]+)", clean_prompt)
    if names is None:
        raise ValueError(f"Could not parse first two names from prompt: {clean_prompt!r}")

    giver = re.search(
        r"(?:\. |, and afterwards )([A-Z][a-z]+) (?:(?:decided|wanted) to give|gave|said)\b",
        clean_prompt,
    )
    if giver is None:
        raise ValueError(f"Could not parse giver name from prompt: {clean_prompt!r}")

    first_name, second_name = names.group(1), names.group(2)
    giver_name = giver.group(1)
    if giver_name == first_name:
        return second_name, first_name
    if giver_name == second_name:
        return first_name, second_name
    raise ValueError(
        f"Giver {giver_name!r} is not one of the prompt names {first_name!r}, {second_name!r}"
    )


def _matches_table2_template(clean_prompt: str) -> bool:
    return any(pattern.match(clean_prompt) for pattern in TABLE2_PATTERNS)


def _single_token_id(model: HookedTransformer, token_text: str) -> int | None:
    try:
        return int(model.to_single_token(token_text))
    except Exception:
        return None


def load_prompt_pairs(
    model: HookedTransformer,
    csv_path: Path,
    num_prompts: int,
    prompt_source: str,
) -> list[PromptPair]:
    if prompt_source == "relp_demo_vocab":
        return generate_relp_demo_vocab_pairs(model, num_prompts)
    if prompt_source == "synthetic_table2_from_csv":
        return generate_synthetic_table2_pairs(model, csv_path, num_prompts)

    pairs: list[PromptPair] = []
    skipped_multi_token = 0
    skipped_parse = 0

    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clean_prompt = row["clean"]
            if prompt_source == "csv_table2_subset" and not _matches_table2_template(clean_prompt):
                continue

            try:
                correct_name, incorrect_name = _parse_ioi_targets(clean_prompt)
            except ValueError:
                skipped_parse += 1
                continue

            correct_token_id = _single_token_id(model, f" {correct_name}")
            incorrect_token_id = _single_token_id(model, f" {incorrect_name}")
            if correct_token_id is None or incorrect_token_id is None:
                skipped_multi_token += 1
                continue

            pairs.append(
                PromptPair(
                    source_row=int(row[""]),
                    clean_prompt=clean_prompt,
                    corrupted_prompt=row["corrupted_hard"],
                    correct_name=correct_name,
                    incorrect_name=incorrect_name,
                    correct_token_id=correct_token_id,
                    incorrect_token_id=incorrect_token_id,
                )
            )
            if len(pairs) >= num_prompts:
                break

    if len(pairs) < num_prompts:
        raise RuntimeError(
            f"Requested {num_prompts} prompt pairs but only found {len(pairs)} valid rows "
            f"(skipped_parse={skipped_parse}, skipped_multi_token={skipped_multi_token})."
        )

    print(
        f"[data] Loaded {len(pairs)} prompt pairs from {csv_path} "
        f"using {prompt_source} "
        f"(skipped_parse={skipped_parse}, skipped_multi_token={skipped_multi_token})"
    )
    return pairs


def _article_for(obj: str) -> str:
    return "an" if obj[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def generate_synthetic_table2_pairs(
    model: HookedTransformer,
    csv_path: Path,
    num_prompts: int,
) -> list[PromptPair]:
    valid_names: set[str] = set()
    valid_places: set[str] = set()
    valid_objects: set[str] = set()

    with csv_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            clean_prompt = row["clean"]
            match = next((pattern.match(clean_prompt) for pattern in TABLE2_PATTERNS), None)
            if match is None:
                continue

            first_name, second_name, place, giver_name, obj = match.groups()
            if giver_name not in {first_name, second_name}:
                continue

            for name in (first_name, second_name):
                if _single_token_id(model, f" {name}") is not None:
                    valid_names.add(name)
            valid_places.add(place)
            valid_objects.add(obj)

    names = sorted(valid_names)
    places = sorted(valid_places)
    objects = sorted(valid_objects)
    if len(names) < 2 or not places or not objects:
        raise RuntimeError(
            "Could not build synthetic Table 2 pools from the reference CSV "
            f"(names={len(names)}, places={len(places)}, objects={len(objects)})."
        )

    rng = random.Random(42)
    seen_prompts: set[str] = set()
    pairs: list[PromptPair] = []
    attempts = 0
    max_attempts = max(5000, num_prompts * 50)

    while len(pairs) < num_prompts and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(names, 2)
        place = rng.choice(places)
        obj = rng.choice(objects)
        template_idx = rng.randrange(len(TABLE2_TEMPLATE_BUILDERS))
        article = _article_for(obj)

        clean_prompt = TABLE2_TEMPLATE_BUILDERS[template_idx](b, a, b, place, obj, article)
        corrupted_prompt = TABLE2_TEMPLATE_BUILDERS[template_idx](b, a, a, place, obj, article)
        if clean_prompt in seen_prompts or corrupted_prompt in seen_prompts:
            continue

        correct_token_id = _single_token_id(model, f" {a}")
        incorrect_token_id = _single_token_id(model, f" {b}")
        if correct_token_id is None or incorrect_token_id is None:
            continue

        seen_prompts.add(clean_prompt)
        seen_prompts.add(corrupted_prompt)
        pairs.append(
            PromptPair(
                source_row=-(len(pairs) + 1),
                clean_prompt=clean_prompt,
                corrupted_prompt=corrupted_prompt,
                correct_name=a,
                incorrect_name=b,
                correct_token_id=correct_token_id,
                incorrect_token_id=incorrect_token_id,
            )
        )

    if len(pairs) < num_prompts:
        raise RuntimeError(
            f"Requested {num_prompts} synthetic Table 2 prompt pairs but only built {len(pairs)} "
            f"after {attempts} attempts (names={len(names)}, places={len(places)}, objects={len(objects)})."
        )

    print(
        f"[data] Built {len(pairs)} synthetic Table 2 prompt pairs from {csv_path} "
        f"(names={len(names)}, places={len(places)}, objects={len(objects)}, seed=42)"
    )
    return pairs


def generate_relp_demo_vocab_pairs(
    model: HookedTransformer,
    num_prompts: int,
) -> list[PromptPair]:
    rng = random.Random(42)
    seen_prompts: set[str] = set()
    pairs: list[PromptPair] = []
    attempts = 0
    max_attempts = max(5000, num_prompts * 50)

    while len(pairs) < num_prompts and attempts < max_attempts:
        attempts += 1
        correct_name, incorrect_name = rng.sample(RELP_DEMO_NAMES, 2)
        place = rng.choice(RELP_DEMO_PLACES)
        obj, article = rng.choice(RELP_DEMO_OBJECTS)
        template_idx = rng.randrange(len(TABLE2_TEMPLATE_BUILDERS))

        clean_prompt = TABLE2_TEMPLATE_BUILDERS[template_idx](
            incorrect_name, correct_name, incorrect_name, place, obj, article
        )
        corrupted_prompt = TABLE2_TEMPLATE_BUILDERS[template_idx](
            incorrect_name, correct_name, correct_name, place, obj, article
        )
        if clean_prompt in seen_prompts or corrupted_prompt in seen_prompts:
            continue

        correct_token_id = _single_token_id(model, f" {correct_name}")
        incorrect_token_id = _single_token_id(model, f" {incorrect_name}")
        if correct_token_id is None or incorrect_token_id is None:
            continue

        seen_prompts.add(clean_prompt)
        seen_prompts.add(corrupted_prompt)
        pairs.append(
            PromptPair(
                source_row=-(len(pairs) + 1),
                clean_prompt=clean_prompt,
                corrupted_prompt=corrupted_prompt,
                correct_name=correct_name,
                incorrect_name=incorrect_name,
                correct_token_id=correct_token_id,
                incorrect_token_id=incorrect_token_id,
            )
        )

    if len(pairs) < num_prompts:
        raise RuntimeError(
            f"Requested {num_prompts} RelP-demo-vocab prompt pairs but only built {len(pairs)} "
            f"after {attempts} attempts."
        )

    print(
        "[data] Built "
        f"{len(pairs)} Table 2 prompt pairs from RelP demo vocabulary "
        f"(names={len(RELP_DEMO_NAMES)}, places={len(RELP_DEMO_PLACES)}, "
        f"objects={len(RELP_DEMO_OBJECTS)}, seed=42)"
    )
    return pairs


def _get_final_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        return logits[:, -1, :]
    return logits


def _per_example_logit_diff(
    logits: torch.Tensor,
    answer_token_indices: torch.Tensor,
) -> torch.Tensor:
    logits = _get_final_logits(logits)
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1)).squeeze(1)
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1)).squeeze(1)
    return correct_logits - incorrect_logits


def _logit_diff_metric(answer_token_indices: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    def metric(logits: torch.Tensor) -> torch.Tensor:
        return _per_example_logit_diff(logits, answer_token_indices).mean()

    return metric


def _prompt_pair_tensors(
    model: HookedTransformer,
    pair: PromptPair,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_tokens = model.to_tokens(pair.clean_prompt, prepend_bos=True)
    corrupted_tokens = model.to_tokens(pair.corrupted_prompt, prepend_bos=True)
    answer_token_indices = torch.tensor(
        [[pair.correct_token_id, pair.incorrect_token_id]],
        dtype=torch.long,
        device=model.cfg.device,
    )
    return clean_tokens, corrupted_tokens, answer_token_indices


def _prompt_batch_tensors(
    model: HookedTransformer,
    pairs: list[PromptPair],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_tokens = model.to_tokens([pair.clean_prompt for pair in pairs], prepend_bos=True)
    corrupted_tokens = model.to_tokens([pair.corrupted_prompt for pair in pairs], prepend_bos=True)
    answer_token_indices = torch.tensor(
        [[pair.correct_token_id, pair.incorrect_token_id] for pair in pairs],
        dtype=torch.long,
        device=model.cfg.device,
    )
    return clean_tokens, corrupted_tokens, answer_token_indices


def _chunked(seq: list[PromptPair], chunk_size: int) -> Iterable[list[PromptPair]]:
    for start in range(0, len(seq), chunk_size):
        yield seq[start : start + chunk_size]


def _build_metric(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    answer_token_indices: torch.Tensor,
    metric_name: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    raw_metric = _logit_diff_metric(answer_token_indices)
    if metric_name == "raw_logit_diff":
        return raw_metric
    if metric_name != "normalized_logit_diff":
        raise ValueError(f"Unknown metric: {metric_name}")

    with torch.inference_mode():
        clean_logits = model(clean_tokens)
        corrupted_logits = model(corrupted_tokens)
        clean_baseline = _per_example_logit_diff(clean_logits, answer_token_indices)
        corrupted_baseline = _per_example_logit_diff(corrupted_logits, answer_token_indices)

    denom = clean_baseline - corrupted_baseline
    if torch.any(denom.abs() < 1e-12):
        raise RuntimeError(
            "Degenerate normalized IOI metric: at least one prompt pair has identical clean "
            "and corrupted baselines."
        )

    def metric(logits: torch.Tensor) -> torch.Tensor:
        return ((_per_example_logit_diff(logits, answer_token_indices) - corrupted_baseline) / denom).mean()

    return metric


def _aggregate_attr_maps(model: HookedTransformer, attr_cache: dict[str, torch.Tensor]) -> torch.Tensor:
    per_type: list[torch.Tensor] = []
    for patch_type in PATCH_TYPES:
        per_layer: list[torch.Tensor] = []
        for layer in range(model.cfg.n_layers):
            hook_name = utils.get_act_name(patch_type, layer)
            attr = attr_cache[hook_name]
            per_layer.append(attr.sum(dim=(0, 2)))
        per_type.append(torch.stack(per_layer, dim=0))
    return torch.stack(per_type, dim=0)


def compute_ap_map(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    metric: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    _, clean_cache = model.run_with_cache(clean_tokens)
    ap_map = patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache, metric)
    model.reset_hooks()
    return ap_map.detach().cpu()


def compute_attr_map(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    metric: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    hook_names = [
        utils.get_act_name(patch_type, layer)
        for patch_type in PATCH_TYPES
        for layer in range(model.cfg.n_layers)
    ]

    clean_cache: dict[str, torch.Tensor] = {}
    corrupted_cache: dict[str, torch.Tensor] = {}
    grad_cache: dict[str, torch.Tensor] = {}

    def make_fwd_hook(store: dict[str, torch.Tensor], name: str):
        def hook(act, hook):
            store[name] = act.detach()

        return hook

    def make_bwd_hook(store: dict[str, torch.Tensor], name: str):
        def hook(grad, hook):
            store[name] = grad.detach()

        return hook

    with torch.inference_mode():
        with model.hooks(fwd_hooks=[(name, make_fwd_hook(clean_cache, name)) for name in hook_names]):
            _ = model(clean_tokens)
    model.reset_hooks()

    with model.hooks(
        fwd_hooks=[(name, make_fwd_hook(corrupted_cache, name)) for name in hook_names],
        bwd_hooks=[(name, make_bwd_hook(grad_cache, name)) for name in hook_names],
    ):
        logits = model(corrupted_tokens)
        metric(logits).backward()

    model.zero_grad(set_to_none=True)
    model.reset_hooks()

    attr_cache = {
        name: grad_cache[name] * (clean_cache[name] - corrupted_cache[name])
        for name in hook_names
    }
    return _aggregate_attr_maps(model, attr_cache).detach().cpu()


def flatten_by_patch_type(maps: Iterable[torch.Tensor]) -> dict[str, torch.Tensor]:
    flat = {patch_type: [] for patch_type in PATCH_TYPES}
    for block_map in maps:
        for idx, patch_type in enumerate(PATCH_TYPES):
            flat[patch_type].append(block_map[idx].reshape(-1))
    return {patch_type: torch.cat(chunks) for patch_type, chunks in flat.items()}


def flatten_single_map_by_patch_type(block_map: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        patch_type: block_map[idx].reshape(-1)
        for idx, patch_type in enumerate(PATCH_TYPES)
    }


def compute_pcc(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict[str, float]:
    out: dict[str, float] = {}
    for patch_type in PATCH_TYPES:
        x = reference[patch_type].numpy()
        y = candidate[patch_type].numpy()
        out[patch_type] = float(pearsonr(x, y).statistic)
    return out


def save_prompt_pairs(path: Path, pairs: list[PromptPair]) -> None:
    payload = [asdict(pair) for pair in pairs]
    path.write_text(json.dumps(payload, indent=2))


def save_plot(path: Path, results: dict[str, dict[str, float]]) -> None:
    labels = [PATCH_LABELS[patch_type] for patch_type in PATCH_TYPES]
    atp_paper = [PAPER_TABLE5_GPT2_SMALL[patch_type]["atp"] for patch_type in PATCH_TYPES]
    relp_paper = [PAPER_TABLE5_GPT2_SMALL[patch_type]["relp"] for patch_type in PATCH_TYPES]
    atp_ours = [results["atp"][patch_type] for patch_type in PATCH_TYPES]
    relp_ours = [results["relp"][patch_type] for patch_type in PATCH_TYPES]

    x = torch.arange(len(labels), dtype=torch.float32).numpy()
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, atp_paper, width, label="AtP paper", color="#8da0cb")
    ax.bar(x - 0.5 * width, atp_ours, width, label="AtP ours", color="#4c78a8")
    ax.bar(x + 0.5 * width, relp_paper, width, label="RelP paper", color="#f28e2b")
    ax.bar(x + 1.5 * width, relp_ours, width, label="RelP ours", color="#e15759")

    ax.set_ylabel("Pearson Correlation vs AP")
    ax.set_title("RelP Table 5 Replication on GPT-2 Small IOI")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[model] Loading {args.model_name} on {args.device} ({args.dtype})")
    model = HookedTransformer.from_pretrained(
        args.model_name,
        device=args.device,
        dtype=_dtype_from_arg(args.dtype),
    )
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    model.set_use_attn_in(True)
    model.set_use_split_qkv_input(True)

    pairs = load_prompt_pairs(model, args.csv_path, args.num_prompts, args.prompt_source)
    save_prompt_pairs(args.output_dir / "selected_prompt_pairs.json", pairs)

    if args.aggregation == "sum_then_correlate":
        disable_lrp(model)
        ap_sum: torch.Tensor | None = None
        atp_sum: torch.Tensor | None = None
        for batch_idx, batch_pairs in enumerate(_chunked(pairs, args.batch_size), start=1):
            clean_tokens, corrupted_tokens, answer_token_indices = _prompt_batch_tensors(model, batch_pairs)
            metric = _build_metric(
                model,
                clean_tokens,
                corrupted_tokens,
                answer_token_indices,
                args.metric,
            )
            batch_weight = len(batch_pairs)
            ap_chunk = compute_ap_map(model, clean_tokens, corrupted_tokens, metric) * batch_weight
            atp_chunk = compute_attr_map(model, clean_tokens, corrupted_tokens, metric) * batch_weight
            ap_sum = ap_chunk if ap_sum is None else ap_sum + ap_chunk
            atp_sum = atp_chunk if atp_sum is None else atp_sum + atp_chunk
            print(f"[atp/ap] processed batch {batch_idx} ({batch_weight} prompts)")

        enable_lrp(model, rules=REFERENCE_LRP_RULES)
        relp_sum: torch.Tensor | None = None
        for batch_idx, batch_pairs in enumerate(_chunked(pairs, args.batch_size), start=1):
            clean_tokens, corrupted_tokens, answer_token_indices = _prompt_batch_tensors(model, batch_pairs)
            metric = _build_metric(
                model,
                clean_tokens,
                corrupted_tokens,
                answer_token_indices,
                args.metric,
            )
            batch_weight = len(batch_pairs)
            relp_chunk = compute_attr_map(model, clean_tokens, corrupted_tokens, metric) * batch_weight
            relp_sum = relp_chunk if relp_sum is None else relp_sum + relp_chunk
            print(f"[relp] processed batch {batch_idx} ({batch_weight} prompts)")
        disable_lrp(model)

        assert ap_sum is not None and atp_sum is not None and relp_sum is not None
        flat_ap = flatten_single_map_by_patch_type(ap_sum)
        flat_atp = flatten_single_map_by_patch_type(atp_sum)
        flat_relp = flatten_single_map_by_patch_type(relp_sum)
    else:
        disable_lrp(model)
        ap_maps: list[torch.Tensor] = []
        atp_maps: list[torch.Tensor] = []
        for idx, pair in enumerate(pairs, start=1):
            clean_tokens, corrupted_tokens, answer_token_indices = _prompt_pair_tensors(model, pair)
            metric = _build_metric(
                model,
                clean_tokens,
                corrupted_tokens,
                answer_token_indices,
                args.metric,
            )
            ap_maps.append(compute_ap_map(model, clean_tokens, corrupted_tokens, metric))
            atp_maps.append(compute_attr_map(model, clean_tokens, corrupted_tokens, metric))
            if idx % 10 == 0 or idx == len(pairs):
                print(f"[atp/ap] {idx}/{len(pairs)} prompt pairs processed")

        enable_lrp(model, rules=REFERENCE_LRP_RULES)
        relp_maps: list[torch.Tensor] = []
        for idx, pair in enumerate(pairs, start=1):
            clean_tokens, corrupted_tokens, answer_token_indices = _prompt_pair_tensors(model, pair)
            metric = _build_metric(
                model,
                clean_tokens,
                corrupted_tokens,
                answer_token_indices,
                args.metric,
            )
            relp_maps.append(compute_attr_map(model, clean_tokens, corrupted_tokens, metric))
            if idx % 10 == 0 or idx == len(pairs):
                print(f"[relp] {idx}/{len(pairs)} prompt pairs processed")
        disable_lrp(model)

        flat_ap = flatten_by_patch_type(ap_maps)
        flat_atp = flatten_by_patch_type(atp_maps)
        flat_relp = flatten_by_patch_type(relp_maps)

    results = {
        "atp": compute_pcc(flat_ap, flat_atp),
        "relp": compute_pcc(flat_ap, flat_relp),
    }
    deltas = {
        method: {
            patch_type: results[method][patch_type] - PAPER_TABLE5_GPT2_SMALL[patch_type][method]
            for patch_type in PATCH_TYPES
        }
        for method in ("atp", "relp")
    }

    payload = {
        "experiment": "relp_table5_gpt2small_ioi_pcc",
        "model_name": args.model_name,
        "device": args.device,
        "dtype": args.dtype,
        "num_prompts": len(pairs),
        "csv_path": str(args.csv_path),
        "prompt_source": args.prompt_source,
        "metric_name": args.metric,
        "aggregation": args.aggregation,
        "lrp_rules": REFERENCE_LRP_RULES,
        "paper_values": PAPER_TABLE5_GPT2_SMALL,
        "results": results,
        "delta_vs_paper": deltas,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    save_plot(args.output_dir / "pcc_vs_paper.png", results)

    print("[done] Results written to", metrics_path)
    print(json.dumps(payload["results"], indent=2))


if __name__ == "__main__":
    main()
