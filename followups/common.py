"""Shared helpers for the reviewer follow-up runs."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from circuit_reuse.dataset import apply_few_shot_prefix, get_dataset
from cross_task_experiment import find_metrics_file, load_shared_components
from models.olmo_adapter import load_model_any

TASKS = ["addition", "boolean", "ioi", "mcqa", "arc_easy", "arc_challenge"]
MODELS = ["google/gemma-2-2b", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B",
          "meta-llama/Llama-3.2-3B-Instruct", "qwen3-4b"]
EXTRACT = Path("results/granularity_parity/granularity_parity_extraction")
OUT = Path("results/followup")


def load_model(name: str, device: str = "cuda"):
    model = load_model_any(name, device=device, torch_dtype=torch.bfloat16)
    model.eval()
    return model


def train_examples(task: str, model_name: str, seed: int = 42, num_examples: int = 1000):
    """The training split main_experiment.py extracts circuits from, in the same order
    as the rows of its attribution cache."""
    random.seed(seed)
    ds = get_dataset(task, num_examples=num_examples, digits=3 if task == "addition" else 0)
    examples = apply_few_shot_prefix(ds, task, model_name)
    random.shuffle(examples)
    return examples[: len(examples) - int(round(0.2 * len(examples)))]


def eval_datasets(model_name: str, seed: int = 42, num_examples: int = 100,
                  order: str = "addition,arc_challenge,arc_easy,boolean,ioi,mcqa"):
    """The evaluation examples of the cross-task runs: one seed, then each task drawn
    in the order the pod scripts pass them, so the generated tasks match exactly."""
    random.seed(seed)
    torch.manual_seed(seed)
    out = {}
    for task in order.split(","):
        ds = get_dataset(task, num_examples=num_examples, digits=3 if task == "addition" else 2)
        out[task] = apply_few_shot_prefix(list(ds), task, model_name)
    return out


def circuit(model_name: str, task: str, method: str, granularity: str, K: int = 10, P: int = 50,
            results_dir: Path = EXTRACT):
    path = find_metrics_file(results_dir, model_name, None, task, P, method=method, granularity=granularity)
    return load_shared_components(path, K, P)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)
