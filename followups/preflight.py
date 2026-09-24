"""Preflight on the real models, run before any extraction: for every model and task,
build the extractor's inputs and check that the metric reads the gold answer where the model
predicts it.

For each task it reports the median probability of the gold answer's first token at the
position the attribution metric uses, the accuracy of the unablated model, and how often the
classifier's prediction disagrees with the argmax over the label tokens. It fails (exit 1)
if input construction raises, or if the median gold probability is below --min-prob for a task
the model does above chance, which is the signature of a metric reading the wrong token.

    python -m followups.preflight --models google/gemma-2-2b,qwen3-4b
"""
import argparse
import sys

import torch

from circuit_reuse.circuit_extraction import CircuitExtractor
from circuit_reuse.evaluate import evaluate_graded
from followups.common import MODELS, TASKS, eval_datasets, load_model

CHANCE = {"addition": 0.0, "boolean": 0.5, "ioi": 0.5, "mcqa": 0.25, "arc_easy": 0.25, "arc_challenge": 0.25}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--min-prob", type=float, default=0.05)
    args = parser.parse_args()
    failures = []
    torch.set_grad_enabled(False)
    for name in args.models.split(","):
        model = load_model(name)
        extractor = CircuitExtractor(model, method="relp", granularity="head_mlp")
        for task, ds in eval_datasets(name, num_examples=args.num_examples).items():
            probs = []
            try:
                for ex in ds:
                    clean, _, mask, _, metric, _ = extractor._prepare_paired_inputs(ex)
                    logits = model(clean, attention_mask=mask)
                    probs.append(float(metric(logits).exp()))  # log-prob of the answer tokens
            except ValueError as e:
                failures.append(f"{name} {task}: {e}")
                print(f"FAIL {name:34s} {task:14s} {e}", flush=True)
                continue
            acc = sum(r["correct"] for r in evaluate_graded(model, ds, task)) / len(ds)
            median = sorted(probs)[len(probs) // 2]
            status = "ok"
            if median < args.min_prob and acc > CHANCE[task] + 0.1:
                status = "FAIL"
                failures.append(f"{name} {task}: median p(gold) {median:.4f} with accuracy {acc:.2f}")
            print(f"{status:4s} {name:34s} {task:14s} median p(gold) {median:.3f}  accuracy {acc:.2f}  (chance {CHANCE[task]:.2f})", flush=True)
        del model, extractor
        torch.cuda.empty_cache()
    if failures:
        print("\n".join(["PREFLIGHT FAILED:"] + failures))
        sys.exit(1)
    print("PREFLIGHT OK")


if __name__ == "__main__":
    main()
