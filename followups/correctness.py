"""Per-example baseline correctness and gold log-probability on the training split,
aligned with the rows of the attribution caches, so reuse can be recomputed on the
examples each model answers correctly.

    python -m followups.correctness --models google/gemma-2-2b,qwen3-4b
"""
import argparse

from circuit_reuse.evaluate import evaluate_graded
from followups.common import MODELS, OUT, TASKS, load_model, train_examples, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    args = parser.parse_args()
    for name in args.models.split(","):
        model = load_model(name)
        for task in args.tasks.split(","):
            path = OUT / "correctness" / f"{name.replace('/', '_')}__{task}.json"
            if path.exists():
                continue
            rows = evaluate_graded(model, train_examples(task, name), task)
            write_json(path, {"model": name, "task": task, "rows": rows})
            print(f"[{name}/{task}] accuracy {sum(r['correct'] for r in rows) / len(rows):.3f} over {len(rows)}")
        del model


if __name__ == "__main__":
    main()
