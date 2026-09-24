"""Checks on task inputs and scoring that need no model weights.

Tokenizer checks use each model's real tokenizer (downloaded from the Hub, a few MB) and
reproduce the extractor's input construction: the prompt's token ids followed by the
target's. They are skipped when a tokenizer cannot be loaded.

    pytest tests/test_task_inputs.py
"""
import random

import pytest
import torch

from circuit_reuse.dataset import apply_few_shot_prefix, get_dataset
from circuit_reuse.evaluate import _classify_from_labels, _classify_ioi, _score_first_token

TASKS = ["addition", "boolean", "ioi", "mcqa", "arc_easy", "arc_challenge"]
LABEL_TASKS = ["ioi", "mcqa", "arc_easy", "arc_challenge"]
MODELS = ["google/gemma-2-2b", "google/gemma-2-2b-it", "meta-llama/Llama-3.2-3B",
          "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3-4B"]


def examples(task, model_name="google/gemma-2-2b", n=50):
    random.seed(42)
    ds = get_dataset(task, num_examples=n, digits=3 if task == "addition" else 2)
    return apply_few_shot_prefix(list(ds), task, model_name)


@pytest.mark.parametrize("task", LABEL_TASKS + ["boolean"])
def test_targets_continue_the_prompt(task):
    """Label answers follow the prompt after a space, so the prompt must not end in one."""
    for ex in examples(task):
        assert not ex.prompt[-1].isspace() and not ex.corrupted_prompt[-1].isspace()
        assert ex.target.startswith(" ") and not ex.target[1].isspace()
        assert ex.corrupted_target.startswith(" ")


@pytest.mark.parametrize("task", LABEL_TASKS)
def test_gold_label_matches_target(task):
    for ex in examples(task):
        assert ex.labels[ex.answer_idx] == ex.target.strip()


@pytest.mark.parametrize("task", ["ioi", "mcqa", "arc_easy", "arc_challenge", "boolean"])
def test_counterfactual_changes_the_answer(task):
    assert all(ex.target != ex.corrupted_target for ex in examples(task))


@pytest.mark.parametrize("task", TASKS)
def test_no_example_is_its_own_counterfactual(task):
    """Identical clean and corrupted inputs make every attribution score zero."""
    assert all(ex.prompt != ex.corrupted_prompt for ex in examples(task, n=500))


class FakeModel:
    """Character-level tokenizer: every string's first token is its first character."""

    def to_tokens(self, text, prepend_bos=False):
        ids = [ord(c) for c in text]
        return torch.tensor([[0] + ids if prepend_bos else ids])


def test_labels_cannot_tie_through_a_shared_first_token():
    """A newline or colon logit must never decide between two labels."""
    model = FakeModel()
    logits = torch.zeros(256)
    logits[ord("\n")] = logits[ord(":")] = 10.0
    logits[ord("H")] = 1.0  # "Henry" beats "Phil"
    assert _score_first_token(logits, model, "Phil") != _score_first_token(logits, model, "Henry")
    assert _classify_ioi(logits, model, ["Phil", "Henry"]) == 1
    logits[ord("C")] = 2.0
    assert _classify_from_labels(logits, model, ["A", "B", "C", "D"]) == "C"


def tokenizer(name):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(name)
    except Exception as e:  # gated repo without a token, or offline
        pytest.skip(f"tokenizer {name} unavailable: {e}")


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("task", TASKS)
def test_answer_tokens_and_equal_lengths(model_name, task):
    """The target tokenizes on its own into the tokens the model should emit next, the
    labels start with distinct tokens, and clean and corrupted prompts have equal length."""
    tok = tokenizer(model_name)
    encode = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    for ex in examples(task, model_name):
        target_ids = encode(ex.target)
        assert target_ids, ex.target
        if task in LABEL_TASKS or task == "boolean":
            assert len(target_ids) == 1, (ex.target, tok.convert_ids_to_tokens(target_ids))
            # the string concatenation agrees with the token concatenation
            assert encode(ex.prompt + ex.target) == encode(ex.prompt) + target_ids
        assert len(encode(ex.prompt)) == len(encode(ex.corrupted_prompt)), ex.prompt
        if task in LABEL_TASKS:
            firsts = {encode(" " + label)[0] for label in ex.labels}
            assert len(firsts) == len(ex.labels)
