from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional
from datasets import load_dataset

@dataclass
class ArithmeticExample:
    prompt: str
    target: str

class AdditionDataset:
    def __init__(self, num_examples: int = 100, digits: int = 2) -> None:
        self.num_examples = num_examples
        self.digits = digits
        self._examples: List[ArithmeticExample] = []
        self._generate_examples()

    def _generate_examples(self) -> None:
        self._examples = []
        low = 10 ** (self.digits - 1)
        high = 10 ** self.digits - 1
        
        for _ in range(self.num_examples):
            a = random.randint(low, high)
            b = random.randint(low, high)
            prompt = f"Compute: {a} + {b} = "
            target = str(a + b)
            self._examples.append(ArithmeticExample(prompt, target))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> ArithmeticExample:
        return self._examples[idx]

    def __iter__(self) -> Iterable[ArithmeticExample]:
        return iter(self._examples)

class BooleanDataset:
    def __init__(
        self, 
        num_examples: int = 100, 
        min_ops: int = 2, 
        max_ops: int = 6, 
        allow_parentheses: bool = True, 
        allow_not: bool = True
    ) -> None:
        self.num_examples = num_examples
        self.min_ops = min_ops
        self.max_ops = max_ops
        self.allow_parentheses = allow_parentheses
        self.allow_not = allow_not
        self._examples: List[ArithmeticExample] = []
        self._generate_examples()

    def _rand_bool(self) -> str:
        return random.choice(['true', 'false'])

    def _maybe_not(self, token: str) -> str:
        if self.allow_not and random.random() < 0.3:
            return 'not ' + token
        return token

    def _gen_expr(self, remaining_ops: int) -> str:
        if remaining_ops == 0:
            atom = self._maybe_not(self._rand_bool())
            return atom
            
        left_ops = random.randint(0, remaining_ops - 1)
        right_ops = remaining_ops - 1 - left_ops
        left = self._gen_expr(left_ops)
        right = self._gen_expr(right_ops)
        op = random.choice(['and', 'or'])
        expr = f"{left} {op} {right}"
        
        if self.allow_parentheses and random.random() < 0.5:
            expr = f"({expr})"
        return expr

    def _evaluate(self, expr: str) -> bool:
        py_expr = expr.replace('true', 'True').replace('false', 'False')
        return bool(eval(py_expr))

    def _generate_examples(self) -> None:
        self._examples = []
        seen = set()
        attempts = 0
        max_attempts = self.num_examples * 20
        
        while len(self._examples) < self.num_examples and attempts < max_attempts:
            attempts += 1
            n_ops = random.randint(self.min_ops, self.max_ops)
            expr = self._gen_expr(n_ops)
            
            if expr in seen:
                continue
                
            seen.add(expr)
            val = str(self._evaluate(expr)).lower()
            prompt = f"Evaluate: {expr} = "
            self._examples.append(ArithmeticExample(prompt, val))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> ArithmeticExample:
        return self._examples[idx]

    def __iter__(self) -> Iterable[ArithmeticExample]:
        return iter(self._examples)

class MMLUDataset:
    def __init__(
        self, 
        subject: str = "high_school_european_history", 
        split: str = "test", 
        num_examples: int | None = None
    ) -> None:
        ds = load_dataset("cais/mmlu", subject, split=split)
        self._examples: List[ArithmeticExample] = []
        
        for i, item in enumerate(ds):
            q = item["question"]
            choices = item["choices"]
            ans_idx = item["answer"]
            
            prompt = (
                f"{q}\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n"
                f"Answer: "
            )
            target = chr(ord('A') + ans_idx)
            self._examples.append(ArithmeticExample(prompt, target))
            
            if num_examples is not None and i + 1 >= num_examples:
                break

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> ArithmeticExample:
        return self._examples[idx]

    def __iter__(self) -> Iterable[ArithmeticExample]:
        return iter(self._examples)

class MIBDataset:
    def __init__(
        self, 
        name: str = "ioi", 
        split: str = "test", 
        num_examples: int | None = None
    ) -> None:
        ds = load_dataset(f"mib-bench/{name}", split=split)
        self._examples: List[ArithmeticExample] = []
        count = 0
        
        for item in ds:
            prompt_text = (
                item.get("prompt") or 
                item.get("template") or 
                item.get("question")
            )
            choices = item.get("choices", [])
            answer_key = item.get("answerKey", -1)
            
            if (prompt_text is None or 
                not choices or 
                answer_key is None or 
                answer_key < 0):
                continue
                
            parts = [
                f"({chr(ord('A') + idx)}) {c}" 
                for idx, c in enumerate(choices)
            ]
            prompt = f"{prompt_text} " + " ".join(parts)
            
            if answer_key >= len(choices):
                continue
                
            target = chr(ord('A') + answer_key)
            self._examples.append(ArithmeticExample(prompt.strip(), target))
            count += 1
            
            if num_examples is not None and count >= num_examples:
                break

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> ArithmeticExample:
        return self._examples[idx]

    def __iter__(self) -> Iterable[ArithmeticExample]:
        return iter(self._examples)

DATASET_DISPLAY_NAMES: dict[str, str] = {
    "addition": "Addition",
    "boolean": "Boolean",
    "mmlu": "MMLU",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "qwen3-0.6b": "Qwen3-0.6B",
    "qwen3-1.7b": "Qwen3-1.7B",
}

def get_task_display_name(task: str) -> str:
    if task in DATASET_DISPLAY_NAMES:
        return DATASET_DISPLAY_NAMES[task]
        
    if task.startswith("mib_"):
        suffix = task[len("mib_"):].replace("_", " ").title()
        return f"MIB: {suffix}"
        
    return task.replace("_", " ").title()

def get_model_display_name(model: str) -> str:
    if model in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model]
        
    parts = model.split('-')
    if not parts:
        return model
        
    first = parts[0].title()
    rest = []
    
    for p in parts[1:]:
        if p.endswith('b') and p[:-1].replace('.', '').isdigit():
            rest.append(p[:-1] + p[-1].upper())
        else:
            rest.append(p)
            
    return " ".join([first] + rest)

def get_dataset(
    task: str, 
    num_examples: int = 100, 
    digits: int = 2
) -> Iterable[ArithmeticExample]:
    if task == "addition":
        return AdditionDataset(num_examples=num_examples, digits=digits)
    if task == "boolean":
        return BooleanDataset(num_examples=num_examples)
    if task == "mmlu":
        return MMLUDataset(split='test', num_examples=num_examples)
    if task.startswith("mib_"):
        return MIBDataset(
            name=task[len("mib_"):], 
            split="test", 
            num_examples=num_examples
        )
    raise ValueError(f"Unsupported task: {task}")

def _corrupt_addition(prompt: str) -> str:
    """Counterfactual: preserve the 'Compute: a + b =' format but replace each 
    term with a random number of the same length.
    """
    m = re.search(r"Compute:\s*(\d+)\s*\+\s*(\d+)\s*=", prompt)
    if not m:
        raise ValueError(f"Addition prompt not in expected format: {prompt!r}")
        
    a, b = m.group(1), m.group(2)

    def rnd_like(x: str) -> str:
        return "".join(str(random.randint(0, 9)) for _ in x)

    aa = rnd_like(a)
    bb = rnd_like(b)

    # Replace using match spans to preserve original spacing and punctuation
    a_start, a_end = m.span(1)
    b_start, b_end = m.span(2)
    return prompt[:a_start] + aa + prompt[a_end:b_start] + bb + prompt[b_end:]

def _corrupt_boolean(prompt: str) -> str:
    """Counterfactual: swap logical operators (and <-> or) using a temp token
    to avoid overlap, and flip boolean literals (true <-> false), preserving
    structure and any parentheses.
    """
    s = prompt
    s = re.sub(r"\band\b", "__TMP_OR__", s)
    s = re.sub(r"\bor\b", "and", s)
    s = re.sub(r"__TMP_OR__", "or", s)
    
    def flip_bool(match):
        return "true" if match.group(0) == "false" else "false"
        
    s = re.sub(r"\btrue\b|\bfalse\b", flip_bool, s)
    return s

def _corrupt_mc(prompt: str) -> str:
    """Counterfactual: detect choices labeled A.-D., shuffle their contents,
    and relabel sequentially (A-D), keeping the question text and trailing 
    'Answer:'.
    """
    lines = prompt.splitlines()
    choice_lines = []
    other_lines = []
    
    for ln in lines:
        if re.match(r"^[A-D]\.\s", ln.strip()):
            choice_lines.append(ln.strip())
        else:
            other_lines.append(ln)
            
    if not choice_lines:
        raise ValueError(f"MC prompt not in expected format: {prompt!r}")
        
    contents = [re.sub(r"^[A-D]\.\s*", "", x) for x in choice_lines]
    random.shuffle(contents)
    new_choices = [
        f"{chr(ord('A')+i)}. {contents[i]}" 
        for i in range(min(4, len(contents)))
    ]
    
    return "\n".join(other_lines + new_choices + ["Answer: "])

def corrupt_prompt_for_eap(
    prompt: str, 
    task_hint: Optional[str] = None
) -> str:
    """Corrupt the prompt for the EAP task. """
    if task_hint == "addition" or prompt.lower().startswith("compute:"):
        return _corrupt_addition(prompt)
        
    if (task_hint == "boolean" or 
        ("true" in prompt or 
         "false" in prompt or 
         " and " in prompt or 
         " or " in prompt)):
        return _corrupt_boolean(prompt)
        
    if ("Answer:" in prompt and 
        ("A." in prompt or 
         "B." in prompt or 
         "C." in prompt or 
         "D." in prompt)):
        return _corrupt_mc(prompt)
        
    toks = prompt.split()
    if len(toks) <= 1:
        return prompt[::-1]
        
    random.shuffle(toks)
    return " ".join(toks)

__all__ = [
    "ArithmeticExample",
    "AdditionDataset",
    "BooleanDataset",
    "MMLUDataset",
    "MIBDataset",
    "get_dataset",
    "get_task_display_name",
    "get_model_display_name",
    "corrupt_prompt_for_eap",
]
