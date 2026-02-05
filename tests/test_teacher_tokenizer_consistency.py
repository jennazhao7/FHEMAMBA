import os
import re
from pathlib import Path

import torch
import pytest
from datasets import load_dataset  # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore


ROOT = Path(__file__).resolve().parents[1]


def _extract_var_from_file(path: Path, var_name: str) -> str | None:
    text = path.read_text()
    m = re.search(rf"{re.escape(var_name)}\s*=\s*\"([^\"]+)\"", text)
    return m.group(1) if m else None


def _extract_int_from_file(path: Path, var_name: str) -> int | None:
    text = path.read_text()
    m = re.search(rf"{re.escape(var_name)}\s*=\s*(\d+)", text)
    return int(m.group(1)) if m else None


def test_static_tokenizer_and_max_len_match():
    gpt_train = ROOT / "gpt_train.py"
    dump_teacher = ROOT / "dump_teacher.py"

    # Extract tokenizer/model ids
    teacher_model_id = _extract_var_from_file(gpt_train, "teacher_model_id")
    dump_model_id = _extract_var_from_file(dump_teacher, "MODEL_ID")

    # Extract max_len used (first assignment occurrence)
    train_max_len = _extract_int_from_file(gpt_train, "max_len")
    # in dump script max_len is passed as a literal in make_loader(..., max_len=128)
    dump_text = dump_teacher.read_text()
    dump_max_match = re.search(r"make_loader\(.*max_len\s*=\s*(\d+)", dump_text)
    dump_max_len = int(dump_max_match.group(1)) if dump_max_match else None

    assert teacher_model_id is not None, "teacher_model_id not found in gpt_train.py"
    assert dump_model_id is not None, "MODEL_ID not found in dump_teacher.py"
    assert teacher_model_id == dump_model_id, (
        f"Tokenizer/model mismatch: gpt_train={teacher_model_id}, dump={dump_model_id}"
    )
    assert train_max_len is not None and dump_max_len is not None, "max_len not found"
    assert train_max_len == dump_max_len, (
        f"max_len mismatch: gpt_train={train_max_len}, dump={dump_max_len}"
    )


@pytest.mark.slow
def test_runtime_subset_predictions_align_with_saved_logits():
    logits_path = ROOT / "teacher_sst2_logits.pt"
    assert logits_path.exists(), f"Missing file: {logits_path}"

    pack = torch.load(str(logits_path), map_location="cpu")
    val_logits = torch.as_tensor(pack["validation"]["logits"]).float()
    val_labels = torch.as_tensor(pack["validation"]["labels"]).long()
    assert val_logits.ndim == 2 and val_logits.size(-1) == 2
    assert val_labels.ndim == 1 and val_labels.size(0) == val_logits.size(0)

    # Read settings from source files
    gpt_train = ROOT / "gpt_train.py"
    dump_teacher = ROOT / "dump_teacher.py"
    teacher_model_id = _extract_var_from_file(gpt_train, "teacher_model_id") or _extract_var_from_file(dump_teacher, "MODEL_ID")
    assert teacher_model_id is not None, "Cannot determine teacher model id"
    max_len = _extract_int_from_file(gpt_train, "max_len")
    assert max_len is not None, "Cannot determine max_len from gpt_train.py"

    tok = AutoTokenizer.from_pretrained(teacher_model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(teacher_model_id)
    model.eval()

    # Build the same validation dataset order (no shuffling) and tokenize
    ds = load_dataset("glue", "sst2")["validation"]
    enc = ds.map(lambda b: tok(b["sentence"], truncation=True, padding="max_length", max_length=max_len), batched=True)
    enc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])  # type: ignore

    # Sample a small, deterministic subset
    torch.manual_seed(0)
    idxs = torch.linspace(0, len(enc) - 1, steps=min(64, len(enc))).long().tolist()
    sample = {k: v[idxs] for k, v in {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "label": enc["label"]}.items()}  # type: ignore

    with torch.no_grad():
        logits = model(input_ids=sample["input_ids"], attention_mask=sample["attention_mask"]).logits.cpu()

    saved_preds = val_logits[idxs].argmax(-1)
    recomputed_preds = logits.argmax(-1)

    # Expect near-identical predictions if tokenizer + max_len match
    match_rate = (saved_preds == recomputed_preds).float().mean().item()
    assert match_rate >= 0.98, f"Prediction mismatch rate too high: match_rate={match_rate:.3f}"


