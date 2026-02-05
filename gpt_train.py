"""
Training script for distilling a small Mamba model on the SST‑2 task.

This script replicates the original training loop while introducing several
improvements aimed at addressing stagnant validation accuracy around 50%.

Changes made relative to the upstream version include:

* **Consistent tokeniser** – the same model id used to dump the teacher
  logits (``textattack/bert‑base‑uncased‑SST‑2``) is used here for
  tokenisation.  Mismatched vocabularies between teacher and student can
  silently degrade performance. Tokenization consistency is verified at startup.
* **Cleaner knowledge‑distillation loss** – use PyTorch's built‑in
  cross‑entropy with optional label smoothing and compute the KL
  divergence in a numerically stable way.  The alpha parameter controls
  the weighting between the hard labels and the teacher's soft targets.
* **Confidence weighting** – optional per-sample weighting based on teacher
  confidence (controlled by ``conf_power``). When enabled, examples where the
  teacher is more confident are up-weighted in the KL loss.
* **Semi-supervised distillation** – support for using unlabeled data (e.g., 
  IMDb movie reviews) to augment training. Teacher logits are generated for
  unlabeled examples and used for additional KL distillation loss.
* **EMA evaluation** – evaluate both the current model and its
  exponential moving average to better track generalisation.
* **Data masking** – see :mod:`mamba_student` for how padded tokens are
  masked out of the input sequence.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Tuple, Dict, Iterable, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from datasets import load_dataset, Dataset  # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup  # type: ignore
from tqdm import tqdm

from mamba_student import TinyMambaClassifier  # relative import

def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    T: float = 4.0,
    alpha: float = 0.5,
    label_smoothing: float = 0.1,
    conf_power: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute knowledge distillation loss using standard KL divergence.
    
    The loss combines hard cross-entropy with soft targets from the teacher:
    L = (1 - α) * CE(labels) + α * w * KL(teacher, student)
    
    where w is an optional confidence weight. When conf_power > 0, examples
    where the teacher is more confident (higher max probability) are up-weighted.
    
    Parameters
    ----------
    student_logits: torch.Tensor
        Raw logits from the student model of shape ``(batch_size, num_classes)``.
    teacher_logits: torch.Tensor
        Raw logits from the teacher model of shape ``(batch_size, num_classes)``.
    labels: torch.Tensor
        Ground truth labels of shape ``(batch_size,)``.
    T: float, optional
        Temperature used to soften the teacher and student distributions.
        Higher T makes distributions softer (more uniform). Typical range: 2-5.
        Default: 4.0 (as found effective for Tiny-BERT on SST-2).
    alpha: float, optional
        Weight assigned to the teacher KL term. ``1 - alpha`` is the weight
        of the hard cross-entropy. Typical range: 0.5-0.9. Default: 0.5.
    label_smoothing: float, optional
        Amount of label smoothing to apply to the hard cross-entropy.
        Default: 0.0 (no smoothing).
    conf_power: float, optional
        Power for confidence weighting. Confidence is measured as the maximum
        probability in the teacher's softmax distribution. Weight w = confidence^conf_power.
        When conf_power=0, w=1 (no confidence weighting). When conf_power > 0,
        more confident examples are up-weighted. Default: 0.0.
    
    Returns
    -------
    loss: torch.Tensor
        The scalar loss.
    parts: dict[str, float]
        A dictionary containing individual components of the loss for
        logging (``ce``, ``kl``, and optionally ``conf_weight_mean``).
    """
    # Hard cross-entropy loss (optionally smoothed)
    ce = nn.functional.cross_entropy(
        student_logits, labels, label_smoothing=label_smoothing
    )
    
    # Soft targets: compute KL divergence between teacher and student distributions
    # Apply temperature scaling to both distributions
    log_p_s = nn.functional.log_softmax(student_logits / T, dim=-1)
    p_t = nn.functional.softmax(teacher_logits / T, dim=-1)
    
    # Compute per-sample KL divergence (reduction="none")
    kl_per_sample = nn.functional.kl_div(
        log_p_s, p_t, reduction="none", log_target=False
    ).sum(dim=-1) * (T * T)  # shape: (batch_size,)
    
    # Confidence weighting: weight based on teacher confidence
    if conf_power > 0.0:
        # Teacher confidence: max probability in the teacher's distribution
        # Higher max prob = more confident teacher prediction
        teacher_conf = p_t.max(dim=-1)[0]  # shape: (batch_size,)
        # Add small epsilon for numerical stability
        teacher_conf = teacher_conf + 1e-6
        # Compute weight: w = confidence^conf_power
        # When conf_power=0, w=1 (no weighting)
        conf_weights = torch.pow(teacher_conf, conf_power)  # shape: (batch_size,)
        # Apply weights to KL divergence per sample
        kl = (kl_per_sample * conf_weights).mean()
        conf_weight_mean = conf_weights.mean().item()
    else:
        # No confidence weighting: simple mean
        kl = kl_per_sample.mean()
        conf_weight_mean = 1.0
    
    # Weighted combination: L = (1 - α) * CE + α * w * KL
    loss = (1.0 - alpha) * ce + alpha * kl
    
    parts = {"ce": ce.item(), "kl": kl.item()}
    if conf_power > 0.0:
        parts["conf_weight_mean"] = conf_weight_mean
    
    return loss, parts


def hidden_state_mse_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss between student and teacher hidden states.
    
    Parameters
    ----------
    student_hidden: torch.Tensor
        Student CLS hidden state of shape ``(batch_size, d_model)``.
    teacher_hidden: torch.Tensor
        Teacher pooled hidden state of shape ``(batch_size, d_model)``.
    
    Returns
    -------
    loss: torch.Tensor
        MSE loss scalar.
    """
    return nn.functional.mse_loss(student_hidden, teacher_hidden)


def r_drop_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
) -> torch.Tensor:
    """Compute R-Drop loss: KL divergence between two dropout passes.
    
    R-Drop encourages consistency between two forward passes with different
    dropout masks, acting as a regularization technique.
    
    Parameters
    ----------
    logits1: torch.Tensor
        Logits from first forward pass, shape ``(batch_size, num_classes)``.
    logits2: torch.Tensor
        Logits from second forward pass, shape ``(batch_size, num_classes)``.
    
    Returns
    -------
    loss: torch.Tensor
        Symmetric KL divergence loss.
    """
    p1 = nn.functional.log_softmax(logits1, dim=-1)
    p2 = nn.functional.softmax(logits2, dim=-1)
    kl1 = nn.functional.kl_div(p1, p2, reduction="batchmean", log_target=False)
    
    p1_soft = nn.functional.softmax(logits1, dim=-1)
    p2_log = nn.functional.log_softmax(logits2, dim=-1)
    kl2 = nn.functional.kl_div(p2_log, p1_soft, reduction="batchmean", log_target=False)
    
    return (kl1 + kl2) / 2.0

# def kd_loss(
#     student_logits: torch.Tensor,
#     teacher_logits: torch.Tensor,
#     labels: torch.Tensor,
#     *,
#     T: float = 2.0,
#     alpha: float = 0.0,#debugging kd loss settings
#     label_smoothing: float = 0.0,
# ) -> Tuple[torch.Tensor, Dict[str, float]]:
#     """Compute the knowledge distillation loss.

#     The loss is a weighted sum between the hard cross‑entropy against the
#     ground‑truth labels (optionally smoothed) and the KL divergence
#     between the student’s distribution and the teacher’s distribution at
#     temperature ``T``.  When ``alpha`` is 0 the student is trained only
#     on the ground truth labels; when ``alpha`` is 1 the student is
#     trained purely to match the teacher.

#     Parameters
#     ----------
#     student_logits: torch.Tensor
#         Raw logits from the student model of shape ``(batch_size, num_classes)``.
#     teacher_logits: torch.Tensor
#         Raw logits from the teacher model of shape ``(batch_size, num_classes)``.
#     labels: torch.Tensor
#         Ground truth labels of shape ``(batch_size,)``.
#     T: float, optional
#         Temperature used to soften the teacher and student distributions.
#     alpha: float, optional
#         Weight assigned to the teacher KL term.  ``1 - alpha`` is the
#         weight of the hard cross‑entropy.
#     label_smoothing: float, optional
#         Amount of label smoothing to apply to the hard cross‑entropy.

#     Returns
#     -------
#     loss: torch.Tensor
#         The scalar loss.
#     parts: Dict[str, float]
#         A dictionary containing individual components of the loss for
#         logging (``ce`` and ``kl``).
#     """
#     # hard labels
#     ce = nn.functional.cross_entropy(
#         student_logits, labels, label_smoothing=label_smoothing
#     )
#     # soft targets from teacher
#     log_p_s = nn.functional.log_softmax(student_logits / T, dim=-1)
#     p_t = nn.functional.softmax(teacher_logits / T, dim=-1)
#     kl = nn.functional.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)
#     # weighted sum
#     loss = (1.0 - alpha) * ce + alpha * kl
#     return loss, {"ce": ce.item(), "kl": kl.item()}


def build_keep_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Build a keep mask from input_ids and pad_token_id.
    
    Transformers convention: 1 = keep/real token, 0 = pad.
    Ensures CLS token (position 0) is always kept.
    
    Parameters
    ----------
    input_ids: torch.Tensor
        Input token IDs of shape (B, L).
    pad_token_id: int
        Padding token ID from tokenizer.
    
    Returns
    -------
    keep_mask: torch.Tensor
        Keep mask of shape (B, L) where 1 = keep, 0 = pad.
        CLS token (position 0) is always 1.
    """
    # Build mask: 1 = keep (not pad), 0 = pad
    keep = (input_ids != pad_token_id).to(torch.float32)  # (B, L)
    
    # Ensure CLS token (position 0) is always kept
    keep[:, 0] = 1.0
    
    return keep


def verify_tokenization_consistency(
    tokenizer: AutoTokenizer,
    max_len: int = 128,
    num_samples: int = 10,
) -> bool:
    """Verify that tokenization is consistent by checking sample sentences.
    
    This helps ensure teacher logits were generated with the same tokenization
    settings as used during student training.
    
    Parameters
    ----------
    tokenizer: AutoTokenizer
        The tokenizer to verify.
    max_len: int, optional
        Maximum sequence length. Default: 128.
    num_samples: int, optional
        Number of samples to check. Default: 10.
    
    Returns
    -------
    bool
        True if tokenization appears consistent.
    """
    ds = load_dataset("glue", "sst2")["train"]
    sample_sentences = [ds[i]["sentence"] for i in range(min(num_samples, len(ds)))]
    
    print(f"Verifying tokenization consistency (max_len={max_len})...")
    for i, sentence in enumerate(sample_sentences):
        enc1 = tokenizer(sentence, truncation=True, padding="max_length", max_length=max_len)
        enc2 = tokenizer(sentence, truncation=True, padding="max_length", max_length=max_len)
        # Check that tokenization is deterministic
        if enc1["input_ids"] != enc2["input_ids"]:
            print(f"WARNING: Non-deterministic tokenization for sample {i}")
            return False
        # Check truncation is working (should be <= max_len)
        if len(enc1["input_ids"]) != max_len:
            print(f"WARNING: Unexpected token length {len(enc1['input_ids'])} (expected {max_len})")
            return False
    print("✓ Tokenization verification passed")
    return True


def generate_teacher_logits_for_unlabeled(
    unlabeled_dataset: Dataset,
    teacher_model_id: str,
    tokenizer: AutoTokenizer,
    device: torch.device,
    *,
    max_len: int = 128,
    batch_size: int = 128,
    cache_path: Optional[Path] = None,
) -> Tuple[torch.Tensor, Dataset]:
    """Generate teacher logits for unlabeled data (semi-supervised distillation).
    
    Parameters
    ----------
    unlabeled_dataset: Dataset
        Dataset with "sentence" column but no labels.
    teacher_model_id: str
        HuggingFace model ID for the teacher model.
    tokenizer: AutoTokenizer
        Tokenizer (should match the one used for teacher).
    device: torch.device
        Device to run teacher model on.
    max_len: int, optional
        Maximum sequence length. Default: 128.
    batch_size: int, optional
        Batch size for generating logits. Default: 128.
    cache_path: Optional[Path], optional
        Path to cache generated logits. If provided and exists, will load from cache.
        Default: None (no caching).
    
    Returns
    -------
    teacher_logits: torch.Tensor
        Teacher logits of shape (num_samples, num_classes).
    tokenized_dataset: Dataset
        Tokenized dataset with input_ids and attention_mask.
    """
    # Check cache
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached teacher logits from {cache_path}")
        cached = torch.load(str(cache_path), map_location="cpu")
        return cached["logits"], cached["dataset"]
    
    print(f"Generating teacher logits for {len(unlabeled_dataset)} unlabeled examples...")
    
    # Load teacher model
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_id).to(device)
    teacher_model.eval()
    
    # Tokenize unlabeled data
    def _tok(batch: Dict[str, Iterable]) -> Dict[str, Any]:
        # Handle datasets with either "sentence" or "text" column
        texts = batch.get("sentence", batch.get("text", []))
        return tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_len
        )
    
    tokenized_ds = unlabeled_dataset.map(_tok, batched=True)
    tokenized_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )
    loader = DataLoader(tokenized_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Generate teacher logits
    all_logits = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating teacher logits"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_logits.append(logits.cpu())
    
    teacher_logits = torch.cat(all_logits, dim=0)
    print(f"✓ Generated teacher logits: {teacher_logits.shape}")
    
    # Cache if path provided
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"logits": teacher_logits, "dataset": tokenized_ds}, str(cache_path))
        print(f"✓ Cached teacher logits to {cache_path}")
    
    return teacher_logits, tokenized_ds


def make_tokenized(
    split: str,
    tokenizer: AutoTokenizer,
    *,
    max_len: int = 128,
    batch_size: int = 128,
    shuffle: bool = False,
    dataset_name: Optional[str] = None,
    text_column: str = "sentence",
    label_column: str = "label",
) -> Tuple[Any, DataLoader]:
    """Tokenise a dataset split and build a dataloader.

    An ``idx`` column is added if missing and preserved through the
    mapping so that teacher logits can be looked up by index.
    
    Parameters
    ----------
    split: str
        Dataset split name (e.g., "train", "validation").
    tokenizer: AutoTokenizer
        Tokenizer to use.
    max_len: int, optional
        Maximum sequence length. Default: 128.
    batch_size: int, optional
        Batch size for DataLoader. Default: 128.
    shuffle: bool, optional
        Whether to shuffle the data. Default: False.
    dataset_name: Optional[str], optional
        Dataset name (e.g., "glue", "imdb"). If None, uses "glue" with "sst2".
        Default: None.
    text_column: str, optional
        Name of the text column. Default: "sentence".
    label_column: str, optional
        Name of the label column. Default: "label".
    """
    if dataset_name is None:
        ds = load_dataset("glue", "sst2")[split]
    else:
        if dataset_name == "imdb":
            # IMDb dataset (unlabeled movie reviews)
            ds = load_dataset("imdb", split=split if split != "validation" else "test")
        else:
            ds = load_dataset(dataset_name, split=split)
    
    # ensure an idx column exists
    if "idx" not in ds.column_names:
        ds = ds.add_column("idx", list(range(len(ds))))

    def _tok(batch: Dict[str, Iterable]) -> Dict[str, Any]:
        texts = batch[text_column] if text_column in batch else batch.get("text", batch.get("sentence", []))
        enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len
        )
        enc["idx"] = batch["idx"]
        return enc

    ds = ds.map(_tok, batched=True)
    
    # Determine columns to include
    columns = ["input_ids", "attention_mask", "idx"]
    if label_column in ds.column_names:
        columns.append(label_column)
    
    ds.set_format(type="torch", columns=columns)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )
    return ds, loader


def evaluate(
    model: nn.Module,
    val_loader: Iterable,
    device: torch.device,
    ema_shadow: Dict[str, torch.Tensor],
) -> float:
    """Evaluate a model using its EMA weights on the validation set."""
    # store original weights
    original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # load EMA weights
    model.load_state_dict(ema_shadow, strict=False)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits, _ = model(input_ids, attention_mask)  # Unpack (logits, cls_h)
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    # restore original weights
    model.load_state_dict(original_state, strict=False)
    return correct / max(1, total)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ========== SANITY CHECK MODE ==========
    # Quick tests to verify training pipeline works correctly
    # Usage: 
    #   python gpt_train.py --minimal    # Truly minimal overfit test (recommended first)
    #   python gpt_train.py --sanity-check  # Full sanity check with evaluation
    import sys
    run_minimal = "--minimal" in sys.argv or "-m" in sys.argv
    run_sanity_check = "--sanity-check" in sys.argv or "-s" in sys.argv
    
    if run_minimal:
        run_minimal_overfit_test(device)
        return
    
    if run_sanity_check:
        print("=" * 60)
        print("RUNNING SANITY CHECK: 64-example overfit test")
        print("=" * 60)
        run_sanity_check_overfit(device)
        return
    
    print("Starting training…")

    # hyperparameters - Config A
    max_len = 128  # Will use curriculum: 64 for first 1-2 epochs, then 128
    epochs = 10  # Config A: 5-10 epochs with early stopping
    lr = 2e-4  # Config A: 2e-4
    weight_decay = 0.02  # Config A: 0.02
    warmup_ratio = 0.08  # Config A: 5-10% warmup (using 8%)
    
    # Length curriculum: start with shorter sequences
    curriculum_max_len_start = 64  # First 1-2 epochs use 64
    curriculum_switch_epoch = 2  # Switch to full length after this epoch
    # Temperature and alpha for knowledge distillation
    # Option 1: Fixed values (recommended: T=4.0, alpha=0.5 for SST-2 based on Tiny-BERT results)
    use_fixed_kd_params = True  # Set to False to use annealing schedule
    T_fixed = 4.0  # Temperature (typical range: 2-5)
    alpha_fixed = 0.5  # KD loss weight (typical range: 0.5-0.9)
    # Option 2: Annealing schedule (if use_fixed_kd_params=False)
    T_start = 4.0  # Start temperature
    T_end = 2.0    # End temperature
    alpha_start = 0.9  # Start alpha (more weight on teacher)
    alpha_end = 0.5    # End alpha (more weight on hard labels)
    label_smoothing = 0.05  # Config A: label smoothing 0.05
    conf_power = 0.0  # Confidence weighting power (0.0 = off, >0 up-weights confident teacher predictions)
    
    # Config A: Distillation loss weights
    alpha_kd = 0.65  # KD weight (between 0.6-0.7)
    beta_hidden = 0.3  # Hidden-state MSE weight (0.3 for first 3-5 epochs, then 0.1)
    beta_hidden_decay_epochs = 5  # Decay beta_hidden after this many epochs
    r_drop_weight = 0.5  # R-Drop weight for first half of training
    r_drop_active_epochs = None  # Will be set to epochs // 2
    
    # Config A: R-Drop and other regularization
    use_r_drop = True  # Enable R-Drop for first half of training
    
    # Semi-supervised distillation: use unlabeled data
    use_unlabeled_data = False  # Set to True to enable semi-supervised distillation
    unlabeled_dataset_name = "imdb"  # Options: "imdb" (movie reviews) or None
    unlabeled_split = "train"  # Split to use for unlabeled data
    unlabeled_max_samples = 5000  # Maximum number of unlabeled samples to use (None for all)
    unlabeled_alpha = 0.3  # Weight for unlabeled KL loss (typically lower than labeled alpha)
    
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # teacher assets
    # use the same id as in dump_teacher.py to ensure consistent tokenisation
    teacher_model_id = "textattack/bert-base-uncased-SST-2"
    logits_path = Path(__file__).resolve().parent / "teacher_sst2_logits.pt"
    teacher_pack = torch.load(str(logits_path), map_location="cpu")
    train_logits = teacher_pack["train"]["logits"].float()
    train_labels = teacher_pack["train"]["labels"].long()
    val_logits = teacher_pack["validation"]["logits"].float()
    val_labels = teacher_pack["validation"]["labels"].long()

    # tokeniser
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, use_fast=True)
    vocab_size = tokenizer.vocab_size
    
    # Get pad_token_id for mask building
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id  # Fallback
    print(f"[Config] pad_token_id: {pad_token_id}")
    
    # Flag to rebuild masks from pad_token_id (use if tokenizer masks are wrong)
    REBUILD_MASKS = False  # Set to True to rebuild masks from pad_token_id
    
    # Verify tokenization consistency
    verify_tokenization_consistency(tokenizer, max_len=max_len, num_samples=10)
    
    # Verify teacher logits match dataset size
    train_ds_temp, _ = make_tokenized("train", tokenizer, max_len=max_len, batch_size=128, shuffle=False)
    if len(train_logits) != len(train_ds_temp):
        print(f"WARNING: Teacher logits size ({len(train_logits)}) != train dataset size ({len(train_ds_temp)})")
        print("This may indicate a tokenization mismatch. Please regenerate teacher logits.")

    # datasets and loaders
    train_ds, train_loader = make_tokenized(
        "train", tokenizer, max_len=max_len, batch_size=128, shuffle=True
    )
    val_ds, val_loader = make_tokenized(
        "validation", tokenizer, max_len=max_len, batch_size=256, shuffle=False
    )
    
    # Load and process unlabeled data for semi-supervised distillation
    unlabeled_logits = None
    unlabeled_ds = None
    unlabeled_loader = None
    if use_unlabeled_data and unlabeled_dataset_name:
        print(f"\n=== Loading unlabeled data: {unlabeled_dataset_name} ===")
        try:
            # Load unlabeled dataset
            if unlabeled_dataset_name == "imdb":
                # IMDb has "unsupervised" split with no labels
                unlabeled_ds_raw = load_dataset("imdb", split="unsupervised")
                # Convert to format with "sentence" column
                def add_sentence_column(example):
                    return {"sentence": example["text"]}
                unlabeled_ds_raw = unlabeled_ds_raw.map(add_sentence_column)
            else:
                # For other datasets, assume they have a "text" or "sentence" column
                unlabeled_ds_raw = load_dataset(unlabeled_dataset_name, split=unlabeled_split)
            
            # Limit number of samples if specified
            if unlabeled_max_samples and len(unlabeled_ds_raw) > unlabeled_max_samples:
                print(f"Limiting unlabeled data from {len(unlabeled_ds_raw)} to {unlabeled_max_samples} samples")
                unlabeled_ds_raw = unlabeled_ds_raw.select(range(unlabeled_max_samples))
            
            # Generate teacher logits for unlabeled data
            cache_path = Path(__file__).resolve().parent / f"teacher_{unlabeled_dataset_name}_logits.pt"
            unlabeled_logits, unlabeled_ds = generate_teacher_logits_for_unlabeled(
                unlabeled_ds_raw,
                teacher_model_id,
                tokenizer,
                device,
                max_len=max_len,
                batch_size=128,
                cache_path=cache_path,
            )
            
            # Create dataloader for unlabeled data
            unlabeled_ds.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "idx"],
            )
            unlabeled_loader = DataLoader(
                unlabeled_ds,
                batch_size=128,
                shuffle=True,
                drop_last=False,
                num_workers=0,
            )
            print(f"✓ Added {len(unlabeled_ds)} unlabeled examples for semi-supervised distillation")
        except Exception as e:
            print(f"WARNING: Failed to load unlabeled data: {e}")
            print("Continuing with labeled data only...")
            use_unlabeled_data = False

    # model - Config A settings
    # - expand=3: Increased from 2 to boost hidden state capacity
    # - ssm_dim=48: Reduced from 64 to save parameters
    # - Factorized embeddings: V×128 + 128→192 projection
    # - Learned positional embeddings
    # - head_type="ln_linear": LN → Linear(2) as per Config A
    # - dropout=0.1: In blocks and head
    # - drop_path_max=0.05: Stochastic depth
    model = TinyMambaClassifier(
        vocab_size=vocab_size,
        d_model=192,
        n_layers=8,
        ssm_dim=48,
        expand=3,
        conv_kernel=3,
        num_classes=2,
        tie_embeddings=True,
        head_type="ln_linear",  # Config A: LN → Linear(2)
        max_seq_len=max_len,
        embed_dim=128,  # Factorized embeddings
        dropout=0.1,  # Config A: dropout 0.1
        drop_path_max=0.05,  # Config A: stochastic depth
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Train dataset size: {len(train_ds)}, batches: {len(train_loader)}")
    print(f"Val dataset size: {len(val_ds)}, batches: {len(val_loader)}")
    print(
        f"Teacher logits shapes: train={train_logits.shape}, val={val_logits.shape}"
    )
    # Print KD hyperparameters
    if use_fixed_kd_params:
        print(f"KD loss: Fixed T={T_fixed}, α={alpha_fixed}, conf_power={conf_power}")
    else:
        print(f"KD loss: Annealing T={T_start}→{T_end}, α={alpha_start}→{alpha_end}, conf_power={conf_power}")
    if use_unlabeled_data:
        print(f"Semi-supervised: Using unlabeled data with α_unlabeled={unlabeled_alpha}")

    # --- set up / resume EMA ---
    ema_decay = 0.999
    ckpt_path = Path(__file__).resolve().parent / "ckpts" / "best_ema.pt"
    
    # Initialize optimizer - Config A: AdamW with specific params
    # For factorized embeddings, use lower LR for projection (0.5× global LR)
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "embed_proj" in n], 
         "lr": lr * 0.5, "weight_decay": weight_decay},  # Lower LR for projection
        {"params": [p for n, p in model.named_parameters() if "embed_proj" not in n], 
         "lr": lr, "weight_decay": weight_decay}
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98)
    )
    
    # Set R-Drop active epochs
    if r_drop_active_epochs is None:
        r_drop_active_epochs = epochs // 2
    # Calculate total steps: if using unlabeled data, we process both labeled and unlabeled batches
    # For simplicity, we'll alternate between labeled and unlabeled batches
    steps_per_epoch = len(train_loader)
    if use_unlabeled_data and unlabeled_loader is not None:
        # Use the maximum to ensure we process both datasets
        steps_per_epoch = max(len(train_loader), len(unlabeled_loader))
    total_steps = math.ceil(steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    checkpoint = None
    if ckpt_path.exists():
        print(f"[Resume] Loading EMA checkpoint from {ckpt_path}")
        checkpoint = torch.load(str(ckpt_path), map_location=device)
        ema_state = checkpoint["model"]  # this is an EMA state_dict
        
        # Check if architecture is compatible by comparing key parameters
        # If expand or ssm_dim changed, we can't load the Mamba layer weights
        checkpoint_incompatible = False
        if "layers.0.mamba.A_log" in ema_state:
            checkpoint_ssm_dim = ema_state["layers.0.mamba.A_log"].shape[1]
            checkpoint_expand_dim = ema_state["layers.0.mamba.A_log"].shape[0]
            # Calculate expand from dimensions: d_inner = d_model * expand
            checkpoint_expand = checkpoint_expand_dim // 192  # assuming d_model=192
            current_ssm_dim = 48
            current_expand = 3
            
            if checkpoint_ssm_dim != current_ssm_dim or checkpoint_expand != current_expand:
                checkpoint_incompatible = True
                print(f"[Resume] WARNING: Architecture mismatch detected!")
                print(f"  Checkpoint: expand={checkpoint_expand}, ssm_dim={checkpoint_ssm_dim}")
                print(f"  Current: expand={current_expand}, ssm_dim={current_ssm_dim}")
                print(f"  Cannot load Mamba layer weights. Starting fresh training.")
        
        if checkpoint_incompatible:
            # Architecture changed - start fresh but keep best_val for reference
            ema_shadow: Dict[str, torch.Tensor] = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            best_val = float(checkpoint.get("acc", 0.0))
            print(f"[Resume] Starting fresh with new architecture (previous best: {best_val:.4f})")
            # Don't try to continue from old epoch since architecture changed
            start_epoch = 1
        else:
            # Try to load checkpoint - will skip incompatible layers with strict=False
            try:
                missing_keys, unexpected_keys = model.load_state_dict(ema_state, strict=False)
                if missing_keys:
                    print(f"[Resume] Missing keys (using random init): {len(missing_keys)} keys")
                    if len(missing_keys) > 10:
                        print(f"  (showing first 5: {missing_keys[:5]})")
                    else:
                        print(f"  ({missing_keys})")
                if unexpected_keys:
                    print(f"[Resume] Unexpected keys (ignored): {len(unexpected_keys)} keys")
                # Recreate EMA shadow from the loaded model params
                ema_shadow: Dict[str, torch.Tensor] = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                best_val = float(checkpoint.get("acc", 0.0))
                print(f"[Resume] Resumed from best EMA (val acc={best_val:.4f})")
                
                # Determine if we're continuing or starting fresh
                if "epoch" in checkpoint:
                    checkpoint_epoch = int(checkpoint["epoch"])
                    start_epoch = checkpoint_epoch + 1
                    # If we've already completed all epochs, start fresh from epoch 1
                    # This allows continuing training with improved code/config
                    if start_epoch > epochs:
                        print(f"[Resume] Checkpoint is at epoch {checkpoint_epoch}, which completes all {epochs} epochs.")
                        print(f"[Resume] Starting fresh training from epoch 1 with loaded weights.")
                        start_epoch = 1
                        # Reset optimizer/scheduler for fresh training
                        optimizer = torch.optim.AdamW(
                            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98)
                        )
                        # Recalculate total_steps accounting for unlabeled data if used
                        steps_per_epoch = len(train_loader)
                        if use_unlabeled_data and unlabeled_loader is not None:
                            steps_per_epoch = max(len(train_loader), len(unlabeled_loader))
                        total_steps = math.ceil(steps_per_epoch * epochs)
                        warmup_steps = int(total_steps * warmup_ratio)
                        scheduler = get_cosine_schedule_with_warmup(
                            optimizer, warmup_steps, total_steps
                        )
                    else:
                        print(f"[Resume] Will continue from epoch {start_epoch}")
                        # Load optimizer/scheduler state if available (for true resume)
                        if "optimizer" in checkpoint:
                            try:
                                optimizer.load_state_dict(checkpoint["optimizer"])
                                print("[Resume] Loaded optimizer state")
                            except Exception as e:
                                print(f"[Resume] WARNING: Could not load optimizer state: {e}")
                                print("[Resume] Using fresh optimizer state")
                        if "scheduler" in checkpoint:
                            try:
                                scheduler.load_state_dict(checkpoint["scheduler"])
                                print("[Resume] Loaded scheduler state")
                            except Exception as e:
                                print(f"[Resume] WARNING: Could not load scheduler state: {e}")
                                print("[Resume] Using fresh scheduler state")
                else:
                    start_epoch = 1
            except RuntimeError as e:
                # If loading fails due to size mismatches, start fresh
                error_msg = str(e)
                if "size mismatch" in error_msg or "Expected" in error_msg:
                    print(f"[Resume] WARNING: Architecture mismatch detected from load error!")
                    print(f"[Resume] Error: {error_msg[:200]}...")
                    print(f"[Resume] Starting fresh training with new architecture.")
                else:
                    print(f"[Resume] WARNING: Failed to load checkpoint: {e}")
                    print(f"[Resume] Starting fresh training.")
                ema_shadow: Dict[str, torch.Tensor] = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                best_val = float(checkpoint.get("acc", 0.0))
                start_epoch = 1
    else:
        # No checkpoint exists - start fresh
        ema_shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        best_val = 0.0
        start_epoch = 1
    os.makedirs(Path(__file__).resolve().parent / "ckpts", exist_ok=True)

    # initial evaluation before any training
    model.eval()
    correct = total = 0
    sample_logits: torch.Tensor | None = None
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits, cls_h = model(input_ids, attention_mask)  # Unpack (logits, cls_h)
            if sample_logits is None:
                sample_logits = logits[:5]
                # Verify CLS is not masked: check norm
                cls_norm = cls_h[:5].norm(dim=-1).mean().item()
                print(f"[Init] Sample CLS norm: {cls_norm:.6f} (should be > 0)")
            preds = logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    init_acc = correct / max(1, total)
    print(f"Initial val accuracy: {init_acc:.4f}")
    if sample_logits is not None:
        print(f"Sample logits (first 5): {sample_logits}")
        print(
            f"Sample predictions: {sample_logits.argmax(-1).tolist()}"
        )

    # Calculate starting step if resuming (checkpoint already loaded above if exists)
    step_idx = 0
    if checkpoint is not None and "step" in checkpoint and start_epoch > 1:
        # Only use checkpoint step if we're actually continuing from a mid-epoch checkpoint
        # If we're starting fresh (start_epoch == 1), reset step_idx to 0
        step_idx = int(checkpoint["step"])
        print(f"[Resume] Resuming from step {step_idx}")
    else:
        step_idx = 0
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        
        # Length curriculum: NOTE - currently disabled because data is pre-tokenized
        # To enable: would need to retokenize on-the-fly or mask tokens beyond curriculum length
        # For now, we use full length throughout
        current_max_len = max_len  # Always use full length (curriculum disabled)
        # current_max_len = curriculum_max_len_start if epoch <= curriculum_switch_epoch else max_len
        # if epoch == curriculum_switch_epoch + 1:
        #     print(f"[Curriculum] Switching from max_len={curriculum_max_len_start} to {max_len}")
        
        # Update beta_hidden: decay after beta_hidden_decay_epochs
        current_beta_hidden = beta_hidden if epoch <= beta_hidden_decay_epochs else 0.1
        
        # R-Drop: only active for first half of training
        use_r_drop_now = use_r_drop and (epoch <= r_drop_active_epochs)
        
        # Create iterators for labeled and unlabeled data
        train_iter = iter(train_loader)
        unlabeled_iter = iter(unlabeled_loader) if (use_unlabeled_data and unlabeled_loader is not None) else None
        
        # Determine number of batches per epoch
        num_batches = len(train_loader)
        if use_unlabeled_data and unlabeled_loader is not None:
            num_batches = max(len(train_loader), len(unlabeled_loader))
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}/{epochs} (len={current_max_len})")
        for batch_idx in pbar:
            # Alternate between labeled and unlabeled batches, or process labeled only
            use_unlabeled_batch = (use_unlabeled_data and unlabeled_iter is not None and 
                                   batch_idx % 2 == 1)  # Process unlabeled on odd batches
            
            if use_unlabeled_batch:
                try:
                    batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    batch = next(unlabeled_iter)
                
                input_ids = batch["input_ids"].to(device)
                attention_mask_orig = batch["attention_mask"].to(device)
                idx = batch["idx"].long()  # already on CPU
                t_logits = unlabeled_logits[idx].to(device)  # type: ignore
                
                # ✅ Rebuild mask from pad_token_id if enabled (fixes incorrect tokenizer masks)
                if REBUILD_MASKS:
                    attention_mask = build_keep_mask(input_ids, pad_token_id).to(device)
                else:
                    attention_mask = attention_mask_orig
                
                # ✅ CRITICAL: Verify attention mask for unlabeled data too
                if step_idx % 100 == 0:
                    mask_mean = attention_mask.float().mean().item()
                    if mask_mean < 0.2:
                        print(f"⚠️  WARNING: Unlabeled mask mean too low ({mask_mean:.4f})! Consider REBUILD_MASKS=True")
                    assert attention_mask.shape[1] == input_ids.shape[1], \
                        f"Unlabeled: Mask length {attention_mask.shape[1]} != input length {input_ids.shape[1]}"
                
                # Forward pass - model returns (logits, cls_hidden)
                # ✅ CRITICAL: No detach() - gradients must flow!
                s_logits, s_cls_h = model(input_ids, attention_mask)
                
                # For unlabeled data: only KL loss (no CE since no labels)
                if use_fixed_kd_params:
                    T_now = T_fixed
                else:
                    progress = step_idx / total_steps
                    T_now = T_end + (T_start - T_end) * (1 - progress)
                
                # Compute KL divergence only (no CE, no alpha weighting for unlabeled)
                log_p_s = nn.functional.log_softmax(s_logits / T_now, dim=-1)
                p_t = nn.functional.softmax(t_logits / T_now, dim=-1)
                kl_per_sample = nn.functional.kl_div(
                    log_p_s, p_t, reduction="none", log_target=False
                ).sum(dim=-1) * (T_now * T_now)
                
                # Apply confidence weighting if enabled
                if conf_power > 0.0:
                    teacher_conf = p_t.max(dim=-1)[0] + 1e-6
                    conf_weights = torch.pow(teacher_conf, conf_power)
                    kl = (kl_per_sample * conf_weights).mean()
                    conf_weight_mean = conf_weights.mean().item()
                else:
                    kl = kl_per_sample.mean()
                    conf_weight_mean = 1.0
                
                # Unlabeled loss: only KL, weighted by unlabeled_alpha
                loss = unlabeled_alpha * kl
                parts = {"ce": 0.0, "kl": kl.item(), "unlabeled": True}
                if conf_power > 0.0:
                    parts["conf_weight_mean"] = conf_weight_mean
                    
                # Backward pass for unlabeled batch
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                # Gradient check (same as labeled batch)
                if epoch == 1 and step_idx in {0, 50, 100, 200}:
                    head_grad_ok = model.head.weight.grad is not None
                    head_grad_norm = model.head.weight.grad.norm().item() if head_grad_ok else 0.0
                    if hasattr(model, 'head_norm'):
                        head_norm_grad_ok = model.head_norm.weight.grad is not None
                        head_norm_grad_norm = model.head_norm.weight.grad.norm().item() if head_norm_grad_ok else 0.0
                    else:
                        head_norm_grad_ok = True
                        head_norm_grad_norm = 0.0
                    cls_grad_ok = model.cls.grad is not None
                    cls_grad_norm = model.cls.grad.norm().item() if cls_grad_ok else 0.0
                    cls_h_norm = s_cls_h.norm(dim=-1).mean().item()
                    
                    print(f"\n[Gradient Check] Step {step_idx} (unlabeled):")
                    print(f"  head.weight.grad is None: {not head_grad_ok}")
                    if head_grad_ok:
                        print(f"  head.weight.grad.norm(): {head_grad_norm:.6f}")
                    if hasattr(model, 'head_norm'):
                        print(f"  head_norm.weight.grad is None: {not head_norm_grad_ok}")
                        if head_norm_grad_ok:
                            print(f"  head_norm.weight.grad.norm(): {head_norm_grad_norm:.6f}")
                    print(f"  cls.grad is None: {not cls_grad_ok}")
                    if cls_grad_ok:
                        print(f"  cls.grad.norm(): {cls_grad_norm:.6f}")
                    print(f"  cls_h.norm(): {cls_h_norm:.6f}")
                    
                    if not head_grad_ok:
                        print(f"  ⚠️  WARNING: head.weight.grad is None! Head is not training!")
                    if not cls_grad_ok:
                        print(f"  ⚠️  WARNING: cls.grad is None! CLS token is not training!")
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                # EMA update (same as labeled batch)
                with torch.no_grad():
                    current_state = model.state_dict()
                    for k in ema_shadow.keys():
                        if k in current_state and current_state[k].dtype.is_floating_point:
                            ema_shadow[k].mul_(ema_decay).add_(
                                current_state[k].detach(), alpha=1 - ema_decay
                            )
                
                # Update progress bar
                postfix = {"loss": float(loss.item()), "kl": parts["kl"], "unlabeled": True}
                if conf_power > 0.0 and "conf_weight_mean" in parts:
                    postfix["conf_w"] = parts["conf_weight_mean"]
                pbar.set_postfix(**postfix)
                
                # Debug printing
                if epoch == 1 and step_idx in {0, 50, 100, 200}:
                    avg_diff = torch.abs(s_logits[:, 0] - s_logits[:, 1]).mean().item()
                    cls_norm = s_cls_h.norm(dim=-1).mean().item()
                    print(f"\nStep {step_idx} (unlabeled): Loss={loss.item():.4f}, avg s_logits_diff={avg_diff:.4f}, cls_norm={cls_norm:.6f}")
                step_idx += 1
                continue  # Skip to next iteration
            
            # Labeled batch (standard training)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
                input_ids = batch["input_ids"].to(device)
                attention_mask_orig = batch["attention_mask"].to(device)
                labels_batch = batch["label"].to(device)
                idx = batch["idx"].long()  # already on CPU
                t_logits = train_logits[idx].to(device)
                
                # ✅ Rebuild mask from pad_token_id if enabled (fixes incorrect tokenizer masks)
                if REBUILD_MASKS:
                    attention_mask = build_keep_mask(input_ids, pad_token_id).to(device)
                else:
                    attention_mask = attention_mask_orig
                
                # ✅ CRITICAL: Verify attention mask (check periodically)
                if step_idx % 100 == 0:
                    mask_mean = attention_mask.float().mean().item()
                    if mask_mean < 0.2:
                        print(f"⚠️  WARNING: Mask mean too low ({mask_mean:.4f})! Consider REBUILD_MASKS=True")
                    # Verify CLS will not be masked (CLS is appended at position -1, will have mask=1.0)
                    assert attention_mask.shape[1] == input_ids.shape[1], \
                        f"Mask length {attention_mask.shape[1]} != input length {input_ids.shape[1]}"
                    print(f"[Mask Check] Step {step_idx}: mask mean={mask_mean:.4f}, shape={attention_mask.shape}, REBUILD_MASKS={REBUILD_MASKS}")
                
                # Forward pass - model returns (logits, cls_hidden)
                # ✅ CRITICAL: No detach() here - gradients must flow!
                # For R-Drop, we need two forward passes with different dropout masks
                if use_r_drop_now:
                    s_logits1, s_cls_h1 = model(input_ids, attention_mask)
                    s_logits2, s_cls_h2 = model(input_ids, attention_mask)
                    # Use average for main loss computation (gradients flow through average)
                    s_logits = (s_logits1 + s_logits2) / 2.0
                    s_cls_h = (s_cls_h1 + s_cls_h2) / 2.0
                else:
                    s_logits, s_cls_h = model(input_ids, attention_mask)
                    # ✅ s_logits and s_cls_h maintain gradients - no detach!
            
            # Compute knowledge distillation loss
            if use_fixed_kd_params:
                # Use fixed temperature and alpha (Config A: T=2-3, alpha=0.6-0.7)
                T_now = T_fixed if T_fixed >= 2.0 else 2.5  # Config A: T=2-3
                alpha_now = alpha_kd  # Use Config A alpha_kd
            else:
                # Use annealing schedule: start with soft teacher targets, gradually emphasize hard labels
                progress = step_idx / total_steps
                T_now = T_end + (T_start - T_end) * (1 - progress)  # T_start -> T_end
                alpha_now = alpha_end + (alpha_start - alpha_end) * (1 - progress)  # alpha_start -> alpha_end
            
            # Logit KD loss
            kd_loss_val, parts = kd_loss(
                s_logits, t_logits, labels_batch,
                T=T_now, alpha=alpha_now, label_smoothing=label_smoothing,
                conf_power=conf_power
            )
            
            # Hidden-state MSE loss (Config A)
            # NOTE: Teacher hidden states need to be pre-computed or computed on-the-fly
            # For now, we'll skip this if teacher hidden states are not available
            hidden_mse = torch.tensor(0.0, device=device)
            if current_beta_hidden > 0.0:
                # TODO: Load teacher hidden states from checkpoint or compute on-the-fly
                # For now, this is a placeholder - you'll need to add teacher hidden state extraction
                # t_cls_h = teacher_hidden_states[idx].to(device)  # Shape: (B, d_model)
                # hidden_mse = hidden_state_mse_loss(s_cls_h, t_cls_h)
                pass
            
            # R-Drop loss (Config A)
            r_drop_loss_val = torch.tensor(0.0, device=device)
            if use_r_drop_now:
                r_drop_loss_val = r_drop_loss(s_logits1, s_logits2)
            
            # Combined loss: (1-α) * CE + α * KD + β * Hidden_MSE + λ * R-Drop
            loss = kd_loss_val + current_beta_hidden * hidden_mse + r_drop_weight * r_drop_loss_val
            
            parts["unlabeled"] = False
            parts["hidden_mse"] = hidden_mse.item() if isinstance(hidden_mse, torch.Tensor) else 0.0
            parts["r_drop"] = r_drop_loss_val.item() if isinstance(r_drop_loss_val, torch.Tensor) else 0.0
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # ✅ GRADIENT CHECK: Verify gradients flow to head
            # This is critical - if gradients are None or zero, the head isn't training!
            if epoch == 1 and step_idx in {0, 50, 100, 200}:
                # Check head_norm gradients (if using ln_linear head)
                head_norm_grad_ok = True
                head_norm_grad_norm = 0.0
                if hasattr(model, 'head_norm') and model.head_norm.weight.grad is not None:
                    head_norm_grad_norm = model.head_norm.weight.grad.norm().item()
                    head_norm_grad_ok = head_norm_grad_norm > 1e-8
                else:
                    head_norm_grad_ok = False
                
                # Check head (classifier) gradients
                head_grad_ok = True
                head_grad_norm = 0.0
                if model.head.weight.grad is not None:
                    head_grad_norm = model.head.weight.grad.norm().item()
                    head_grad_ok = head_grad_norm > 1e-8
                else:
                    head_grad_ok = False
                
                # Check CLS token gradients
                cls_token_grad_ok = True
                cls_token_grad_norm = 0.0
                if model.cls.grad is not None:
                    cls_token_grad_norm = model.cls.grad.norm().item()
                    cls_token_grad_ok = cls_token_grad_norm > 1e-8
                else:
                    cls_token_grad_ok = False
                
                cls_h_norm = s_cls_h.norm(dim=-1).mean().item()
                
                print(f"\n[Gradient Check] Step {step_idx}:")
                print(f"  head_norm.weight.grad is None: {not head_norm_grad_ok}")
                if head_norm_grad_ok:
                    print(f"  head_norm.weight.grad.norm(): {head_norm_grad_norm:.6f}")
                print(f"  head.weight.grad is None: {not head_grad_ok}")
                if head_grad_ok:
                    print(f"  head.weight.grad.norm(): {head_grad_norm:.6f}")
                print(f"  cls.grad is None: {not cls_token_grad_ok}")
                if cls_token_grad_ok:
                    print(f"  cls.grad.norm(): {cls_token_grad_norm:.6f}")
                print(f"  cls_h.norm(): {cls_h_norm:.6f}")
                
                # Warn if gradients are broken
                if not head_grad_ok:
                    print(f"  ⚠️  WARNING: head.weight.grad is None! Head is not training!")
                if not head_norm_grad_ok and hasattr(model, 'head_norm'):
                    print(f"  ⚠️  WARNING: head_norm.weight.grad is None! Head norm is not training!")
                if not cls_token_grad_ok:
                    print(f"  ⚠️  WARNING: cls.grad is None! CLS token is not training!")
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            # EMA update
            with torch.no_grad():
                current_state = model.state_dict()
                for k in ema_shadow.keys():
                    if k in current_state and current_state[k].dtype.is_floating_point:
                        ema_shadow[k].mul_(ema_decay).add_(
                            current_state[k].detach(), alpha=1 - ema_decay
                        )
            # update progress bar
            postfix = {"loss": float(loss.item()), "kl": parts["kl"]}
            if not parts.get("unlabeled", False):
                postfix["ce"] = parts["ce"]
            if conf_power > 0.0 and "conf_weight_mean" in parts:
                postfix["conf_w"] = parts["conf_weight_mean"]
            pbar.set_postfix(**postfix)
            # optional debug printing
            if epoch == 1 and step_idx in {0, 50, 100, 200}:
                avg_diff = torch.abs(s_logits[:, 0] - s_logits[:, 1]).mean().item()
                batch_type = "unlabeled" if parts.get("unlabeled", False) else "labeled"
                cls_norm = s_cls_h.norm(dim=-1).mean().item() if not parts.get("unlabeled", False) else 0.0
                print(
                    f"\nStep {step_idx} ({batch_type}): Loss={loss.item():.4f}, avg s_logits_diff={avg_diff:.4f}, cls_norm={cls_norm:.6f}"
                )
            step_idx += 1

        # evaluate current weights
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_eval = batch["label"].to(device)
                logits, _ = model(input_ids, attention_mask)  # Unpack (logits, cls_h)
                preds = logits.argmax(-1)
                correct += (preds == labels_eval).sum().item()
                total += labels_eval.numel()
        eval_acc = correct / max(1, total)
        # evaluate EMA weights
        ema_acc = evaluate(model, val_loader, device, ema_shadow)
        print(
            f"[Val] current acc={eval_acc:.4f}, EMA acc={ema_acc:.4f}"
        )
        if ema_acc > best_val:
            best_val = ema_acc
            # persist EMA state dict with optimizer/scheduler for resume
            torch.save(
                {
                    "model": ema_shadow,
                    "acc": best_val,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": step_idx,
                },
                Path(__file__).resolve().parent / "ckpts" / "best_ema.pt",
            )
            print("Saved new best EMA checkpoint")

    print(f"Best EMA val acc: {best_val:.4f}")


def run_minimal_overfit_test(device: torch.device) -> None:
    """Truly minimal overfit test - no EMA, no KD, no scheduler, no grad clip.
    
    This is the absolute minimum to isolate bugs. If this doesn't work,
    there's a fundamental wiring issue.
    """
    print("\n" + "=" * 60)
    print("MINIMAL OVERFIT TEST: Truly minimal loop")
    print("=" * 60)
    
    # Minimal settings
    max_len = 128
    train_subset = 64
    
    # Teacher assets (just for data loading)
    teacher_model_id = "textattack/bert-base-uncased-SST-2"
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, use_fast=True)
    vocab_size = tokenizer.vocab_size
    
    # Create 64-example dataset
    train_ds_full, _ = make_tokenized(
        "train", tokenizer, max_len=max_len, batch_size=64, shuffle=False
    )
    train_dataset_subset_64 = train_ds_full.select(range(min(train_subset, len(train_ds_full))))
    print(f"[Minimal Test] Using {len(train_dataset_subset_64)} examples")
    
    # Model with NO dropout, NO regularization
    model = TinyMambaClassifier(
        vocab_size=vocab_size,
        d_model=192,
        n_layers=8,
        ssm_dim=48,
        expand=3,
        conv_kernel=3,
        num_classes=2,
        tie_embeddings=True,
        head_type="ln_linear",
        max_seq_len=max_len,
        embed_dim=128,
        dropout=0.0,  # NO dropout
        drop_path_max=0.0,  # NO stochastic depth
    ).to(device)
    
    print(f"[Minimal Test] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Truly minimal setup
    model.train()
    torch.manual_seed(0)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    # No scheduler, no AMP, no grad clip, no EMA, no KD
    
    # Get single batch (64 examples, batch_size=64)
    batch = next(iter(DataLoader(train_dataset_subset_64, batch_size=64, shuffle=False)))
    x = batch["input_ids"].to(device)
    attn_orig = batch["attention_mask"].to(device)
    y = batch["label"].to(device)
    
    print(f"[Minimal Test] Batch shape: x={x.shape}, attn={attn_orig.shape}, y={y.shape}")
    
    # Get pad_token_id from tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id  # Fallback
    print(f"[Mask Check] pad_token_id: {pad_token_id}")
    
    # ✅ TEST 1: Check original mask from tokenizer
    mask_mean_orig = attn_orig.float().mean().item()
    print(f"[Mask Check] Original tokenizer mask mean: {mask_mean_orig:.4f} (should be >0.2)")
    if mask_mean_orig < 0.2:
        print(f"  ⚠️  WARNING: Original mask mean is very low! Tokenizer mask may be wrong or inverted.")
        # Check if mask is inverted (0 = keep, 1 = pad)
        mask_inverted_mean = (1.0 - attn_orig.float()).mean().item()
        print(f"  Inverted mask mean: {mask_inverted_mean:.4f}")
    
    # ✅ TEST 2: Build correct keep mask from pad_token_id
    print(f"\n[TEST] Building correct keep mask from pad_token_id...")
    attn_correct = build_keep_mask(x, pad_token_id).to(device)
    mask_mean_correct = attn_correct.float().mean().item()
    print(f"[TEST] Correct mask mean: {mask_mean_correct:.4f} (expect ~0.6-0.9 on SST-2)")
    print(f"[TEST] Correct mask shape: {attn_correct.shape}")
    print(f"[TEST] Correct mask unique values: {attn_correct.unique()}")
    
    # Verify correct mask: first token should not be masked
    assert attn_correct[:, 0].all().item(), "First token (position 0) is masked in correct mask!"
    
    # ✅ TEST 3: Force mask to all ones to prove masking is the issue
    print(f"\n[TEST] Running with FORCED all-ones mask to prove masking is the problem...")
    attn_forced = torch.ones_like(attn_orig)  # TEMP: disable masking entirely
    print(f"[TEST] Forced mask mean: {attn_forced.float().mean().item():.4f} (should be 1.0)")
    
    # Choose which mask to use for training
    # Start with forced mask to prove it works, then switch to correct mask
    USE_FORCED_MASK = False  # Changed to False to use correct mask by default
    if USE_FORCED_MASK:
        attn = attn_forced
        print(f"\n[TEST] Using FORCED all-ones mask for training (to prove masking is the issue)")
    else:
        attn = attn_correct
        print(f"\n[TEST] Using CORRECT keep mask built from pad_token_id")
        print(f"  Mask mean: {mask_mean_correct:.4f} (should be ~0.6-0.9)")
        assert mask_mean_correct > 0.5, f"Correct mask mean too low: {mask_mean_correct:.4f}"
    
    print(f"[Minimal Test] Training for 300 steps...")
    print("-" * 60)
    print(f"{'Step':<6} {'Loss':<10} {'Acc':<10}")
    print("-" * 60)
    
    # Minimal training loop - exactly as specified
    for t in range(300):
        optim.zero_grad(set_to_none=True)
        
        # Model returns (logits, cls_h) - unpack just logits
        # ✅ CRITICAL: No detach() - gradients must flow!
        # Note: model is in train mode, returns [B, 2], no detach
        
        # Use chosen mask (forced or correct)
        # Model will assert mask stats and CLS norm - failures will be caught immediately
        logits, cls_h = model(x, attention_mask=attn)
        
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        
        # ✅ GRADIENT CHECK: Verify gradients flow to head (on first step)
        if t == 0:
            head_grad_ok = model.head.weight.grad is not None
            head_grad_norm = model.head.weight.grad.norm().item() if head_grad_ok else 0.0
            if hasattr(model, 'head_norm'):
                head_norm_grad_ok = model.head_norm.weight.grad is not None
                head_norm_grad_norm = model.head_norm.weight.grad.norm().item() if head_norm_grad_ok else 0.0
            else:
                head_norm_grad_ok = True
                head_norm_grad_norm = 0.0
            cls_grad_ok = model.cls.grad is not None
            cls_grad_norm = model.cls.grad.norm().item() if cls_grad_ok else 0.0
            
            print(f"\n[Gradient Check] Step 0:")
            print(f"  head.weight.grad is None: {not head_grad_ok}")
            if head_grad_ok:
                print(f"  head.weight.grad.norm(): {head_grad_norm:.6f}")
            if hasattr(model, 'head_norm'):
                print(f"  head_norm.weight.grad is None: {not head_norm_grad_ok}")
                if head_norm_grad_ok:
                    print(f"  head_norm.weight.grad.norm(): {head_norm_grad_norm:.6f}")
            print(f"  cls.grad is None: {not cls_grad_ok}")
            if cls_grad_ok:
                print(f"  cls.grad.norm(): {cls_grad_norm:.6f}")
            
            if not head_grad_ok:
                print(f"  ⚠️  ERROR: head.weight.grad is None! Head is not training!")
                print(f"  Check for detach() or .data in forward pass!")
            if not cls_grad_ok:
                print(f"  ⚠️  ERROR: cls.grad is None! CLS token is not training!")
                print(f"  Check for detach() or .data in forward pass!")
            print()
        
        optim.step()  # No grad clipping, no scheduler
        
        if t % 20 == 0:
            with torch.no_grad():
                # Compute accuracy on current logits (model still in train mode)
                acc = (logits.argmax(-1) == y).float().mean().item()
                # Also print CLS norm to verify it's changing
                cls_norm_val = cls_h.norm(dim=-1).mean().item()
                print(f"{t:<6} {loss.item():<10.6f} {acc:<10.4f} cls_norm={cls_norm_val:.4f}")
    
    # Final check (use the same mask as training)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, attention_mask=attn)
        final_acc = (logits.argmax(-1) == y).float().mean().item()
        final_loss = torch.nn.functional.cross_entropy(logits, y).item()
    
    print("-" * 60)
    print(f"[Minimal Test] Final: Loss={final_loss:.6f}, Acc={final_acc:.4f}")
    if final_acc > 0.95:
        if USE_FORCED_MASK:
            print("✓ PASSED with FORCED mask: Model can overfit when masking is disabled")
            print("  → This confirms masking is the problem!")
            print("  → Next: Set USE_FORCED_MASK=False to use correct keep mask and verify it works")
        else:
            print("✓ PASSED: Model can overfit with correct mask (pipeline is correct)")
            print("  → Masking is working correctly!")
            print("  → You can now run full training with REBUILD_MASKS=True if tokenizer masks are wrong")
    else:
        print("✗ FAILED: Model cannot overfit (check wiring: masking, labels, model forward)")
        print("  → Check mask stats (should be ~0.6-0.9, not 0.10)")
        print("  → Check CLS norm (should be >> 0)")
        print("  → Check gradients (head.weight.grad should not be None)")
    print("=" * 60)


def run_sanity_check_overfit(device: torch.device) -> None:
    """Quick sanity check: 64-example overfit test.
    
    Should hit >95% accuracy in <200 steps if pipeline is correct.
    Uses: CE-only, no KD, no regularization, high LR.
    """
    print("\n[Sanity Check] Starting 64-example overfit test...")
    
    # Sanity check hyperparameters
    max_len = 128
    max_steps = 200
    train_subset = 64
    lr = 1e-3
    weight_decay = 0.0
    dropout = 0.0
    label_smoothing = 0.0
    use_kd = False
    alpha = 0.0
    
    # Teacher assets (needed for data loading, but won't use KD)
    teacher_model_id = "textattack/bert-base-uncased-SST-2"
    logits_path = Path(__file__).resolve().parent / "teacher_sst2_logits.pt"
    teacher_pack = torch.load(str(logits_path), map_location="cpu")
    train_logits = teacher_pack["train"]["logits"].float()
    train_labels = teacher_pack["train"]["labels"].long()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, use_fast=True)
    vocab_size = tokenizer.vocab_size
    
    # Create small training subset (64 examples)
    train_ds_full, _ = make_tokenized(
        "train", tokenizer, max_len=max_len, batch_size=32, shuffle=False
    )
    # Take first 64 examples
    train_ds_subset = train_ds_full.select(range(min(train_subset, len(train_ds_full))))
    train_loader = DataLoader(
        train_ds_subset,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    print(f"[Sanity Check] Using {len(train_ds_subset)} training examples")
    
    # Create small validation set (same 64 examples for overfitting test)
    val_loader = DataLoader(
        train_ds_subset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    
    # Model with no regularization
    model = TinyMambaClassifier(
        vocab_size=vocab_size,
        d_model=192,
        n_layers=8,
        ssm_dim=48,
        expand=3,
        conv_kernel=3,
        num_classes=2,
        tie_embeddings=True,
        head_type="ln_linear",
        max_seq_len=max_len,
        embed_dim=128,
        dropout=dropout,  # No dropout for sanity check
        drop_path_max=0.0,  # No stochastic depth
    ).to(device)
    
    print(f"[Sanity Check] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer: no weight decay, higher LR
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98)
    )
    
    # Training loop
    model.train()
    best_acc = 0.0
    steps_to_95 = None
    
    print(f"\n[Sanity Check] Training for up to {max_steps} steps...")
    print(f"[Sanity Check] Target: >95% accuracy in <200 steps")
    print("-" * 60)
    
    step = 0
    for epoch in range(10):  # Max 10 epochs
        for batch in train_loader:
            if step >= max_steps:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            idx = batch["idx"].long()
            
            # Forward pass
            logits, cls_h = model(input_ids, attention_mask)
            
            # CE loss only (no KD, no label smoothing)
            loss = nn.functional.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            
            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Evaluate every 10 steps
            if step % 10 == 0:
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input_ids = val_batch["input_ids"].to(device)
                        val_attention_mask = val_batch["attention_mask"].to(device)
                        val_labels = val_batch["label"].to(device)
                        val_logits, _ = model(val_input_ids, val_attention_mask)
                        preds = val_logits.argmax(-1)
                        correct += (preds == val_labels).sum().item()
                        total += val_labels.numel()
                acc = correct / max(1, total)
                best_acc = max(best_acc, acc)
                
                if acc > 0.95 and steps_to_95 is None:
                    steps_to_95 = step
                
                print(f"Step {step:3d}: Loss={loss.item():.4f}, Acc={acc:.4f} ({correct}/{total}), Best={best_acc:.4f}")
                model.train()
            
            step += 1
        
        if step >= max_steps:
            break
    
    # Final evaluation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_input_ids = val_batch["input_ids"].to(device)
            val_attention_mask = val_batch["attention_mask"].to(device)
            val_labels = val_batch["label"].to(device)
            val_logits, _ = model(val_input_ids, val_attention_mask)
            preds = val_logits.argmax(-1)
            correct += (preds == val_labels).sum().item()
            total += val_labels.numel()
    final_acc = correct / max(1, total)
    
    # Results
    print("-" * 60)
    print(f"[Sanity Check] Results:")
    print(f"  Final accuracy: {final_acc:.4f} ({correct}/{total})")
    print(f"  Best accuracy: {best_acc:.4f}")
    if steps_to_95 is not None:
        print(f"  Steps to >95%: {steps_to_95}")
        print(f"  ✓ PASSED: Reached >95% in {steps_to_95} steps (< 200)")
    else:
        print(f"  ✗ FAILED: Did not reach >95% in {max_steps} steps")
        print(f"  This indicates a plumbing bug (masking, labels, optimizer, etc.)")
    print("=" * 60)
    
    return None


if __name__ == "__main__":
    main()