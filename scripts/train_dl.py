#!/usr/bin/env python
"""Five-fold PyTorch training with GBDT-compatible prediction artifacts.

Examples:
    python scripts/train_dl.py --model mlp --config configs/mlp.yaml --cv skf
    python scripts/train_dl.py --model set_encoder --config configs/gene_set_encoder.yaml --cv skf
    python scripts/train_dl.py --model hybrid --config configs/hybrid_set_mlp.yaml --cv skf
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cancer_hack.dataset_dl import (  # noqa: E402
    MutationSampleDataset,
    MutationTokenizer,
    collate_mutation_samples,
    load_dense_feature_bundle,
    tokenize_frame,
)
from cancer_hack.features_basic import BurdenBinner  # noqa: E402
from cancer_hack.features_frequency import (  # noqa: E402
    AA_TRANSITION_FEATURE_COLUMNS,
    FREQUENCY_RARITY_FEATURE_COLUMNS,
    TrainAATransitionFeatures,
    TrainFrequencyFeatures,
)
from cancer_hack.features_latent import build_fold_latent_block  # noqa: E402
from cancer_hack.io import save_csv, write_submission  # noqa: E402
from cancer_hack.metrics import (  # noqa: E402
    build_prediction_frame,
    evaluate_classification,
    macro_f1,
    validate_prediction_frame_schema,
)
from cancer_hack.models_dl import create_dl_model  # noqa: E402
from cancer_hack.validation import CV_SLUG, fold_column  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data/raw"
PROC_DIR = PROJECT_ROOT / "data/process"
ARTIFACTS = PROJECT_ROOT / "artifacts"
FREQUENCY_BLOCKS = ("freq21", "aatrans9")
BURDEN_COLUMNS = ("hypermutated_flag", "burden_quantile_bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", required=True, choices=["mlp", "set_encoder", "hybrid"]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cv", choices=["skf", "sgkf"], default="skf")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="one epoch, one fold")
    parser.add_argument("--no-submission", action="store_true")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def append_fold_burden(
    train_dense: np.ndarray,
    test_dense: np.ndarray,
    rollup_train: pd.DataFrame,
    rollup_test: pd.DataFrame,
    train_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the two stateful rollup16 columns on fold-train only."""
    binner = BurdenBinner().fit(rollup_train.iloc[train_index])
    columns = list(BURDEN_COLUMNS)
    train_extra = binner.transform(rollup_train)[columns].to_numpy(np.float32)
    test_extra = binner.transform(rollup_test)[columns].to_numpy(np.float32)
    return (
        np.hstack([train_dense, train_extra]),
        np.hstack([test_dense, test_extra]),
    )


def build_fold_frequency_blocks(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    train_index: np.ndarray,
    *,
    gene_columns: list[str],
    blocks: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Fit frequency/rarity lookups on fold-train and transform every row."""
    unknown = sorted(set(blocks) - set(FREQUENCY_BLOCKS))
    if unknown:
        raise ValueError(f"unknown DL frequency blocks: {unknown}")

    names: list[str] = []
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    fit_frame = raw_train.iloc[np.asarray(train_index)]
    for block in blocks:
        if block == "freq21":
            builder = TrainFrequencyFeatures(rare_df_threshold=2)
            expected = list(FREQUENCY_RARITY_FEATURE_COLUMNS)
        else:
            builder = TrainAATransitionFeatures(
                alpha=1.0,
                unique_transitions_per_sample=True,
            )
            expected = list(AA_TRANSITION_FEATURE_COLUMNS)
        builder.fit(fit_frame, gene_columns=gene_columns)
        train_frame = builder.transform(raw_train, gene_columns=gene_columns)
        test_frame = builder.transform(raw_test, gene_columns=gene_columns)
        if list(train_frame.columns) != expected or list(test_frame.columns) != expected:
            raise RuntimeError(f"{block} output schema differs from its constants")
        names.extend(f"{block}__{column}" for column in expected)
        train_parts.append(train_frame.to_numpy(np.float32))
        test_parts.append(test_frame.to_numpy(np.float32))

    if not train_parts:
        return (
            [],
            np.empty((len(raw_train), 0), dtype=np.float32),
            np.empty((len(raw_test), 0), dtype=np.float32),
        )
    return (
        names,
        np.hstack(train_parts).astype(np.float32, copy=False),
        np.hstack(test_parts).astype(np.float32, copy=False),
    )


def load_latent_source(
    process_dir: Path,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load the encoded gene matrix used to fit a latent basis inside each fold."""
    train = pd.read_parquet(process_dir / "train_mutation_encoded.parquet")
    test = pd.read_parquet(process_dir / "test_mutation_encoded.parquet")
    if not np.array_equal(train["ID"].astype(str).to_numpy(), train_ids):
        raise ValueError("latent train ID order differs from dense feature blocks")
    if not np.array_equal(test["ID"].astype(str).to_numpy(), test_ids):
        raise ValueError("latent test ID order differs from dense feature blocks")
    genes = [column for column in train.columns if column not in ("ID", "SUBCLASS")]
    test_genes = [column for column in test.columns if column not in ("ID", "SUBCLASS")]
    if genes != test_genes:
        raise ValueError("latent train/test gene columns or order differ")
    return (
        genes,
        train[genes].to_numpy(np.float32),
        test[genes].to_numpy(np.float32),
    )


def append_fold_engineered_features(
    train_dense: np.ndarray,
    test_dense: np.ndarray,
    train_index: np.ndarray,
    *,
    labels: np.ndarray,
    frequency_blocks: list[str],
    raw_train: pd.DataFrame | None,
    raw_test: pd.DataFrame | None,
    raw_gene_columns: list[str] | None,
    latent_source: tuple[list[str], np.ndarray, np.ndarray] | None,
    latent_config: dict[str, object],
    latent_metadata: list[dict[str, object]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Append every configured fold-fitted dense block without data leakage."""
    names: list[str] = []
    if frequency_blocks:
        if raw_train is None or raw_test is None or raw_gene_columns is None:
            raise RuntimeError("frequency blocks require aligned raw train/test frames")
        frequency_names, frequency_train, frequency_test = build_fold_frequency_blocks(
            raw_train,
            raw_test,
            train_index,
            gene_columns=raw_gene_columns,
            blocks=frequency_blocks,
        )
        train_dense = np.hstack([train_dense, frequency_train])
        test_dense = np.hstack([test_dense, frequency_test])
        names.extend(frequency_names)

    if latent_source is not None:
        latent_genes, latent_train, latent_test = latent_source
        methods_value = latent_config.get(
            "methods", [latent_config.get("method", "svd")]
        )
        methods = (
            [str(methods_value)]
            if isinstance(methods_value, str)
            else [str(method) for method in methods_value]
        )
        unknown_methods = sorted(set(methods) - {"svd", "nmf"})
        if unknown_methods:
            raise ValueError(f"unknown DL latent methods: {unknown_methods}")
        shared_config = {
            key: value
            for key, value in latent_config.items()
            if key not in ("method", "methods")
        }
        for method in methods:
            latent_names, fold_latent_train, fold_latent_test, _, basis = (
                build_fold_latent_block(
                    latent_train,
                    latent_test,
                    train_index,
                    labels[train_index],
                    gene_names=latent_genes,
                    method=method,
                    **shared_config,
                )
            )
            if latent_metadata is not None:
                latent_metadata.append(
                    {
                        "method": method,
                        "dimension": int(basis.n_components),
                        "fit_row_count": int(len(train_index)),
                        "support_gene_count": int(len(basis.gene_index)),
                    }
                )
            train_dense = np.hstack([train_dense, fold_latent_train])
            test_dense = np.hstack([test_dense, fold_latent_test])
            names.extend(latent_names)
    return train_dense, test_dense, names


def scale_fold_dense(
    train_dense: np.ndarray,
    test_dense: np.ndarray,
    train_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize using fold-train statistics and sanitize constant columns."""
    scaler = StandardScaler()
    scaler.fit(train_dense[train_index])
    train_scaled = scaler.transform(train_dense).astype(np.float32)
    test_scaled = scaler.transform(test_dense).astype(np.float32)
    return (
        np.nan_to_num(train_scaled, copy=False),
        np.nan_to_num(test_scaled, copy=False),
    )


def move_batch(
    batch: dict[str, object], device: torch.device
) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def resolve_loss_config(training: dict[str, object]) -> dict[str, object]:
    """Resolve the new loss mapping while preserving legacy class_weight configs."""
    configured = dict(training.get("loss", {}))
    default_weight = "balanced" if bool(training.get("class_weight", True)) else "none"
    resolved = {
        "name": str(configured.get("name", "cross_entropy")),
        "class_weight": str(configured.get("class_weight", default_weight)),
        "label_smoothing": float(configured.get("label_smoothing", 0.0)),
        "focal_gamma": float(configured.get("focal_gamma", 2.0)),
    }
    if resolved["name"] not in ("cross_entropy", "focal"):
        raise ValueError(f"unsupported loss name: {resolved['name']}")
    if resolved["class_weight"] not in ("balanced", "sqrt_balanced", "none"):
        raise ValueError(
            f"unsupported loss class_weight: {resolved['class_weight']}"
        )
    if not 0.0 <= resolved["label_smoothing"] < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if resolved["focal_gamma"] < 0.0:
        raise ValueError("focal_gamma must be >= 0")
    return resolved


def class_weights(
    labels: np.ndarray,
    num_classes: int,
    *,
    mode: str = "balanced",
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode not in ("balanced", "sqrt_balanced"):
        raise ValueError(f"unsupported class-weight mode: {mode}")
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = len(labels) / (num_classes * np.maximum(counts, 1.0))
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    return torch.as_tensor(weights, dtype=torch.float32)


class FocalCrossEntropy(nn.Module):
    """Multiclass focal loss with optional smoothing and one class-weight factor."""

    def __init__(
        self,
        *,
        gamma: float,
        weight: torch.Tensor | None,
        label_smoothing: float,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probability = torch.softmax(logits, dim=1)
        true_probability = probability.gather(1, targets[:, None]).squeeze(1)
        losses = (1.0 - true_probability).pow(self.gamma) * cross_entropy
        if self.weight is None:
            return losses.mean()
        sample_weight = self.weight[targets]
        return (losses * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)


def build_classification_loss(
    loss_config: dict[str, object],
    weights: torch.Tensor | None,
    device: torch.device,
) -> nn.Module:
    weight = None if weights is None else weights.to(device)
    if loss_config["name"] == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=float(loss_config["label_smoothing"]),
        )
    return FocalCrossEntropy(
        gamma=float(loss_config["focal_gamma"]),
        weight=weight,
        label_smoothing=float(loss_config["label_smoothing"]),
    ).to(device)


def make_loader(
    samples,
    ids,
    *,
    labels,
    dense,
    indices,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    dataset = MutationSampleDataset(
        samples,
        ids,
        labels=labels,
        dense=dense,
        indices=indices,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_mutation_samples,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    weights: torch.Tensor | None,
    loss_config: dict[str, object],
    early_stopping: dict[str, object],
) -> dict[str, object]:
    criterion = build_classification_loss(loss_config, weights, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    train_history: list[float] = []
    valid_loss_history: list[float] = []
    valid_f1_history: list[float] = []
    enabled = bool(early_stopping.get("enabled", False))
    monitor = str(early_stopping.get("monitor", "val_macro_f1"))
    if monitor not in ("val_macro_f1", "val_loss"):
        raise ValueError(f"unsupported early-stopping monitor: {monitor}")
    patience = int(early_stopping.get("patience", 10))
    min_epochs = int(early_stopping.get("min_epochs", 1))
    min_delta = float(early_stopping.get("min_delta", 0.0))
    restore_best = bool(early_stopping.get("restore_best_weights", True))
    if patience < 1 or min_epochs < 1 or min_delta < 0:
        raise ValueError(
            "early stopping requires patience/min_epochs >= 1 and min_delta >= 0"
        )
    best_metric = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    stopped_epoch = epochs

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch["labels"])
            loss.backward()
            if gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            rows = int(batch["labels"].shape[0])
            total_loss += float(loss.detach()) * rows
            total_rows += rows
        scheduler.step()
        train_loss = total_loss / max(total_rows, 1)
        train_history.append(train_loss)

        if not enabled:
            continue

        model.eval()
        valid_loss = 0.0
        valid_rows = 0
        valid_true: list[np.ndarray] = []
        valid_pred: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in valid_loader:
                batch = move_batch(batch, device)
                logits = model(batch)
                loss = criterion(logits, batch["labels"])
                rows = int(batch["labels"].shape[0])
                valid_loss += float(loss) * rows
                valid_rows += rows
                valid_true.append(batch["labels"].cpu().numpy())
                valid_pred.append(logits.argmax(dim=1).cpu().numpy())
        valid_loss /= max(valid_rows, 1)
        valid_f1 = macro_f1(
            np.concatenate(valid_true), np.concatenate(valid_pred)
        )
        valid_loss_history.append(valid_loss)
        valid_f1_history.append(valid_f1)
        metric = valid_f1 if monitor == "val_macro_f1" else -valid_loss
        if metric > best_metric + min_delta:
            best_metric = metric
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        print(
            f"    epoch {epoch + 1:02d}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={valid_loss:.4f} val_macro_f1={valid_f1:.4f}"
        )
        if epoch + 1 >= min_epochs and stale_epochs >= patience:
            stopped_epoch = epoch + 1
            break

    if enabled and restore_best and best_state is not None:
        model.load_state_dict(best_state)
    if not enabled:
        best_epoch = len(train_history)
        best_metric = None
        stopped_epoch = len(train_history)
    return {
        "train_loss": train_history,
        "val_loss": valid_loss_history,
        "val_macro_f1": valid_f1_history,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "stopped_epoch": stopped_epoch,
        "monitor": monitor if enabled else None,
    }


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = []
    for batch in loader:
        batch = move_batch(batch, device)
        probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.vstack(probabilities)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configured_model = config.get("model")
    if configured_model is not None and configured_model != args.model:
        raise ValueError(
            f"CLI model {args.model!r} differs from config model {configured_model!r}"
        )
    training = dict(config.get("training", {}))
    loss_config = resolve_loss_config(training)
    dense_features = dict(config.get("dense_features", {}))
    frequency_blocks = list(dense_features.get("frequency_blocks", []))
    unknown_frequency = sorted(set(frequency_blocks) - set(FREQUENCY_BLOCKS))
    if unknown_frequency:
        raise ValueError(f"unknown dense frequency blocks: {unknown_frequency}")
    latent_config = dict(dense_features.get("latent", {}))
    latent_enabled = bool(latent_config.pop("enabled", False))
    if (frequency_blocks or latent_enabled) and args.model == "set_encoder":
        raise ValueError(
            "fold dense features require model=mlp or model=hybrid; "
            "set_encoder does not consume dense inputs"
        )
    early_stopping = dict(training.get("early_stopping", {}))
    tokenizer_config = dict(config.get("tokenizer", {}))
    model_params = dict(config.get("model_params", {}))
    epochs = int(args.epochs or training.get("epochs", 30))
    batch_size = int(args.batch_size or training.get("batch_size", 64))
    num_workers = int(training.get("num_workers", 0))
    if args.dry_run:
        epochs, num_workers = 1, 0

    seed_everything(args.seed)
    device = resolve_device(args.device)
    domain_feature_set = str(dense_features.get("domain_feature_set", "default"))
    dense_bundle = load_dense_feature_bundle(
        PROC_DIR,
        domain_feature_set=domain_feature_set,
    )
    classes = np.unique(dense_bundle.labels)
    label_to_index = {label: index for index, label in enumerate(classes)}
    labels = np.asarray([label_to_index[label] for label in dense_bundle.labels])

    folds = pd.read_parquet(PROC_DIR / "train_folds.parquet")
    if not np.array_equal(
        folds["ID"].astype(str).to_numpy(), dense_bundle.train_ids
    ):
        raise ValueError("train_folds.parquet ID order differs from feature blocks")
    fold_name = fold_column(args.cv, args.n_splits)
    if fold_name not in folds.columns:
        raise ValueError(f"{fold_name} is missing from train_folds.parquet")

    tokenizer = None
    vocab_sizes = None
    train_raw = None
    test_raw = None
    raw_gene_columns: list[str] | None = None
    needs_raw = args.model != "mlp" or bool(frequency_blocks)
    if needs_raw:
        train_raw = pd.read_csv(RAW_DIR / "train.csv", dtype=str, na_filter=False)
        test_raw = pd.read_csv(RAW_DIR / "test.csv", dtype=str, na_filter=False)
        if not np.array_equal(
            train_raw["ID"].astype(str).to_numpy(), dense_bundle.train_ids
        ) or not np.array_equal(
            test_raw["ID"].astype(str).to_numpy(), dense_bundle.test_ids
        ):
            raise ValueError("raw CSV ID order differs from feature blocks")
        raw_gene_columns = [
            column for column in train_raw.columns if column not in ("ID", "SUBCLASS")
        ]

    if args.model == "mlp":
        train_samples = [[] for _ in dense_bundle.train_ids]
        test_samples = [[] for _ in dense_bundle.test_ids]
    else:
        assert train_raw is not None and test_raw is not None
        assert raw_gene_columns is not None
        tokenizer = MutationTokenizer(raw_gene_columns, **tokenizer_config)
        vocab_sizes = tokenizer.vocab_sizes
        max_tokens = config.get("max_tokens_per_gene", 32)
        print(f"tokenizing {len(train_raw):,} train / {len(test_raw):,} test samples")
        train_samples = tokenize_frame(
            train_raw, tokenizer, max_tokens_per_gene=max_tokens
        )
        test_samples = tokenize_frame(
            test_raw, tokenizer, max_tokens_per_gene=max_tokens
        )

    latent_source = (
        load_latent_source(
            PROC_DIR, dense_bundle.train_ids, dense_bundle.test_ids
        )
        if latent_enabled
        else None
    )

    oof = np.zeros((len(labels), len(classes)), dtype=np.float32)
    test_probability = np.zeros(
        (len(dense_bundle.test_ids), len(classes)), dtype=np.float32
    )
    fold_scores: list[float] = []
    fold_losses: list[list[float]] = []
    fold_training: list[dict[str, object]] = []
    fold_dense_dimensions: list[int] = []
    fold_feature_columns: list[list[str]] = []
    latent_dimensions: list[int] = []
    latent_methods: list[list[str]] = []
    latent_fit_row_counts: list[int] = []
    latent_support_gene_counts: list[list[int]] = []
    started = time.perf_counter()
    fold_values = range(1) if args.dry_run else range(args.n_splits)

    for fold in fold_values:
        seed_everything(args.seed + fold)
        valid_index = np.flatnonzero(folds[fold_name].to_numpy() == fold)
        train_index = np.flatnonzero(folds[fold_name].to_numpy() != fold)
        dense_train, dense_test = append_fold_burden(
            dense_bundle.train,
            dense_bundle.test,
            dense_bundle.rollup_train,
            dense_bundle.rollup_test,
            train_index,
        )
        feature_columns = [*dense_bundle.columns, *BURDEN_COLUMNS]
        fold_latent_metadata: list[dict[str, object]] = []
        dense_train, dense_test, engineered_names = append_fold_engineered_features(
            dense_train,
            dense_test,
            train_index,
            labels=dense_bundle.labels,
            frequency_blocks=frequency_blocks,
            raw_train=train_raw,
            raw_test=test_raw,
            raw_gene_columns=raw_gene_columns,
            latent_source=latent_source,
            latent_config=latent_config,
            latent_metadata=fold_latent_metadata,
        )
        feature_columns.extend(engineered_names)
        dense_train, dense_test = scale_fold_dense(
            dense_train, dense_test, train_index
        )
        use_dense = args.model in ("mlp", "hybrid")
        dense_dim = dense_train.shape[1]
        fold_dense_dimensions.append(int(dense_dim))
        fold_feature_columns.append(feature_columns)
        latent_dimensions.append(
            sum(int(item["dimension"]) for item in fold_latent_metadata)
        )
        latent_methods.append(
            [str(item["method"]) for item in fold_latent_metadata]
        )
        latent_fit_row_counts.append(
            int(fold_latent_metadata[0]["fit_row_count"])
            if fold_latent_metadata
            else 0
        )
        latent_support_gene_counts.append(
            [int(item["support_gene_count"]) for item in fold_latent_metadata]
        )
        print(
            f"fold {fold + 1} dense_dim={dense_dim} "
            f"latent_dim={latent_dimensions[-1]} "
            f"latent_methods={latent_methods[-1]} "
            f"latent_fit_rows={latent_fit_row_counts[-1]} "
            f"latent_support_genes={latent_support_gene_counts[-1]}"
        )
        common = dict(
            samples=train_samples,
            ids=dense_bundle.train_ids,
            labels=labels,
            dense=dense_train if use_dense else None,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        train_loader = make_loader(
            **common,
            indices=train_index,
            shuffle=True,
            seed=args.seed + fold,
        )
        valid_loader = make_loader(
            **common,
            indices=valid_index,
            shuffle=False,
            seed=args.seed + fold,
        )
        test_loader = make_loader(
            test_samples,
            dense_bundle.test_ids,
            labels=None,
            dense=dense_test if use_dense else None,
            indices=np.arange(len(dense_bundle.test_ids)),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=args.seed + fold,
        )
        model = create_dl_model(
            args.model,
            num_classes=len(classes),
            dense_dim=dense_dim,
            vocab_sizes=vocab_sizes,
            model_params=model_params,
        ).to(device)
        weights = class_weights(
            labels[train_index],
            len(classes),
            mode=str(loss_config["class_weight"]),
        )
        training_result = train_one_fold(
            model,
            train_loader,
            valid_loader,
            device=device,
            epochs=epochs,
            learning_rate=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            gradient_clip=float(training.get("gradient_clip", 1.0)),
            weights=weights,
            loss_config=loss_config,
            early_stopping=early_stopping,
        )
        valid_probability = predict(model, valid_loader, device)
        oof[valid_index] = valid_probability
        test_probability += predict(model, test_loader, device) / len(fold_values)
        score = macro_f1(
            labels[valid_index], valid_probability.argmax(axis=1)
        )
        fold_scores.append(score)
        fold_losses.append(training_result["train_loss"])
        fold_training.append(training_result)
        print(
            f"fold {fold + 1}/{len(fold_values)} "
            f"best_epoch={training_result['best_epoch']} "
            f"loss={training_result['train_loss'][-1]:.4f} Macro F1={score:.4f}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.dry_run:
        print("dry-run complete; artifacts were not written")
        return

    predicted_labels = classes[oof.argmax(axis=1)]
    summary = evaluate_classification(
        dense_bundle.labels, predicted_labels, classes
    )
    group_sizes = folds.groupby("group_key").size()
    singleton_groups = set(group_sizes[group_sizes == 1].index)
    singleton_mask = folds["group_key"].isin(singleton_groups).to_numpy()
    singleton_score = macro_f1(
        dense_bundle.labels[singleton_mask], predicted_labels[singleton_mask]
    )
    tag = args.tag or args.config.stem
    stem = f"dl_{args.model}_{tag}_{CV_SLUG[args.cv]}"
    oof_frame = build_prediction_frame(
        dense_bundle.train_ids, oof, classes, y_true=dense_bundle.labels
    )
    test_frame = build_prediction_frame(
        dense_bundle.test_ids, test_probability, classes
    )
    validate_prediction_frame_schema(
        oof_frame, dense_bundle.train_ids, classes, oof=True
    )
    validate_prediction_frame_schema(
        test_frame, dense_bundle.test_ids, classes, oof=False
    )
    save_csv(oof_frame, ARTIFACTS / "oof" / f"oof_{stem}.csv")
    save_csv(
        test_frame, ARTIFACTS / "test_predictions" / f"test_{stem}.csv"
    )
    result = {
        "stem": stem,
        "model": args.model,
        "config": str(args.config),
        "cv": args.cv,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "domain_feature_set": domain_feature_set,
        "frequency_blocks": frequency_blocks,
        "latent": {"enabled": latent_enabled, **latent_config},
        "n_features": fold_dense_dimensions[0],
        "fold_n_features": fold_dense_dimensions,
        "fold_dense_dimensions": fold_dense_dimensions,
        "latent_dimensions": latent_dimensions,
        "latent_methods": latent_methods,
        "latent_fit_row_counts": latent_fit_row_counts,
        "latent_support_gene_counts": latent_support_gene_counts,
        "feature_columns": fold_feature_columns[0],
        "fold_macro_f1": fold_scores,
        "fold_train_loss": fold_losses,
        "fold_training": fold_training,
        "early_stopping": early_stopping,
        "loss": loss_config,
        "oof_macro_f1": summary["macro_f1"],
        "oof_macro_f1_singleton": singleton_score,
        "n_singleton": int(singleton_mask.sum()),
        "oof_accuracy": summary["accuracy"],
        "per_class_f1": summary["per_class_f1"],
        "support": summary["support"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    log_path = ARTIFACTS / "logs" / f"{stem}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    if not args.no_submission:
        write_submission(
            test_frame,
            RAW_DIR / "sample_submission.csv",
            ARTIFACTS / "submissions" / f"submission_{stem}.csv",
        )
    print(
        f"OOF Macro F1={summary['macro_f1']:.4f} "
        f"singleton={singleton_score:.4f}"
    )
    print(f"wrote {log_path}")


if __name__ == "__main__":
    main()
