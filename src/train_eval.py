"""Train/evaluate baseline and synthetic-augmented fault detection models.

This module follows the project rule of not using random shuffling on time-series
signals. It uses a time-based split and compares two scenarios:
1) baseline-only anomaly model
2) normal + synthetic fault augmentation scenario
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit

from src.baseline_generator import NormalBaselineGenerator
from src.fault_injector import FaultInjector
from src.features import extract_signal_features


@dataclass
class EvaluationSummary:
    """Container for model comparison metrics."""

    precision: float
    recall: float
    f1: float
    roc_auc: float
    false_positive_rate: float
    confusion_matrix_values: tuple[int, int, int, int]


def _build_dataset(
    n_normal: int,
    n_fault: int,
    sample_rate: float,
    normal_duration: float,
    fault_duration: float,
    f_base: float,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Construct a balanced feature dataset with normal and synthetic-fault samples."""
    generator = NormalBaselineGenerator(sample_rate=sample_rate, duration=normal_duration, f_base=f_base, random_state=random_state)
    injector = FaultInjector(sample_rate=sample_rate, f_base=f_base, random_state=random_state)

    normal_records: list[dict[str, float]] = []
    fault_records: list[dict[str, float]] = []

    for _ in range(n_normal):
        signal = generator.generate_signal(length=int(sample_rate * normal_duration))
        feats = extract_signal_features(signal, sample_rate=sample_rate, base_frequency=f_base)
        normal_records.append(feats)

    for _ in range(n_fault):
        baseline = generator.generate_signal(length=int(sample_rate * fault_duration))
        evil = injector.generate_loose_bolt_signal(
            baseline_signal=baseline,
            looseness_level=0.9,
            impact_probability=0.18,
            impact_amplitude=1.6,
            frequency_shift=0.05,
        )
        feats = extract_signal_features(evil, sample_rate=sample_rate, base_frequency=f_base)
        fault_records.append(feats)

    records: list[dict[str, float]] = []
    labels: list[int] = []
    for i in range(max(n_normal, n_fault)):
        if i < n_normal:
            normal_row = normal_records[i].copy()
            normal_row["label"] = 0
            records.append(normal_row)
            labels.append(0)
        if i < n_fault:
            fault_row = fault_records[i].copy()
            fault_row["label"] = 1
            records.append(fault_row)
            labels.append(1)

    df = pd.DataFrame(records)
    y = np.asarray(labels, dtype=int)
    return df, y


def _train_and_score(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    figures_dir: Path,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Train a random forest classifier and compute industrial metrics."""
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, prob)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Fault"])
    ax.set_yticklabels(["Normal", "Fault"])
    ax.set_title(f"{model_name} Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{model_name.lower()}_confusion_matrix.png", dpi=300)
    plt.close(fig)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "false_positive_rate": float(fpr),
        "confusion_matrix": (int(tn), int(fp), int(fn), int(tp)),
    }
    return model, metrics


def _time_series_split_dataset(df: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Create train/test split without shuffling, preserving time order."""
    tss = TimeSeriesSplit(n_splits=3)
    train_idx, test_idx = next(tss.split(df))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy(), y[train_idx], y[test_idx]


def _plot_roc_curve(
    baseline_metrics: dict[str, Any],
    augmented_metrics: dict[str, Any],
    baseline_probs: np.ndarray,
    augmented_probs: np.ndarray,
    output_path: Path,
) -> None:
    """Create a comparison ROC curve for both scenarios."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    for label, probs, metrics in [("Baseline", baseline_probs, baseline_metrics), ("Augmented", augmented_probs, augmented_metrics)]:
        fpr, tpr, _ = roc_curve(metrics["y_true"], probs)
        ax.plot(fpr, tpr, label=f"{label} (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_training_evaluation(
    n_normal: int = 150,
    n_fault: int = 150,
    sample_rate: float = 2000,
    normal_duration: float = 1.0,
    fault_duration: float = 1.0,
    f_base: float = 30.0,
    random_state: int = 42,
    figures_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the baseline and augmented scenarios and compare industrial metrics."""
    base_dir = Path(figures_dir) if figures_dir is not None else Path("reports/figures")
    base_dir.mkdir(parents=True, exist_ok=True)

    df, y = _build_dataset(n_normal, n_fault, sample_rate, normal_duration, fault_duration, f_base, random_state)
    X_train, X_test, y_train, y_test = _time_series_split_dataset(df, y)

    baseline_model, baseline_metrics = _train_and_score(
        X_train[["rms", "kurtosis", "crest_factor", "peak_to_peak", "fft_amplitude_base", "fft_amplitude_2x", "fft_amplitude_3x"]],
        X_test[["rms", "kurtosis", "crest_factor", "peak_to_peak", "fft_amplitude_base", "fft_amplitude_2x", "fft_amplitude_3x"]],
        y_train,
        y_test,
        "Baseline",
        base_dir,
    )
    baseline_probs = baseline_model.predict_proba(X_test[["rms", "kurtosis", "crest_factor", "peak_to_peak", "fft_amplitude_base", "fft_amplitude_2x", "fft_amplitude_3x"]])[:, 1]
    baseline_metrics["y_true"] = y_test

    augmented_df = df.copy()
    augmented_df["label"] = y
    augmented_X = augmented_df.drop(columns=["label"]).copy()
    augmented_y = augmented_df["label"].to_numpy(dtype=int)
    aug_train, aug_test, aug_y_train, aug_y_test = _time_series_split_dataset(augmented_X, augmented_y)

    augmented_model, augmented_metrics = _train_and_score(
        aug_train,
        aug_test,
        aug_y_train,
        aug_y_test,
        "Augmented",
        base_dir,
    )
    augmented_probs = augmented_model.predict_proba(aug_test)[:, 1]
    augmented_metrics["y_true"] = aug_y_test

    _plot_roc_curve(baseline_metrics, augmented_metrics, baseline_probs, augmented_probs, base_dir / "roc_curve_comparison.png")

    summary = {
        "baseline": {
            "precision": baseline_metrics["precision"],
            "recall": baseline_metrics["recall"],
            "f1": baseline_metrics["f1"],
            "roc_auc": baseline_metrics["roc_auc"],
            "false_positive_rate": baseline_metrics["false_positive_rate"],
            "confusion_matrix": baseline_metrics["confusion_matrix"],
        },
        "augmented": {
            "precision": augmented_metrics["precision"],
            "recall": augmented_metrics["recall"],
            "f1": augmented_metrics["f1"],
            "roc_auc": augmented_metrics["roc_auc"],
            "false_positive_rate": augmented_metrics["false_positive_rate"],
            "confusion_matrix": augmented_metrics["confusion_matrix"],
        },
    }
    return summary


if __name__ == "__main__":
    summary = run_training_evaluation()
    print(summary)
