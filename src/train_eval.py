"""Leakage-audited, time-ordered benchmark for severe and incipient faults."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.baseline_generator import NormalBaselineGenerator
from src.fault_injector import FaultInjector
from src.features import extract_signal_features

FEATURE_COLUMNS = [
    "rms", "kurtosis", "crest_factor", "peak_to_peak",
    "fft_amplitude_base", "fft_amplitude_2x", "fft_amplitude_3x",
]
SPEED_DRIFT_RANGE = (28.5, 31.5)


def _build_dataset(
    n_normal: int,
    n_fault: int,
    sample_rate: float,
    normal_duration: float,
    fault_duration: float,
    random_state: int,
    severity: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create independent, non-overlapping signal blocks in chronological order."""
    if severity not in {"severe", "incipient"}:
        raise ValueError("severity must be 'severe' or 'incipient'")

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []
    labels: list[int] = []
    block_id = 0

    for index in range(max(n_normal, n_fault)):
        if index < n_normal:
            operating_frequency = rng.uniform(*SPEED_DRIFT_RANGE)
            generator = NormalBaselineGenerator(
                sample_rate=sample_rate,
                duration=normal_duration,
                f_base=operating_frequency,
                random_state=random_state + block_id,
            )
            signal = generator.generate_signal(length=int(sample_rate * normal_duration))
            features = extract_signal_features(signal, sample_rate, operating_frequency)
            rows.append({**features, "block_id": block_id, "operating_frequency": operating_frequency})
            labels.append(0)
            block_id += 1

        if index < n_fault:
            operating_frequency = rng.uniform(*SPEED_DRIFT_RANGE)
            generator = NormalBaselineGenerator(
                sample_rate=sample_rate,
                duration=fault_duration,
                f_base=operating_frequency,
                random_state=random_state + block_id,
            )
            injector = FaultInjector(
                sample_rate=sample_rate,
                f_base=operating_frequency,
                random_state=random_state + block_id,
            )
            baseline = generator.generate_signal(length=int(sample_rate * fault_duration))
            signal = injector.generate_loose_bolt_signal(
                baseline_signal=baseline,
                looseness_level=0.9 if severity == "severe" else 0.35,
                impact_probability=0.18 if severity == "severe" else 0.04,
                impact_amplitude=1.6 if severity == "severe" else 0.4,
                frequency_shift=0.05,
                severity=severity,
            )
            features = extract_signal_features(signal, sample_rate, operating_frequency)
            rows.append({**features, "block_id": block_id, "operating_frequency": operating_frequency})
            labels.append(1)
            block_id += 1

    dataset = pd.DataFrame(rows)
    # One row represents one complete signal block; no sliding windows exist.
    if dataset["block_id"].duplicated().any():
        raise RuntimeError("Duplicate signal block detected; refusing to evaluate")
    return dataset, np.asarray(labels, dtype=int)


def _iter_time_splits(
    dataset: pd.DataFrame,
    labels: np.ndarray,
    n_splits: int = 3,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]]:
    """Yield chronological folds with a one-block embargo between train/test."""
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=1)
    for train_idx, test_idx in splitter.split(dataset):
        if set(train_idx).intersection(test_idx):
            raise RuntimeError("Train/test overlap detected")
        if dataset.iloc[train_idx]["block_id"].isin(dataset.iloc[test_idx]["block_id"]).any():
            raise RuntimeError("Signal block leakage detected")
        yield dataset.iloc[train_idx].copy(), dataset.iloc[test_idx].copy(), labels[train_idx], labels[test_idx]


def _summarize_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    scenario: str,
    severity: str,
    fold: int,
) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Severity": severity,
        "Scenario": scenario,
        "Fold": fold,
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, score)),
        "False Positive Rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def _save_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Normal", "Fault"])
    ax.set_yticks([0, 1], ["Normal", "Fault"])
    ax.set_title(title)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(cm[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _run_scenario(dataset: pd.DataFrame, labels: np.ndarray, severity: str, figures_dir: Path) -> pd.DataFrame:
    fold_rows: list[dict[str, Any]] = []
    total_cms = {name: np.zeros((2, 2), dtype=int) for name in ("Baseline", "Augmented")}

    for fold, (train, test, y_train, y_test) in enumerate(_iter_time_splits(dataset, labels), start=1):
        train_features = train[FEATURE_COLUMNS]
        test_features = test[FEATURE_COLUMNS]
        normal_train = train_features.loc[y_train == 0]
        if normal_train.empty or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            raise RuntimeError(f"Fold {fold} does not contain both required classes")

        baseline_model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
        baseline_model.fit(normal_train)
        baseline_train_score = -baseline_model.score_samples(normal_train)
        baseline_score = -baseline_model.score_samples(test_features)
        baseline_pred = (baseline_score > np.quantile(baseline_train_score, 0.95)).astype(int)
        baseline_cm = confusion_matrix(y_test, baseline_pred, labels=[0, 1])
        total_cms["Baseline"] += baseline_cm
        fold_rows.append(_summarize_metrics(y_test, baseline_pred, baseline_score, "Baseline", severity, fold))

        augmented_model = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
        augmented_model.fit(train_features, y_train)
        class_one = int(np.flatnonzero(augmented_model.classes_ == 1)[0])
        augmented_score = augmented_model.predict_proba(test_features)[:, class_one]
        augmented_pred = (augmented_score >= 0.5).astype(int)
        augmented_cm = confusion_matrix(y_test, augmented_pred, labels=[0, 1])
        total_cms["Augmented"] += augmented_cm
        fold_rows.append(_summarize_metrics(y_test, augmented_pred, augmented_score, "Augmented", severity, fold))

    _save_confusion_matrix(total_cms["Baseline"], f"{severity} Baseline Confusion Matrix", figures_dir / f"{severity.lower()}_baseline_confusion_matrix.png")
    _save_confusion_matrix(total_cms["Augmented"], f"{severity} Augmented Confusion Matrix", figures_dir / f"{severity.lower()}_augmented_confusion_matrix.png")
    return pd.DataFrame(fold_rows)


def run_training_evaluation(
    n_normal: int = 150,
    n_fault: int = 150,
    sample_rate: float = 2000,
    normal_duration: float = 1.0,
    fault_duration: float = 1.0,
    f_base: float = 30.0,
    random_state: int = 42,
    figures_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Benchmark both models across Severe and Incipient drifting-speed faults."""
    output_dir = Path(figures_dir) if figures_dir is not None else Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    del f_base

    all_rows = []
    for severity in ("Severe", "Incipient"):
        dataset, labels = _build_dataset(
            n_normal, n_fault, sample_rate, normal_duration, fault_duration, random_state, severity.lower()
        )
        all_rows.append(_run_scenario(dataset, labels, severity, output_dir))

    fold_results = pd.concat(all_rows, ignore_index=True)
    metric_columns = ["Precision", "Recall", "F1-Score", "ROC-AUC", "False Positive Rate"]
    aggregate = fold_results.groupby(["Severity", "Scenario"])[metric_columns].agg(["mean", "std"]).reset_index()
    aggregate.columns = [
        " ".join(column).strip() if isinstance(column, tuple) else column
        for column in aggregate.columns
    ]
    fold_results.to_csv(output_dir / "metric_comparison_by_fold.csv", index=False)
    aggregate.to_csv(output_dir / "metric_comparison_summary.csv", index=False)

    print("\n=== Leakage-audited Industrial Stress Test ===")
    print(f"TimeSeriesSplit: 3 folds, gap=1 block, speed drift={SPEED_DRIFT_RANGE[0]}-{SPEED_DRIFT_RANGE[1]} Hz")
    print(aggregate.to_string(index=False))
    print(f"\nSaved fold results: {output_dir / 'metric_comparison_by_fold.csv'}")
    print(f"Saved aggregate results: {output_dir / 'metric_comparison_summary.csv'}")
    return aggregate


if __name__ == "__main__":
    run_training_evaluation()
