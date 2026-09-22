"""Integration test for the leakage-audited evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_eval import run_training_evaluation


def test_run_training_evaluation_generates_metrics_and_figures() -> None:
    output_dir = PROJECT_ROOT / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_training_evaluation(
        n_normal=60,
        n_fault=60,
        sample_rate=2000,
        normal_duration=1.0,
        fault_duration=1.0,
        f_base=30.0,
        random_state=42,
        figures_dir=output_dir,
    )

    assert isinstance(summary, pd.DataFrame)
    assert {"Severity", "Scenario", "Recall mean", "F1-Score mean", "False Positive Rate mean"}.issubset(set(summary.columns))
    assert set(summary["Severity"]) == {"Severe", "Incipient"}
    assert set(summary["Scenario"]) == {"Baseline", "Augmented"}
    assert summary[["Recall mean", "F1-Score mean", "False Positive Rate mean"]].notna().all().all()
    assert (output_dir / "severe_baseline_confusion_matrix.png").exists()
    assert (output_dir / "severe_augmented_confusion_matrix.png").exists()
    assert (output_dir / "incipient_baseline_confusion_matrix.png").exists()
    assert (output_dir / "incipient_augmented_confusion_matrix.png").exists()
    assert (output_dir / "metric_comparison_by_fold.csv").exists()
