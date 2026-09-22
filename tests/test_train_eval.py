"""Integration test for the training and evaluation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_eval import run_training_evaluation


def main() -> None:
    output_dir = PROJECT_ROOT / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_training_evaluation(
        n_normal=120,
        n_fault=120,
        sample_rate=2000,
        normal_duration=1.0,
        fault_duration=1.0,
        f_base=30.0,
        random_state=42,
        figures_dir=output_dir,
    )

    assert set(summary.keys()) == {"baseline", "augmented"}
    assert summary["baseline"]["roc_auc"] >= 0.5
    assert summary["augmented"]["roc_auc"] >= 0.5
    assert (output_dir / "baseline_confusion_matrix.png").exists()
    assert (output_dir / "augmented_confusion_matrix.png").exists()
    assert (output_dir / "roc_curve_comparison.png").exists()

    print("CHECK PASSED: training and evaluation pipeline is valid.")


if __name__ == "__main__":
    main()
