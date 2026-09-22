"""Smoke test for the healthy baseline signal generation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_generator import NormalBaselineGenerator
from src.features import extract_signal_features


def test_baseline_generation_and_feature_extraction() -> None:
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    generator = NormalBaselineGenerator(
        sample_rate=2000,
        duration=1.0,
        f_base=30.0,
        noise_std=0.12,
        random_state=42,
    )
    signal = generator.generate_signal(length=2000)
    output_path = data_dir / "sample_baseline.csv"

    df = pd.DataFrame({
        "timestamp_s": np.arange(len(signal)) / generator.sample_rate,
        "sensor_value": signal,
    })
    df.to_csv(output_path, index=False)

    feature_dict = extract_signal_features(signal, sample_rate=generator.sample_rate, base_frequency=generator.f_base)
    features_df = pd.DataFrame([feature_dict])

    assert len(signal) == 2000, "Signal length must be 2000 samples."
    assert np.all(np.isfinite(signal)), "Signal contains non-finite values."
    assert feature_dict["rms"] > 0, "RMS should be positive."
    assert feature_dict["fft_amplitude_base"] > 0, "Fundamental FFT amplitude should be positive."
    assert len(features_df) == 1
