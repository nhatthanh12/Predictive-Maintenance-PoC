"""Smoke test for the physics-informed loose-bolt fault injector."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_generator import NormalBaselineGenerator
from src.fault_injector import FaultInjector


def main() -> None:
    report_dir = PROJECT_ROOT / "reports" / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    base_generator = NormalBaselineGenerator(sample_rate=2000, duration=2.0, f_base=30.0, random_state=42)
    baseline = base_generator.generate_signal(length=4000)

    injector = FaultInjector(sample_rate=2000.0, f_base=30.0, random_state=42)
    fault_signal = injector.generate_loose_bolt_signal(
        baseline_signal=baseline,
        looseness_level=0.9,
        impact_probability=0.18,
        impact_amplitude=1.6,
        frequency_shift=0.05,
    )

    freqs, amp_baseline = injector.fft_amplitude(baseline)
    _, amp_fault = injector.fft_amplitude(fault_signal)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
    axes[0].plot(np.arange(len(baseline)) / 2000.0, baseline, label="Normal baseline")
    axes[0].plot(np.arange(len(fault_signal)) / 2000.0, fault_signal, label="Loose-bolt fault", alpha=0.8)
    axes[0].set_title("Time-domain comparison")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()

    axes[1].plot(freqs, amp_baseline, label="Normal")
    axes[1].plot(freqs, amp_fault, label="Fault")
    axes[1].set_title("FFT amplitude comparison")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()

    plot_path = report_dir / "fft_comparison.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    fault_amp_05x = amp_fault[np.argmin(np.abs(freqs - 15.0))]
    fault_amp_2x = amp_fault[np.argmin(np.abs(freqs - 60.0))]
    fault_amp_3x = amp_fault[np.argmin(np.abs(freqs - 90.0))]
    baseline_amp_05x = amp_baseline[np.argmin(np.abs(freqs - 15.0))]
    baseline_amp_2x = amp_baseline[np.argmin(np.abs(freqs - 60.0))]
    baseline_amp_3x = amp_baseline[np.argmin(np.abs(freqs - 90.0))]

    assert len(fault_signal) == len(baseline)
    assert np.all(np.isfinite(fault_signal))
    assert fault_amp_05x > baseline_amp_05x * 1.05
    assert fault_amp_2x > baseline_amp_2x * 1.05
    assert fault_amp_3x > baseline_amp_3x * 1.05
    assert plot_path.exists(), "FFT comparison plot was not created."

    print("Generated fault signal length:", len(fault_signal))
    print("FFT comparison saved to:", plot_path)
    print(f"0.5x amplitude: normal={baseline_amp_05x:.6f}, fault={fault_amp_05x:.6f}")
    print(f"2x amplitude: normal={baseline_amp_2x:.6f}, fault={fault_amp_2x:.6f}")
    print(f"3x amplitude: normal={baseline_amp_3x:.6f}, fault={fault_amp_3x:.6f}")
    print("CHECK PASSED: synthetic fault injection creates physics-informed spectral signatures.")


if __name__ == "__main__":
    main()
