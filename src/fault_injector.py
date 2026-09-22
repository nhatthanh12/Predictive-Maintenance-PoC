"""Physics-informed synthetic fault generator for loose-bolt / misalignment conditions.

The fault model is designed to match common industrial signatures observed in
rotating machinery: a dominant shaft component, subharmonic content near 0.5x,
integer harmonics at 2x and 3x, and intermittent impulsive impacts caused by
loose fasteners or asymmetric stiffness. These signatures are injected into a
healthy baseline signal while preserving a reproducible random seed.
"""

from __future__ import annotations

import numpy as np


class FaultInjector:
    """Inject loose-bolt fault signatures into a healthy vibration signal.

    Parameters
    ----------
    sample_rate : float, default=2000.0
        Sampling frequency in Hz.
    f_base : float, default=30.0
        Fundamental shaft frequency in Hz.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(self, sample_rate: float = 2000.0, f_base: float = 30.0, random_state: int = 42) -> None:
        self.sample_rate = sample_rate
        self.f_base = f_base
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fft_amplitude(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return frequency bins and single-sided FFT amplitudes."""
        signal = np.asarray(signal, dtype=float)
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / self.sample_rate)
        amplitudes = np.abs(spectrum) / signal.size
        return freqs, amplitudes

    def generate_loose_bolt_signal(
        self,
        baseline_signal: np.ndarray,
        looseness_level: float = 0.8,
        impact_probability: float = 0.15,
        impact_amplitude: float = 1.5,
        frequency_shift: float = 0.05,
    ) -> np.ndarray:
        """Create a synthetic loose-bolt signal using physics-informed modulation.

        Parameters
        ----------
        baseline_signal : np.ndarray
            Healthy signal to inject the fault into.
        looseness_level : float
            Scalar controlling magnitude of asymmetry and harmonic amplification.
        impact_probability : float
            Probability of an intermittent impact in each time step.
        impact_amplitude : float
            Peak amplitude of each impact pulse.
        frequency_shift : float
            Relative frequency detuning used to simulate stiffness asymmetry.
        """
        signal = np.asarray(baseline_signal, dtype=float).copy()
        n = signal.size
        t = np.arange(n) / self.sample_rate

        base_frequency = self.f_base * (1.0 + frequency_shift * (self.rng.uniform(-1.0, 1.0)))
        subharmonic = 8.0 * looseness_level * np.sin(2 * np.pi * (0.5 * base_frequency) * t)
        second_harmonic = 12.0 * looseness_level * np.sin(2 * np.pi * (2 * base_frequency) * t + 0.9)
        third_harmonic = 15.0 * looseness_level * np.sin(2 * np.pi * (3 * base_frequency) * t + 1.3)

        modulation = 1.0 + 0.35 * np.sin(2 * np.pi * 1.2 * t)
        signal = signal * (1.0 + 0.35 * looseness_level * modulation)

        impact_mask = self.rng.random(n) < impact_probability
        impact_pattern = np.zeros(n)
        impact_pattern[impact_mask] = 4.0 * impact_amplitude * self.rng.normal(1.0, 0.25, size=np.sum(impact_mask))
        impact_pattern *= np.exp(-2.5 * np.abs(np.sin(2 * np.pi * 2.0 * t)))

        fault_signal = signal + subharmonic + second_harmonic + third_harmonic + impact_pattern
        fault_signal = fault_signal - np.mean(fault_signal)
        return fault_signal.astype(float)


if __name__ == "__main__":
    baseline = np.random.default_rng(42).normal(0.0, 1.0, 2000)
    injector = FaultInjector(sample_rate=2000.0, f_base=30.0)
    synthetic = injector.generate_loose_bolt_signal(baseline)
    print(f"Fault signal length: {len(synthetic)}")
