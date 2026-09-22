"""Generate healthy baseline sensor signals for predictive maintenance experiments.

The signal models a rotating machine under normal operating condition. The
combination of the fundamental shaft frequency, weak harmonics and Gaussian
industrial noise reflects healthy sensor behavior commonly observed in bearing
and motor diagnostics.
"""

from __future__ import annotations

import numpy as np


class NormalBaselineGenerator:
    """Generate a normal baseline vibration/current signal for a healthy machine.

    Parameters
    ----------
    sample_rate : float, default=2000
        Sampling rate in Hertz. The project requirement uses 2000 Hz for a
        realistic industrial monitoring setup.
    duration : float, default=2.0
        Signal duration in seconds.
    f_base : float, default=30.0
        Fundamental shaft frequency in Hertz. This represents the dominant
        rotational component of the healthy motor.
    noise_std : float, default=0.12
        Standard deviation of the additive Gaussian background noise that models
        industrial floor noise.
    random_state : int, default=42
        Seed used to make the generated data reproducible.
    """

    def __init__(
        self,
        sample_rate: float = 2000.0,
        duration: float = 2.0,
        f_base: float = 30.0,
        noise_std: float = 0.12,
        random_state: int = 42,
    ) -> None:
        self.sample_rate = sample_rate
        self.duration = duration
        self.f_base = f_base
        self.noise_std = noise_std
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def generate_signal(self, length: int | None = None) -> np.ndarray:
        """Return a normalized healthy baseline signal.

        The waveform combines a fundamental component with second and third
        harmonics, a slow amplitude modulation and Gaussian noise. This structure
        is consistent with healthy behavior where the dominant frequency and its
        harmonics remain stable while background noise is present.
        """
        if length is None:
            length = int(self.sample_rate * self.duration)

        t = np.arange(length) / self.sample_rate

        fundamental = np.sin(2 * np.pi * self.f_base * t)
        second_harmonic = 0.25 * np.sin(2 * np.pi * (2 * self.f_base) * t + 0.5)
        third_harmonic = 0.15 * np.sin(2 * np.pi * (3 * self.f_base) * t + 0.8)
        modulation = 1.0 + 0.06 * np.sin(2 * np.pi * 0.8 * t)

        signal = (fundamental + second_harmonic + third_harmonic) * modulation
        signal += self.rng.normal(0.0, self.noise_std, size=length)

        signal = signal - np.mean(signal)
        signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
        return signal.astype(float)


if __name__ == "__main__":
    generator = NormalBaselineGenerator(sample_rate=2000, duration=2.0, f_base=30.0)
    signal = generator.generate_signal()
    print(f"Generated {len(signal)} samples at {generator.sample_rate} Hz.")
