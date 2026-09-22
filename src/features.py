"""Feature extraction for industrial vibration and electrical signals."""

from __future__ import annotations

import numpy as np


def rms(signal: np.ndarray) -> float:
    """Root Mean Square (RMS) of the signal."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal))))


def kurtosis(signal: np.ndarray) -> float:
    """Fourth standardized moment of the signal distribution."""
    signal = np.asarray(signal, dtype=float)
    if signal.size < 2:
        return 0.0
    centered = signal - np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return 0.0
    return float(np.mean(np.power(centered, 4)) / (std**4))


def crest_factor(signal: np.ndarray) -> float:
    """Peak-to-RMS ratio, indicating impulsive behavior in the waveform."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return 0.0
    r = rms(signal)
    if r == 0:
        return 0.0
    return float(np.max(np.abs(signal)) / r)


def peak_to_peak(signal: np.ndarray) -> float:
    """Difference between maximum and minimum amplitude."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return 0.0
    return float(np.max(signal) - np.min(signal))


def fft_amplitude_spectrum(signal: np.ndarray, sample_rate: float = 2000.0) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency bins and amplitude spectrum for a real-valued signal."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.array([]), np.array([])
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    amplitudes = np.abs(spectrum) / signal.size
    return freqs, amplitudes


def extract_signal_features(signal: np.ndarray, sample_rate: float = 2000.0, base_frequency: float = 30.0) -> dict:
    """Extract a compact set of industrial health indicators from a time series.

    The returned dictionary includes time-domain metrics and FFT amplitudes near
    the dominant operating frequency. This is useful for baseline comparison and
    fault detection pipelines.
    """
    signal = np.asarray(signal, dtype=float)
    freqs, amplitudes = fft_amplitude_spectrum(signal, sample_rate)

    if freqs.size == 0:
        return {
            "rms": 0.0,
            "kurtosis": 0.0,
            "crest_factor": 0.0,
            "peak_to_peak": 0.0,
            "fft_amplitude_base": 0.0,
            "fft_amplitude_2x": 0.0,
            "fft_amplitude_3x": 0.0,
        }

    base_idx = int(np.argmin(np.abs(freqs - base_frequency)))
    second_idx = int(np.argmin(np.abs(freqs - 2 * base_frequency)))
    third_idx = int(np.argmin(np.abs(freqs - 3 * base_frequency)))

    return {
        "rms": rms(signal),
        "kurtosis": kurtosis(signal),
        "crest_factor": crest_factor(signal),
        "peak_to_peak": peak_to_peak(signal),
        "fft_amplitude_base": float(amplitudes[base_idx]),
        "fft_amplitude_2x": float(amplitudes[second_idx]),
        "fft_amplitude_3x": float(amplitudes[third_idx]),
    }
