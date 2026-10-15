import numpy as np
import pytest

from audio_analyser import extract_features


SAMPLE_RATE = 44100


def get_features(audio):
    features, times, rms, spectra, window_size = extract_features(
        audio, SAMPLE_RATE
    )
    return np.asarray(features), np.asarray(times), np.asarray(rms)


def test_sine_wave():
    """A 440 Hz sine wave should produce valid audio features."""
    duration = 1.0
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    features, times, rms = get_features(audio)

    assert features.ndim == 2
    assert features.shape[1] == 18
    assert len(features) == len(times) == len(rms)
    assert np.all(np.isfinite(features))
    assert np.mean(rms) > 0


def test_silence():
    """Silence should not produce NaN or infinite values."""
    audio = np.zeros(SAMPLE_RATE)

    features, times, rms = get_features(audio)

    assert np.all(np.isfinite(features))
    assert np.allclose(rms, 0)
    assert np.allclose(features[:, 0], 0)


def test_louder_signal_has_higher_rms():
    """Increasing amplitude should increase RMS energy."""
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE

    quiet = 0.1 * np.sin(2 * np.pi * 440 * t)
    loud = 0.8 * np.sin(2 * np.pi * 440 * t)

    _, _, quiet_rms = get_features(quiet)
    _, _, loud_rms = get_features(loud)

    assert np.mean(loud_rms) > np.mean(quiet_rms)
    