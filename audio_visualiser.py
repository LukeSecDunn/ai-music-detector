import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from audio_analyser import extract_features

import argparse
from pathlib import Path


def load_audio(filename):
    sample_rate, audio = wavfile.read(filename)

    # CONVERT INTIGER PCM AUDIO INTO FLOATING-POINT SAMPLES.
    if np.issubdtype(audio.dtype, np.integer):
        if np.issubdtype(audio.dtype, np.unsignedinteger):
            midpoint = (np.iinfo(audio.dtype).max + 1) / 2
            audio = (audio.astype(np.float64) - midpoint) / midpoint
        else:
            scale = max(
                abs(np.iinfo(audio.dtype).min),
                np.iinfo(audio.dtype).max
            )
            audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)

    # CONVERT STEREO TO MONO
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    return sample_rate, audio


def visualise_audio(filename):
    sample_rate, audio = load_audio(filename)

    features, times, rms_values, spectra, window_size = extract_features(
        audio, sample_rate
    )

    if len(features) == 0:
        raise ValueError("Audio is too short for feature extraction.")

    duration = len(audio) / sample_rate
    waveform_times = np.arange(len(audio)) / sample_rate

    # COLUMN 3 OF OUR FEATURE ARRAY IS SPECTRAL CENTROID
    centroid = features[:, 2]

    fig, axes = plt.subplots(4, 1, figsize=(16, 12))

    # 1. WAVEFORM
    axes[0].plot(waveform_times, audio, linewidth=0.4)
    axes[0].set_title("Waveform")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, duration)

    # 2. SPECTROGRAM
    axes[1].specgram(
        audio,
        NFFT=2048,
        Fs=sample_rate,
        noverlap=1024,
        cmap="magma",
        scale="dB",
    )
    axes[1].set_title("Spectrogram")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_ylim(0, min(5000, sample_rate / 2))
    axes[1].set_xlim(0, duration)

    # 3. RMS ENERGY
    axes[2].plot(times, rms_values)
    axes[2].set_title("RMS Energy")
    axes[2].set_ylabel("RMS")
    axes[2].set_xlim(0, duration)

    # 4. SPECTRAL CENTROID
    axes[3].plot(times, centroid)
    axes[3].set_title("Spectral Centroid")
    axes[3].set_ylabel("Frequency (Hz)")
    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_xlim(0, duration)

    fig.suptitle(f"Audio Analysis: {filename}", fontsize=15)
    plt.tight_layout()

    # SAVE AN IMAGE OF THE TRACK ANALYSIS
    plt.savefig("audio_analysis.png", dpi=150)
    plt.show()

    print("Dashboard saved as audio_analysis.png")

# COMMAND LINE INTERFACE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Generate an audio DSP analysis dashboard from a WAV file."
    )

    parser.add_argument(
        "filename",
        help = "Path to the WAV file you want to analyse."
    )

    args = parser.parse_args()

    audio_file = Path(args.filename)

    if not audio_file.is_file():
        parser.error(f"Audio file not found: {audio_file}")

    visualise_audio(str(audio_file))