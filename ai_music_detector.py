# AI MUSIC DETECTOR - DATASET PREPARATION
# EXTRACTS AUDIO FEATURS FOR FUTURE MACHINE LEARNING CLASSIFICATION

import numpy as np
from scipy.io.wavfile import read
from audio_analyser import extract_features

# LOAD AUDIO AND EXTRACT A TRACK-LEVEL FEATURE SUMMARY
def analyse_track(file_name):
    sample_rate, audio = read(file_name)

    # NORMALISE SIGNED INTEGER AUDIO TO FLOATING POINT VALUES
    if np.issubdtype(audio.dtype, np.signedinteger):
        max_value = np.iinfo(audio.dtype) .max + 1
        audio = audio.astype(np.float64) / max_value

    # CONVERT STEREO AUDIO TO MONO
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # EXTRACT 18 DSP FEATURES ACROSS ALL WINDOWS TO REPRESENT THE WHOLE TRACK
    features, times, rms_values, spectra, window_size = extract_features(
        audio, sample_rate
    )

    track_summary = np.mean(features, axis=0)

    return track_summary

# DEFINE TRACKS AND LABELS: 0 = HUMAN, 1 = AI
songs = [
    {
        "filename": "Lotrax - Give Me Some Luv.wav",
        "label": 0
    }
]

# CREATE LISTS TO STORE TRACK FEATURES AND LABELS
features_list = []
labels_list = []

# ANALYSE EACH TRACK AND COLLECT ITS FEATURES
for song in songs:
    summary = analyse_track(song["filename"])

    features_list.append(summary)
    labels_list.append(song["label"])

# CONVERT FEATURES AND LABELS INTO NUMPY ARRAY FOR MACHINE LEARNING
X = np.array(features_list)
y = np.array(labels_list)

# CHECK EACH TRACK HAS A CORRESPONDING LABEL
assert len(X) == len(y)

# CHECK FEATURES CONTAIN NO NAN OR INFINITE VALUES
assert np.isfinite(X).all(), "Dataset contains invalid audio features!"

# CHECK LABELS ARE VALID: 0 = HUMAN, 1 = AI
assert np.isin(y, [0, 1]).all(), "Dataset contains an unknown label!"

# VALIDATE DATASET DIMENSIONS AND FEATURE COUNT
assert X.ndim == 2, "Feature matrix must be 2D!"

assert y.ndim == 1, "Labels must be a 1D array!"

assert X.shape[1] == 18, "Expected 18 audio features per track!"

assert X.shape[0] > 0, "Dataset contains no tracks!"

# COUNT THE NUMBER OF HUMAN AND AI TRACKS
unique_labels, counts = np.unique(y, return_counts=True)

for label, count in zip(unique_labels, counts):
    print("Human" if label == 0 else "AI", "tracks:", count)

# DISPLAY FINAL DATASET DIMENSIONS
print("Dataset ready:", X.shape, y.shape)