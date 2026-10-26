# AI MUSIC DETECTOR - DATASET PREPARATION
# EXTRACTS AUDIO FEATURS FOR FUTURE MACHINE LEARNING CLASSIFICATION

import numpy as np
from scipy.io.wavfile import read
from audio_analyser import extract_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

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
        "label": 0,
    },
    {
        "filename": "Lotrax - BOK.wav",
        "label": 0,
    },
    {
        "filename": "Lotrax - Lose Ma Mind.wav",
        "label": 0,
    },
    {
        "filename": "Lotrax - Nothing 2 Lose.wav",
        "label": 0,
    },
    {
        "filename": "Lotrax - Suga Dip.wav",
        "label": 0,
    },
    {
        "filename": "Slop Detective - Track 1.wav",
        "label": 1,
    },
    {
        "filename": "Slop Detective - Track 2.wav",
        "label": 1,
    },
    {
        "filename": "Slop Detective - Track 3.wav",
        "label": 1,
    },
    {
        "filename": "Slop Detective - Track 4.wav",
        "label": 1,
    },
    {
        "filename": "Slop Detective - Track 5.wav",
        "label": 1,
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

# STAGE 14 - TRAIN FIRST AI MUSIC CLASSIFIER

# SPLIT TRACKS INTO TRAINING AND TESTING DATASETS
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.4,
    random_state = 42,
    stratify = y
)

# IDENTIFY TRAINING AND TESTING TRACKS
track_indices = np.arange(len(songs))

train_indices, test_indices = train_test_split(
    track_indices,
    test_size = 0.4,
    random_state = 42,
    stratify = y
)

print("\nTRAINING TRACKS:")
for i in train_indices:
    print(songs[i]["filename"])

print("\nTESTING TRACKS:")
for i in test_indices:
    print(songs[i]["filename"])

print("Training tracks:", len(X_train))
print("Testing tracks:", len(X_test))

# MACHINE LEARNING PIPELINE
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors = 3))
])

# TRAIN THE MODEL USING TRAINING TRACKS
model.fit(X_train, y_train)

# PREDICT LABELS FOR TRACKS THE MODEL HASN'T TRAINED ON
predictions = model.predict(X_test)

# EVALUATE THE PREDICTIONS
accuracy = accuracy_score(y_test, predictions)

print("\nModel accuracy:", f"{accuracy:.1%}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels = [0, 1]))

# COMPARE EACH PREDICTION WITH THE CORRECT ANSWER
for actual, predicted in zip(y_test, predictions):
    print(
        f"Actual: {'AI' if actual == 1 else 'Human'} | "
        f"Predicted: {'AI' if predicted == 1 else 'Human'}"
    )

# PREDICT AN UNSEEN TRACK
new_track = "Bl3ss x CamrinWatsin Feat. Bbyclose - Kisses (Malugi Remix).wav"

new_features = analyse_track(new_track)

new_features = new_features.reshape(1, -1)

prediction = model.predict(new_features)[0]

print("\nUNSEEN TRACK PREDICTION")
print("Track:", new_track)
print("Prediction:", "AI" if prediction == 1 else "Human")
