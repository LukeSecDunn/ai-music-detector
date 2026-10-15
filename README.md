## AI MUSIC DETECTOR

## AUDIO DSP & MACHINE LEARNING RESEARCH PROJECT

A Python-based audio analysis and machine-learning research project
exploring whether digital signal processing (DSP) features can help
distinguish human-produced music from AI-generated music.

## PROJECT STATUS
Active development. The DSP analysis and visualisation pipeline is
operational; AI classification is awaiting a suitably documented
training dataset.

## PROJECT OVERVIEW
This project combines my background in Audio Technology and music
production with Python programming, DSP and machine learning.

The initial objective is to classify complete human-produced and
AI-generated recordings. The longer-term goal is to investigate
the detection of AI-generated vocals and instruments within
otherwise human-produced music.

## IMPLEMENTED FUNCTIONALITY
- WAV audio loading, PCM normalisation and stereo to mono conversion
- Frame-based FFT analysis with Hann windowing
- RMS energy and zero-crossing rate
- Spectral centroid, spectral rolloff and peak frequency
- 13 Mel-frequency cepstral coefficients (MFCCs)
- 18-dimensional audio feature vectors
- Track-level feature aggregation
- Four-panel visualisation dashboard
- Command-line interface for analysing WAV recordings
- Automated DSP tests using pytest

## AUDIO ANALYSIS DASHBOARD
The visualiser generates four plots: waveform, spectrogram, RMS energy,
and spectral centroid.

## TECHNOLOGY
Python, NumPy, SciPy, Matplotlib, scikit-learn and pytest

## INSTALLATION
Install the project dependencies:
- python -m pip install -r requirements.txt

## USAGE
Run the audio dashboard on a WAV recording:
- python audio_visualiser.py "your_track.wav"

## Run the DSP tests:
- python -m pytest -v test_audio_analyser.py

## RESEARCH METHODOLOGY
The planned initial dataset will use complete recordings, with human-
produced and AI-generated music matched closely as practical by
genre and duration.

The first classifier will be treated as an experimental baseline
rather than a validated commercial AI-detection system.

## DATASET LIMITATIONS AND LICENSING
Potential confounding factors include genre, mastering, recording
length, compression, producer identity and AI generator identity.

Training and evaluation must be seperated by recording to prevent
data leakage.

Audio files are excluded from the public repository. Each dataset
source will require documented provenance and premission for the
intended research use. Download or commercial-use rights do not
automatically grant unrestricted ML-training or redistribution
rights.

## ROADMAP
- [x] DSP feature extraction
- [x] Audio analysis dashboard
- [x] Automated tests
- [x] Command-line interface
- [ ] Curate a documented dataset
- [ ] Train and evaluate a baseline classifier
- [ ] Expand to additional generators and human producers
- [ ] Investigate partially AI-generated music

## ABOUT
Developed as a personal, non-commercial research portfolio project
to demonstrate practical audio DSP, Python programming and
machine-learning skills