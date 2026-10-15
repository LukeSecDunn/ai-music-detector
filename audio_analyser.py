import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, dct

# CONVERT HERTZ TO MEL
def hz_to_mel(frequency):
    mel = 2595 * np.log10(1 + frequency / 700)
    return mel

# CONVERT MEL TO HERTZ
def mel_to_hz(mel):
    frequency = 700 * (10 ** (mel / 2595) - 1)
    return frequency

# CALCULATE THE LENGTH OF AUDIO
def calculate_duration(number_of_samples, sample_rate):
    duration = number_of_samples / sample_rate
    return duration

# CALCULATE RMS
def calculate_rms(window):
    rms = np.sqrt(np.mean(window.astype(float) ** 2))
    return rms

# CALCULATE ZERO-CROSSING RATE
def calculate_zcr(window):
    float_window = window.astype(float)

    zero_crossings = np.sum(
        float_window[:-1] * float_window[1:] < 0
    )

    zero_crossing_rate = zero_crossings / len(float_window)

    return zero_crossing_rate

# CALCULATE SPECTRAL CENTROID
def calculate_spectral_centroid(frequencies, magnitude):
    magnitude_sum = np.sum(magnitude)

    if magnitude_sum > 0:
        spectral_centroid = (
            np.sum(frequencies * magnitude) / magnitude_sum
        )
    else:
        spectral_centroid = 0.0

    return spectral_centroid

# CALCULATE SPECTRAL ROLLOFF
def calculate_spectral_rolloff(frequencies, magnitude):
    cumulative_magnitude = np.cumsum(magnitude)

    rolloff_threshold = cumulative_magnitude[-1] * 0.85

    rolloff_index = np.argmax(
        cumulative_magnitude >= rolloff_threshold
    )

    rolloff_frequency = frequencies[rolloff_index]

    return rolloff_frequency

# CALCULATE MFCCs
def calculate_mfccs(magnitude, filter_bank):
    # CONVERT MAGNITUDE INTO POWER SPECTRUM
    power_spectrum = magnitude ** 2

    # APPLY THE MEL FILTER BANK TO EXTRACT FREQUENCY BAND ENERGIES
    mel_energies = np.dot(
        filter_bank,
        power_spectrum
    )

    # CONVERT MEL ENERGIES TO LOG SCALE
    log_mel_energies = np.log(
        mel_energies + 1e-10
    )

    # APPLY DISCRETE COSINE TRANSFORM TO CALCULATE MFCCS
    mfccs = dct(
        log_mel_energies,
        type=2,
        norm="ortho"
    )

    mfccs = mfccs[:13]

    return mfccs

# EXTRACT FEATURES FOR EVERY WINDOW
def extract_features(audio, sample_rate):
    """
    EXTRACT 18 DSP FEATURES FROM OVERLAPPING AUDIO WINDOWS.
    
    INCLUDES RMS, ZCR, SPECTRAL FEATURES AND 13 MFCCS.
    RETURNS FRAME-LEVEL FEATURES AND SUPPORTING ANALYSIS DATA.
    """
    # SET WINDOW SIZE AND OVERLAP FOR AUDIO ANALYSIS
    window_size = 2048
    hop_size = 1024

    # SET NUMBER OF MEL FILTERS AND FREQUENCY RANGE
    num_mel_filters = 26

    low_freq = 0
    high_freq = sample_rate / 2

    # CONVERT FREQUENCY RANGE FROM HERTZ TO MEL SCALE
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)

    mel_points = np.linspace(
        low_mel,
        high_mel,
        num_mel_filters + 2
    )

    # CONVERT MEL POINTS BACK TO HERTZ AND FFT BIN POSITIONS
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor(
        (window_size + 1) * hz_points / sample_rate
    ).astype(int)

    num_fft_bins = window_size // 2 + 1

    # CREATE AN EMPTY MEL FILTER BANK
    filter_bank = np.zeros(
        (num_mel_filters, num_fft_bins)
    )

    # BUILD TRIANGULAR FILTERS FOR EACH MEL FREQUENCY BAND
    for m in range(num_mel_filters):
        left = bin_points[m]
        centre = bin_points[m + 1]
        right = bin_points[m + 2]

        for k in range(left, centre):
            filter_bank[m, k] = (k - left) / (centre - left)

        for k in range(centre, right):
            filter_bank[m, k] = (right - k) / (right - centre)

    # CREATE LISTS TO STORE AUDIO FEATURES AND ANALYSIS DATA
    rms_values = []
    zcr_values = []
    centroid_values = []
    rolloff_values = []
    mfcc_values = []
    feature_table = []
    spectra = []
    times = []

    # CREATE A HANN WINDOW TO REDUCE SPECTRAL LEAKAGE
    hann_window = np.hanning(window_size)

    # PROCESS AUDIO IN OVERLAPPING WINDOWS
    for start in range(0, len(audio), hop_size):
        end = start + window_size
        window = audio[start:end]

        if len(window) < window_size:
            break  # STOP IF THE LAST WINDOW IS SMALLER THAN THE SPECIFIED WINDOW SIZE

        # APPLY THE WINDOW TO THE CURRENT AUDIO FRAME
        windowed_audio = window * hann_window

        # CALCULATE RMS ENERGY AND ZERO CROSSING RATE
        rms = calculate_rms(window)
        rms_values.append(rms)

        zero_crossing_rate = calculate_zcr(window)
        zcr_values.append(zero_crossing_rate)

        # PERFORM FFT TO EXTRACT FREQUENCY INFORMATION
        fft_result = rfft(windowed_audio)
        magnitude = np.abs(fft_result)

        # CALCULATE AND STORE MFCCS FOR CURRENT FRAME
        mfccs = calculate_mfccs(
            magnitude,
            filter_bank
        )
        mfcc_values.append(mfccs)
    
        spectra.append(magnitude)

        # GENERATE FREQUENCY VALUES FOR FFT BINS
        frequencies = rfftfreq(window_size, 1 / sample_rate)

        # CALCULATE SPECTRAL CENTROID (FREQUENCY BRIGTHNESS)
        spectral_centroid = calculate_spectral_centroid(
            frequencies,
            magnitude
        )
        centroid_values.append(spectral_centroid)

        # CALCULATE SPECTRAL ROLLOFF FREQUENCY
        rolloff_frequency = calculate_spectral_rolloff(
            frequencies,
            magnitude
        )
        rolloff_values.append(rolloff_frequency)

        # IDENTIFY THE STRONGEST FREQUENCY IN THE SPECTRUM
        peak_index = np.argmax(magnitude)
        peak_frequency = frequencies[peak_index]

        # CALCULATE THE TIME POSITION OF THE CURRENT AUDIO FRAME
        time = start / sample_rate
        times.append(time)

        # COLLECT DSP FEATURES FOR THE CURRENT FRAME
        features = [
            time,
            rms,
            peak_frequency,
            spectral_centroid,
            rolloff_frequency,
            zero_crossing_rate
        ]
        feature_table.append(features)

    # CONVERT FEATURE LISTS INTO NUMPY ARRAYS
    feature_array = np.array(feature_table)
    mfcc_array = np.array(mfcc_values)

    # REMOVE TIME COLUMN FROM MACHINE LEARNING FEATURES
    audio_features = feature_array[:, 1:]

    # COMBINE 5 DSP FEATURES AND 13 MFCCS INTO 18
    combined_features = np.hstack((
        audio_features,
        mfcc_array
    ))

    return combined_features, times, rms_values, spectra, window_size

# RUN AUDIO ANALYSIS ONLY WHEN THIS FILE IS EXECUTED DIRECTLY
if __name__ == "__main__":

    file_name = input("Enter the WAV filename: ")

    sample_rate, audio = read(file_name)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    number_of_samples = len(audio)
    duration = calculate_duration(number_of_samples, sample_rate)

    print("File:", file_name)
    print("Sample rate:", sample_rate, "Hz")
    print("Samples:", number_of_samples)
    print("Duration:", duration, "seconds")
    print("Audio shape:", audio.shape)
    print("Number of dimensions:", audio.ndim)

    combined_features, times, rms_values, spectra, window_size = extract_features(
        audio,
        sample_rate
    )

    print("Extracted feature shape:", combined_features.shape)

    plt.figure()

    #RMS PLOT
    plt.plot(times, rms_values) 

    plt.xlabel("Time (seconds)")
    plt.ylabel("RMS Energy")
    plt.title("RMS Energy Over Time")

    plt.show()

    spectra = np.array(spectra)  # CONVERT THE LIST OF SPECTRA TO A NUMPY ARRAY FOR EASIER MANIPULATION

    spectra = spectra.T

    spectra_db = 20 * np.log10(spectra + 1e-10)

    image = plt.imshow(
        spectra_db,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, sample_rate / 2]
    )

    # SPECTROGRAM PLOT
    plt.colorbar(image, label="Magnitude (dB)")
    plt.ylim(0, 2000)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.show()

    window_duration = window_size / sample_rate
    print("Window duration:", window_duration, "seconds")
