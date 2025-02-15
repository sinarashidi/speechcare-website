import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import welch
import parselmouth


def visualize_shap_spectrogram(
    audio_path,
    shap_values,
    label,
    sr=16000,
    segment_length=5,
    overlap=0.2,
    merge_frame_duration=0.3,
    formants_to_plot=None,
    fig_save_path=None,
    ax=None
):
    """
    Visualize the spectrogram with intensity modified by SHAP values, with optional formant plotting.

    Args:
        audio_path (str): Path to the audio file.
        shap_values (np.ndarray): SHAP values of shape (1, num_segments, seq_length, num_labels).
        label (int): The target label for visualization (0, 1, or 2).
        sr (int): Sampling rate of the audio file.
        segment_length (float): Length of each segment in seconds.
        overlap (float): Overlap ratio between segments.
        merge_frame_duration (float): Duration of merged frames in seconds.
        formants_to_plot (list): List of formants to plot (e.g., ["F0", "F1", "F2", "F3"]).
        fig_save_path (str, optional): Path to save the figure.
        ax (matplotlib.axes.Axes, optional): Axis to plot on for subplots. If None, creates a new plot.

    Returns:
        None: Displays or saves the spectrogram.
    """
    name = os.path.splitext(os.path.basename(audio_path))[0]

    # Step 1: Load audio and compute spectrogram
    audio, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, power=2.0)
    log_S = librosa.power_to_db(S, ref=np.max)  # Convert to decibels

    # Step 2: Aggregate SHAP values for the specified label
    shap_values_label = shap_values[0, :, :, label]
    segment_samples = int(segment_length * sr)
    hop_samples = int(segment_samples * (1 - overlap))

    # Merge SHAP values into larger frames
    merge_samples = int(merge_frame_duration * sr)
    merged_shap_values = []
    for segment in shap_values_label:
        reshaped_segment = segment[: len(segment) // merge_samples * merge_samples]
        reshaped_segment = reshaped_segment.reshape(-1, merge_samples)
        merged_shap_values.append(reshaped_segment.mean(axis=1))
    merged_shap_values = np.concatenate(merged_shap_values)

    # Normalize SHAP values for enhanced contrast
    merged_shap_values_normalized = (merged_shap_values - np.percentile(merged_shap_values, 5)) / (
        np.percentile(merged_shap_values, 95) - np.percentile(merged_shap_values, 5)
    )
    merged_shap_values_normalized = np.clip(merged_shap_values_normalized, 0, 1)

    # Apply nonlinear transformation for more intensity difference
    merged_shap_values_transformed = merged_shap_values_normalized**5
    merged_shap_values_transformed *= 10
    merged_shap_values_transformed += 0.01

    # Step 3: Modify the spectrogram intensity
    audio_duration = len(audio) / sr
    merged_frame_times = np.arange(0, len(merged_shap_values)) * merge_frame_duration
    time_bins = np.linspace(0, audio_duration, S.shape[1])

    for i, t in enumerate(merged_frame_times):
        idx_start = np.searchsorted(time_bins, t)
        idx_end = np.searchsorted(time_bins, t + merge_frame_duration)
        if idx_start < len(S[0]) and idx_end < len(S[0]):
            S[:, idx_start:idx_end] *= merged_shap_values_transformed[i]

    # Step 4: Extract formants if formants_to_plot is specified
    formant_values = {}
    if formants_to_plot:
        sound = parselmouth.Sound(audio_path)
        pitch = sound.to_pitch()
        time_stamps = pitch.ts()
        f0_values = pitch.selected_array["frequency"]
        f0_values[f0_values == 0] = np.nan
        formant = sound.to_formant_burg(time_step=0.1)
        times = np.arange(0, audio_duration, 0.01)
        formant_values = {"F0": f0_values, "F1": [], "F2": [], "F3": []}
        for t in times:
            formant_values["F1"].append(formant.get_value_at_time(1, t))
            formant_values["F2"].append(formant.get_value_at_time(2, t))
            formant_values["F3"].append(formant.get_value_at_time(3, t))

    # Step 5: Plot the spectrogram
    labels = {0: "Healthy", 1: "MCI", 2: "ADRD"}
    modified_log_S = librosa.power_to_db(S, ref=np.max)

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 4))

    img = librosa.display.specshow(modified_log_S, sr=sr, x_axis="time", y_axis="mel", cmap="viridis", ax=ax)
    # ax.set_title(f"UID: {name}, Spectrogram with SHAP-Adjusted Intensity (Label: {labels[label]})")
    ax.set_xlabel("")
    ax.set_ylabel("Frequency (Hz)", fontsize=16)
    ax.set_xlim(0, audio_duration)
    # ax.set_xticks(np.arange(0, audio_duration + 1, 1))
    ax.set_xticks([])

    formant_colors = {"F0": 'red', "F1": 'cyan', "F2": 'white', "F3": 'orange'}
    if formants_to_plot:
        for formant in formants_to_plot:
            if formant in formant_values:
                ax.plot(
                    times if formant != "F0" else time_stamps,
                    formant_values[formant],
                    label=formant,
                    linewidth=3 if formant == "F0" else 2,
                    color=formant_colors[formant]
                )
        ax.legend(loc='upper right')

    # plt.colorbar(img, ax=ax, format="%+2.0f dB")
    if fig_save_path:
        folder_path = os.path.dirname(fig_save_path)
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(fig_save_path, dpi=600, bbox_inches="tight")

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def frequency_shannon_entropy(
    audio_path, frame_length_ms=25, frame_step_ms=10, windowing_function="hamming",
    smooth=True, smooth_window=5, ax=None
):
    """
    Calculates and plots the frequency Shannon entropy for an audio file, with optional smoothing.

    Parameters:
    - audio_path (str): Path to the audio file.
    - frame_length_ms (float): Frame length in milliseconds.
    - frame_step_ms (float): Step size between frames in milliseconds.
    - windowing_function (str): Windowing function to apply (default: "hamming").
    - smooth (bool): Whether to smooth the entropy values.
    - smooth_window (int): Window size for smoothing.
    - ax (matplotlib.axes.Axes, optional): Axis to plot on for subplots. If None, creates a new plot.

    Returns:
    - np.ndarray: (Original or smoothed) entropy values.
    """
    name = os.path.splitext(os.path.basename(audio_path))[0]
    # Load the audio file
    signal, sr = librosa.load(audio_path, sr=None)
    audio_duration = len(signal) / sr

    # Convert frame length and step size to samples
    frame_length_samples = int(frame_length_ms * sr / 1000)
    frame_step_samples = int(frame_step_ms * sr / 1000)

    # Select windowing function
    if windowing_function == "hamming":
        window = np.hamming(frame_length_samples)
    else:
        raise ValueError("Unsupported windowing function")

    # Calculate the number of frames (no padding)
    num_frames = max(1, 1 + (len(signal) - frame_length_samples) // frame_step_samples)

    entropy_values = []

    # Calculate entropy for each frame
    for i in range(num_frames):
        start_idx = i * frame_step_samples
        end_idx = start_idx + frame_length_samples
        if end_idx > len(signal):  # Skip frames beyond the actual signal
            break
        frame = signal[start_idx:end_idx] * window

        # Calculate the power spectral density using Welch's method
        frequencies, power_spectrum = welch(frame, fs=sr, nperseg=frame_length_samples)

        # Convert power spectrum to probability distribution
        power_spectrum_prob_dist = power_spectrum / (np.sum(power_spectrum) + np.finfo(float).eps)

        # Calculate Shannon entropy
        entropy = -np.sum(power_spectrum_prob_dist * np.log2(power_spectrum_prob_dist + np.finfo(float).eps))
        entropy_values.append(entropy)

    entropy_values = np.array(entropy_values)

    # Apply smoothing if enabled
    if smooth:
        entropy_values = moving_average(entropy_values, window_size=smooth_window)

    # Generate time axis for entropy values
    if smooth:
        time_axis = np.linspace(0, audio_duration, len(entropy_values))
    else:
        time_axis = np.arange(num_frames) * frame_step_ms / 1000

    # Plot entropy over time
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(time_axis, entropy_values, label="Frequency Shannon Entropy")
    ax.set_xlim(0, audio_duration)
    ax.set_xticks(np.arange(0, audio_duration + 1, 1))  # Set x-ticks based on actual duration
    # ax.set_title(f"UID: {name}, Frequency Shannon Entropy Over Time")
    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_ylabel("Entropy", fontsize=16)
    ax.grid(axis='x')
    ax.legend(loc='upper right')

    return entropy_values
