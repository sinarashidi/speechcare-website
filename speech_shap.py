# General
import random, itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

# System
from IPython.display import clear_output, Markdown, display
from tqdm import tqdm
import sys, os, fnmatch, time
import os.path

# Huggingface
from transformers import AutoModel, AutoTokenizer, AdamW
from transformers import Wav2Vec2FeatureExtractor

# Pytorch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchaudio
from torchaudio import transforms

import warnings
warnings.filterwarnings("ignore")

# Explainability
import shap
import librosa
import librosa.display
import parselmouth
from scipy.signal import welch


device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""# Model code"""

TRAIN, VALID, TEST = 0, 1, 2
MEAN, GRU, GRU2, GRU3, CNN, CNN2, SIMPLE_ATTENTION, COMPLEX_ATTENTION = 10, 11, 12, 13, 14, 15, 16, 17
FINE_TUNE_ENTIRE, FINE_TUNE_LAST = 20, 21

class Config():

    HUBERT = 'facebook/hubert-base-ls960'
    WAV2VEC2 = 'facebook/wav2vec2-base-960h'
    mHuBERT = 'utter-project/mHuBERT-147'


    def __init__(self, seed= None, lr= None, wd= None,
                 epochs= None, bs= None, transformer_chp= None,
                 integration = None, hidden_size= None, dropout= None,
                 active_layers = 12, num_labels=2):

        self.seed = seed
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.bs = bs
        self.transformer_chp = transformer_chp
        self.integration= integration
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.active_layers = active_layers
        self.num_labels = num_labels


    def get_subnet_insize(self):
        if self.transformer_checkpoint == self.HUBERT:
            return 768
        elif self.transformer_checkpoint == self.WAV2VEC2:
            return 768

config = Config()
config.seed = 133
config.bs = 4
config.epochs = 20
config.lr = 1e-5
config.hidden_size = 128
config.dropout = 0
config.wd = 1e-3
config.integration = SIMPLE_ATTENTION
config.num_labels = 3
config.transformer_chp = config.mHuBERT
config.segment_size = 5 # seconds
config.max_num_segments = 7


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class MultiHeadAttentionAddNorm(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):

        super(MultiHeadAttentionAddNorm, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Multi-Head Attention
        attn_output, _ = self.mha(x, x, x)  # Self-attention: Q = K = V = x

        # Add & Norm
        x = self.norm(x + self.dropout(attn_output))
        return x



class TBNet(nn.Module):
    def __init__(self, config):
        super(TBNet, self).__init__()
        set_seed(config.seed)

        self.transformer = AutoModel.from_pretrained(config.transformer_chp)
        embedding_dim = self.transformer.config.hidden_size

        # Trainable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        max_seq_length = int(config.max_num_segments * ((config.segment_size / 0.02) - 1)) + 1  # +1 for CLS embedding

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, embedding_dim))

        num_layers = 2
        self.layers = nn.ModuleList([
            MultiHeadAttentionAddNorm(embedding_dim, 1, 0.4)
            for _ in range(num_layers)
        ])

        self.projector = nn.Linear(embedding_dim, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_values, return_embeddings=False):
        """
        Forward method for TBNet model.
        Ensures that the input tensor has the correct shape: [batch_size, num_segments, seq_length].
        """
        if input_values.dim() == 2:
            # Reshape to [batch_size, num_segments, seq_length]
            batch_size = 1
            num_segments, seq_length = input_values.size()
            input_values = input_values.view(batch_size, num_segments, seq_length)

        batch_size, num_segments, seq_length = input_values.size()

        input_values = input_values.view(batch_size * num_segments, seq_length)
        transformer_output = self.transformer(input_values)
        output_embeddings = transformer_output.last_hidden_state

        output_embeddings = output_embeddings.view(batch_size, num_segments, -1, output_embeddings.size(-1))
        output_embeddings = output_embeddings.view(batch_size, num_segments * output_embeddings.size(2), -1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, output_embeddings), dim=1)
        embeddings += self.positional_encoding[:, :embeddings.size(1), :]

        for layer in self.layers:
            embeddings = layer(embeddings)

        cls = embeddings[:, 0, :]
        x = self.projector(cls)
        x = F.tanh(x)
        x = self.classifier(x)

        if return_embeddings:
            return x, transformer_output.last_hidden_state
        return x


    def inference(self, audio_path, segment_length=5, overlap=0.2, target_sr=16000, device='cuda'):
        """
        Inference method for the TBNet model. Processes an audio file, splits it, and returns predictions and embeddings.
        """
        self.eval()
        self.to(device)

        # Load and resample audio
        audio, sr = torchaudio.load(audio_path)
        resampler = transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)

        # Convert to mono
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0)
        else:
            audio = audio.squeeze(0)

        # Segment the audio
        segment_samples = segment_length * target_sr
        overlap_samples = int(segment_samples * overlap)
        step_samples = segment_samples - overlap_samples
        num_segments = (int(audio.size(0)) - segment_samples) // step_samples + 1

        segments = []
        end_sample = 0
        for i in range(num_segments):
            start_sample = i * step_samples
            end_sample = start_sample + segment_samples
            segments.append(audio[start_sample:end_sample])

        remaining_part = audio[end_sample:]
        if remaining_part.size(0) >= segment_length * target_sr:
            segments.append(remaining_part)

        # Stack segments into a tensor
        segments_tensor = torch.stack(segments)  # Shape: [num_segments, segment_length * sr]

        # Add batch dimension
        input_values = segments_tensor.unsqueeze(0).to(device)  # Shape: [1, num_segments, seq_length]

        with torch.no_grad():
            predictions, embeddings = self(input_values, return_embeddings=True)

        return {
            "predictions": predictions.cpu().numpy(),
            "embeddings": embeddings.cpu().numpy(),
            "segments_tensor": segments_tensor.cpu().numpy()
        }

"""# SHAP values Calculation"""

def calculate_shap_values(
    model,
    audio_path,
    segment_length=5,
    overlap=0.2,
    target_sr=16000,
    device='cuda',
    baseline_type='zeros'
):
    result = model.inference(
        audio_path,
        segment_length=segment_length,
        overlap=overlap,
        target_sr=target_sr,
        device=device
    )

    segments_tensor = torch.tensor(result["segments_tensor"]).to(device)  # Input tensor for SHAP
    predictions = result["predictions"]

    if baseline_type == 'zeros':
        baseline_data = torch.zeros_like(segments_tensor)  # Zero baseline
    elif baseline_type == 'mean':
        baseline_data = torch.mean(segments_tensor, dim=0, keepdim=True).repeat(
            segments_tensor.size(0), 1, 1
        )  # Mean baseline

    baseline_data = baseline_data.unsqueeze(0) if baseline_data.dim() == 2 else baseline_data
    segments_tensor = segments_tensor.unsqueeze(0) if segments_tensor.dim() == 2 else segments_tensor

    explainer = shap.DeepExplainer(model, baseline_data)

    shap_values = explainer.shap_values(segments_tensor, check_additivity=False)  # Disable additivity check

    shap_values_aggregated = [shap_val.sum(axis=-1) for shap_val in shap_values]

    return {
        "shap_values": shap_values,
        "shap_values_aggregated": shap_values_aggregated,
        "segments_tensor": segments_tensor.cpu().numpy(),
        "predictions": predictions
    }

"""# Plot Functions"""

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

"""# Run for subjects"""

MODEL_CHECKPOINT_PATH = "/content/drive/MyDrive/NIA-competition-Phase-2/Explainability/HuBERT_Checkpoint/model_best.pt"

# Changeable parameters
AUDIO_PATH = "/content/drive/MyDrive/NIA Competition/Datasets/LPF_test_audios/qnvo.wav"
name = os.path.splitext(os.path.basename(AUDIO_PATH))[0]
# Audio label
AUDIO_LABEL = 1  # 0: HEALTHY, 1: MCI, 2: ADRD
FRAME_DURATION = 0.3
FORMANTS_TO_PLOT = ["F0", "F3"]  # ["F0", "F1", "F2", "F3"]
FIG_SAVE_PATH = f"Explainability/{name}_{FRAME_DURATION}s_{'_'.join(FORMANTS_TO_PLOT)}_shannon.png"

# Load the trained TBNet model
model = TBNet(config)
model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH))
model.eval()

# Calculate SHAP values
shap_results = calculate_shap_values(
    model,
    AUDIO_PATH,
    segment_length=5,
    overlap=0.2,
    target_sr=16000,
    device='cuda',
    baseline_type='zeros'
)

# Access SHAP values and predictions
shap_values = shap_results["shap_values"]
shap_values_aggregated = shap_results["shap_values_aggregated"]
predictions = shap_results["predictions"]

fig = plt.figure(figsize=(20, 5.5))  # Adjust the overall figure size
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.5])  # Define grid spec with height ratios

# Spectrogram subplot
ax0 = plt.subplot(gs[0])
spectrogram = visualize_shap_spectrogram(
    AUDIO_PATH, shap_values, AUDIO_LABEL,
    sr=16000, segment_length=5, overlap=0.2,
    merge_frame_duration=FRAME_DURATION,
    formants_to_plot=FORMANTS_TO_PLOT,
    fig_save_path=None, ax=ax0
)

# Frequency Shannon Entropy subplot
ax1 = plt.subplot(gs[1])
entropy = frequency_shannon_entropy(
    AUDIO_PATH, ax=ax1, smooth_window=50
)

plt.tight_layout()
if FIG_SAVE_PATH:
    os.makedirs(os.path.dirname(FIG_SAVE_PATH), exist_ok=True)
    plt.savefig(FIG_SAVE_PATH, dpi=600, bbox_inches="tight", transparent=True)
plt.show()

import shutil

shutil.make_archive("Explainability_bigLabels", "zip", "Explainability")

