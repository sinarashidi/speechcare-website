import os
import shap
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import torch.nn.functional as F
from transformers import (AutoModel, 
                          AutoTokenizer, 
                          AutoModelForSpeechSeq2Seq, 
                          AutoProcessor, pipeline)
from .shap_visualization import text


class Config():

    HUBERT = 'facebook/hubert-base-ls960'
    WAV2VEC2 = 'facebook/wav2vec2-base-960h'
    mHuBERT = 'utter-project/mHuBERT-147'
    MGTEBASE = 'Alibaba-NLP/gte-multilingual-base'
    WHISPER = "openai/whisper-large-v3-turbo"

    def __init__(self):
        return

    def get_subnet_insize(self):
        if self.transformer_checkpoint == self.HUBERT:
            return 768
        elif self.transformer_checkpoint == self.WAV2VEC2:
            return 768


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


class GatingNetwork(nn.Module):
    def __init__(self, input_dim):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, 3)  # Output 3 weights for speech, text, and demography
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        gate_weights = self.fc(x)
        return self.softmax(gate_weights)  # Ensure weights sum to 1


class TBNet(nn.Module):
    def __init__(self, config):
        super(TBNet, self).__init__()
        # set_seed(config.seed)
        self.predicted_label = None
        self.transcription = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.txt_transformer_chp, trust_remote_code=True)
        self.speech_transformer = AutoModel.from_pretrained(config.speech_transformer_chp)
        self.txt_transformer = AutoModel.from_pretrained(config.txt_transformer_chp, trust_remote_code=True)
        speech_embedding_dim = self.speech_transformer.config.hidden_size
        txt_embedding_dim = self.txt_transformer.config.hidden_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, speech_embedding_dim))
        max_seq_length = int(config.max_num_segments * ((config.segment_size / 0.02) - 1)) + 1 # +1 for CLS embedding
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, speech_embedding_dim))
        num_layers = 2
        self.layers = nn.ModuleList([
            MultiHeadAttentionAddNorm(speech_embedding_dim, 4, 0.1)
            for _ in range(num_layers)
        ])
        self.speech_head = nn.Sequential(
            nn.Linear(speech_embedding_dim, config.hidden_size),
            nn.Tanh(),
        )
        self.txt_head = nn.Sequential(
            nn.Linear(txt_embedding_dim, config.hidden_size),
            nn.Tanh(),
        )
        self.demography_head = nn.Sequential(
            nn.Linear(1, config.demography_hidden_size),
            nn.Tanh(),
        )
        self.speech_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.txt_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.demography_classifier = nn.Linear(config.demography_hidden_size, config.num_labels)
        self.weight_gate = GatingNetwork((config.hidden_size * 2) + config.demography_hidden_size)

        # Initialize Whisper pipeline
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = config.WHISPER
        whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        whisper_model.to(self.device)
        processor = AutoProcessor.from_pretrained(model_id)
        self.whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=whisper_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )
        
        self.labels = ['control', 'mci', 'adrd']
        self.label_map = {'control':0, 'mci':1, 'adrd':2}
        self.label_rev_map = {0:'control', 1:'mci', 2:'adrd'}
        self.text_explainer = shap.Explainer(self.calculate_shap_values, self.tokenizer, output_names=self.labels, hierarchical_values=True)

    def calculate_num_segments(self, audio_duration, segment_length, overlap, min_acceptable):
        """
        Calculate the maximum number of segments for a given audio duration.

        Args:
            audio_duration (float): Total duration of the audio in seconds.
            segment_length (float): Length of each segment in seconds.
            overlap (float): Overlap between consecutive segments as a fraction of segment length.
            min_acceptable (float): Minimum length of the remaining part to be considered a segment.

        Returns:
            int: Maximum number of segments.
        """
        overlap_samples = segment_length * overlap
        step_samples = segment_length - overlap_samples
        num_segments = int((audio_duration - segment_length) // step_samples + 1)
        remaining_audio = (audio_duration - segment_length) - (step_samples * (num_segments - 1))
        if remaining_audio >= min_acceptable:
            num_segments += 1
        return num_segments

    def preprocess_audio(self, audio_path, segment_length=5, overlap=0.2, target_sr=16000):
        """
        Preprocess a single audio file into segments.

        Args:
            audio_path (str): Path to the input audio file.
            segment_length (int): Length of each segment in seconds.
            overlap (float): Overlap between consecutive segments as a fraction of segment length.
            target_sr (int): Target sampling rate for resampling.

        Returns:
            torch.Tensor: Tensor containing the segmented audio data.
        """
        # Load and resample audio
        audio, sr = torchaudio.load(audio_path)
        resampler = transforms.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)

        # Convert to mono (average across channels)
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0)  # Average across channels
        else:
            audio = audio.squeeze(0)

        # Calculate segment parameters
        segment_samples = int(segment_length * target_sr)
        overlap_samples = int(segment_samples * overlap)
        step_samples = segment_samples - overlap_samples
        num_segments = self.calculate_num_segments(len(audio) / target_sr, segment_length, overlap, 5)
        segments = []
        end_sample = 0

        # Create segments
        for i in range(num_segments):
            start_sample = i * step_samples
            end_sample = start_sample + segment_samples
            segment = audio[start_sample:end_sample]
            segments.append(segment)

        # Handle remaining part
        remaining_part = audio[end_sample:]
        if len(remaining_part) >= 5 * target_sr:
            segments.append(remaining_part)

        # Stack segments into a tensor
        waveform = torch.stack(segments)  # Shape: [num_segments, seq_length]
        return waveform.unsqueeze(0)  # Add batch dimension: [1, num_segments, seq_length]

    def forward(self, input_values, input_ids=None, demography=None, attention_mask=None, speech_shap=False, return_embeddings=False):
        """
        Forward pass of the TBNet model.
        Args:
            input_values (torch.Tensor): Audio embeddings of shape [batch_size, num_segments, seq_length].
            input_ids (torch.Tensor, optional): Tokenized text input of shape [batch_size, max_seq_length].
            demography (torch.Tensor, optional): Demographic information of shape [batch_size, 1].
            attention_mask (torch.Tensor, optional): Attention mask for the text input of shape [batch_size, max_seq_length].
            speech_shap (bool): Whether to use the SHAP-specific forward logic for speech.
            return_embeddings (bool): Whether to return embeddings along with logits.
        Returns:
            tuple: A tuple containing:
                logits (torch.Tensor): Output logits of shape [batch_size, num_labels].
                probabilities (torch.Tensor): Probabilities for each class of shape [batch_size, num_labels].
                embeddings (torch.Tensor, optional): Embeddings if `return_embeddings` is True.
        """
        if not speech_shap:
            # Step 1: Reshape input_values to process each segment independently
            batch_size, num_segments, seq_length = input_values.size()
            input_values = input_values.view(batch_size * num_segments, seq_length)
            
            # Step 2: Pass through the speech transformer
            speech_embeddings = self.speech_transformer(input_values).last_hidden_state
            
            # Step 3: Process the text modality
            txt_embeddings = self.txt_transformer(input_ids=input_ids, attention_mask=attention_mask)
            
            # Step 4: Reshape speech embeddings back to [batch_size, num_segments, num_embeddings, dim]
            speech_embeddings = speech_embeddings.view(batch_size, num_segments, -1, speech_embeddings.size(-1))
            
            # Step 5: Flatten num_segments and num_embeddings to [batch_size, num_segments * num_embeddings, dim]
            speech_embeddings = speech_embeddings.view(batch_size, num_segments * speech_embeddings.size(2), -1)
            
            # Step 6: Prepend a trainable CLS token to the speech embeddings
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embedding_dim)
            speech_embeddings = torch.cat((cls_tokens, speech_embeddings), dim=1)  # Shape: (batch_size, seq_len+1, embedding_dim)
            
            # Step 7: Add positional encodings
            speech_embeddings += self.positional_encoding[:, :speech_embeddings.size(1), :]  # Match the sequence length
            
            # Step 8: Pass through MultiHead Attention layers
            for layer in self.layers:
                speech_embeddings = layer(speech_embeddings)
            
            # Step 9: Extract the CLS embedding vector for speech and text modalities
            speech_cls = speech_embeddings[:, 0, :]
            txt_cls = txt_embeddings.last_hidden_state[:, 0, :]
            
            # Step 10: Process the demographic modality
            demography = demography.unsqueeze(1)  # Ensure demography has shape [batch_size, 1]
            demography_x = self.demography_head(demography.squeeze(-1))  # Squeeze to [batch_size] before processing
            
            # Step 11: Project modalities into a shared space
            speech_x = self.speech_head(speech_cls)  # Shape: [batch_size, hidden_size]
            txt_x = self.txt_head(txt_cls)  # Shape: [batch_size, hidden_size]
            
            # Ensure demography_x has the same shape as speech_x and txt_x
            demography_x = demography_x.unsqueeze(1)  # Shape: [batch_size, 1, demography_hidden_size]
            demography_x = demography_x.squeeze(1)  # Shape: [batch_size, demography_hidden_size]
            
            # Step 12: Compute gating weights for modality fusion
            gate_weights = self.weight_gate(torch.cat([speech_x, txt_x, demography_x], dim=1))
            weight_speech, weight_txt, weight_demography = gate_weights[:, 0], gate_weights[:, 1], gate_weights[:, 2]
            
            # Step 13: Apply classifiers to each modality
            speech_out = self.speech_classifier(speech_x)
            txt_out = self.txt_classifier(txt_x)
            demography_out = self.demography_classifier(demography_x)
            
            # Step 14: Combine outputs using gated fusion
            fused_output = (
                weight_speech.unsqueeze(1) * speech_out +
                weight_txt.unsqueeze(1) * txt_out +
                weight_demography.unsqueeze(1) * demography_out
            )
            
            # Step 15: Compute probabilities using softmax
            probabilities = F.softmax(fused_output, dim=1)  # Convert logits to probabilities
            
            if return_embeddings:
                return fused_output, probabilities, speech_embeddings
            return fused_output, probabilities
    
        else:
            # Step 1: Ensure input tensor has the correct shape [batch_size, num_segments, seq_length]
            if input_values.dim() == 2:
                batch_size = 1
                num_segments, seq_length = input_values.size()
                input_values = input_values.view(batch_size, num_segments, seq_length)
            
            batch_size, num_segments, seq_length = input_values.size()
            input_values = input_values.view(batch_size * num_segments, seq_length)
            
            # Step 2: Pass through the speech transformer
            transformer_output = self.speech_transformer(input_values)
            output_embeddings = transformer_output.last_hidden_state
            
            # Step 3: Reshape output embeddings
            output_embeddings = output_embeddings.view(batch_size, num_segments, -1, output_embeddings.size(-1))
            output_embeddings = output_embeddings.view(batch_size, num_segments * output_embeddings.size(2), -1)
            
            # Step 4: Prepend a trainable CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, output_embeddings), dim=1)
            
            # Step 5: Add positional encodings
            embeddings += self.positional_encoding[:, :embeddings.size(1), :]
            
            # Step 6: Pass through MultiHead Attention layers
            for layer in self.layers:
                embeddings = layer(embeddings)
            
            # Step 7: Extract the CLS embedding vector
            cls = embeddings[:, 0, :]
            
            # Step 8: Project the CLS embedding
            x = self.speech_head(cls)
            x = F.tanh(x)
            x = self.speech_classifier(x)
            
            if return_embeddings:
                return x, transformer_output.last_hidden_state
            return x

    def inference(self, audio_path, demography_info, config):
        """
        Perform inference on a single audio file.

        Args:
            audio_path (str): Path to the input audio file.
            demography_info (float): Demographic information (e.g., age or other scalar value).
            config: Configuration object containing model-specific parameters.

        Returns:
            tuple: A tuple containing:
                - predicted_label (int): Predicted label (0 for healthy, 1 for MCI, 2 for ADRD).
                - probabilities (torch.Tensor): Probabilities for each class.
        """
        # Step 1: Transcribe the audio file using Whisper
        print("Transcribing audio...")
        transcription_result = self.whisper_pipeline(audio_path)
        self.transcription = transcription_result["text"]
        print(f"Transcription: {self.transcription}")

        # Step 2: Preprocess the audio file into segments
        print("Preprocessing audio...")
        waveform = self.preprocess_audio(audio_path, segment_length=config.segment_size, overlap=0.2)

        # Step 3: Tokenize the transcription
        print("Tokenizing transcription...")
        # tokenizer = AutoTokenizer.from_pretrained(config.txt_transformer_chp, trust_remote_code=True)
        tokenized_text = self.tokenizer(self.transcription, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenized_text["input_ids"]  # Shape: [1, max_seq_length]
        attention_mask = tokenized_text["attention_mask"]  # Shape: [1, max_seq_length]

        # Step 4: Prepare demographic information
        demography_tensor = torch.tensor([demography_info], dtype=torch.float32).unsqueeze(0)  # Shape: [1, 1]

        # Step 5: Move all inputs to the appropriate device
        device = next(self.parameters()).device
        waveform = waveform.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        demography_tensor = demography_tensor.to(device)

        # Step 6: Run the model
        print("Running inference...")
        with torch.no_grad():
            logits, probabilities = self(waveform, input_ids, demography_tensor, attention_mask)

        # Step 7: Get the predicted label
        predicted_label = torch.argmax(logits, dim=1).item()
        self.predicted_label = predicted_label

        return predicted_label, probabilities[0].tolist()
    
    
    def text_only_classification(self, input_ids, attention_mask):
        print('text only classifier in...')
        txt_embeddings = self.txt_transformer(input_ids=input_ids, attention_mask=attention_mask)
        print('transformer - done')
        txt_cls = txt_embeddings.last_hidden_state[:, 0, :]
        print('cls - done')
        txt_x = self.txt_head(txt_cls)  
        print('head - done')
        txt_out = self.txt_classifier(txt_x)
        print('classifier - done')
        print('text only classifier out...')
        return txt_out
    
    def calculate_shap_values(self, text):
        device = next(self.parameters()).device
        # Tokenize and encode the input
        input_ids = torch.tensor([self.tokenizer.encode(v, padding="max_length", max_length=300, truncation=True) for v in text]).to(device)
        attention_masks = (input_ids != 0).type(torch.int64).to(device)
        # Pass through the model
        # outputs = self.text_only_classification(input_ids, attention_masks).detach().cpu().numpy()
        txt_embeddings = self.txt_transformer(input_ids=input_ids, attention_mask=attention_masks)
        txt_cls = txt_embeddings.last_hidden_state[:, 0, :]
        txt_x = self.txt_head(txt_cls)  
        txt_out = self.txt_classifier(txt_x)
        outputs = txt_out.detach().cpu().numpy()

        # Apply softmax to get probabilities
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T

        # Define a helper function to calculate logit with special handling
        def safe_logit(p):
            with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for divide by zero or invalid ops
                logit = np.log(p / (1 - p))
                logit[p == 0] = -np.inf  # logit(0) = -inf
                logit[p == 1] = np.inf   # logit(1) = inf
                logit[(p < 0) | (p > 1)] = np.nan  # logit(p) is nan for p < 0 or p > 1
            return logit

        # Calculate the new scores based on the specified criteria
        p_0, p_1, p_2 = scores[:, 0], scores[:, 1], scores[:, 2]

        score_0 = safe_logit(p_0)
        p_1_p_2_sum = p_1 + p_2
        score_1 = safe_logit(p_1_p_2_sum)
        score_2 = score_1  # Same as score_1 per your criteria

        # Combine the scores into a single array
        new_scores = np.stack([score_0, score_1, score_2], axis=-1)

        return new_scores
    
    def illustrate_shap_values(self):
        print('Running shap values...')
        input_text = [str(self.transcription)]
        print('input text:',input_text)
        
        shap_values = self.text_explainer(input_text)
        print('Values explained...')
        shap_html_code = text(shap_values[:,:,self.predicted_label], display=False)
        return shap_html_code

    def speech_only_classification(self, input_values, return_embeddings=False):
        """
        Perform classification using only the speech modality.
        
        Args:
            input_values (torch.Tensor): Audio embeddings of shape [batch_size, num_segments, seq_length].
            return_embeddings (bool): If True, returns the transformer's last hidden state along with logits.
            
        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_labels].
            torch.Tensor (optional): Last hidden state from the transformer.
        """
        print('speech only classifier in...')
    
        # Ensure the input tensor has the correct shape: [batch_size, num_segments, seq_length]
        if input_values.dim() == 2:
            batch_size = 1
            num_segments, seq_length = input_values.size()
            input_values = input_values.view(batch_size, num_segments, seq_length)
    
        batch_size, num_segments, seq_length = input_values.size()
        input_values = input_values.view(batch_size * num_segments, seq_length)
    
        # Pass through the speech transformer
        transformer_output = self.speech_transformer(input_values)
        output_embeddings = transformer_output.last_hidden_state
    
        # Reshape embeddings back to [batch_size, num_segments, num_embeddings, dim]
        output_embeddings = output_embeddings.view(batch_size, num_segments, -1, output_embeddings.size(-1))
    
        # Flatten num_segments and num_embeddings to [batch_size, num_segments * num_embeddings, dim]
        output_embeddings = output_embeddings.view(batch_size, num_segments * output_embeddings.size(2), -1)
    
        # Prepend a trainable CLS token to the speech embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embedding_dim)
        embeddings = torch.cat((cls_tokens, output_embeddings), dim=1)  # Shape: (batch_size, seq_len+1, embedding_dim)
    
        # Add positional encodings
        embeddings += self.positional_encoding[:, :embeddings.size(1), :]  # Match the sequence length
    
        # Pass through MultiHead Attention layers
        for layer in self.layers:
            embeddings = layer(embeddings)
    
        # Extract the CLS embedding vector
        cls = embeddings[:, 0, :]
    
        # Project the CLS embedding into the hidden space
        x = self.speech_head(cls)  # Shape: [batch_size, hidden_size]
    
        # Apply the classifier
        speech_out = self.speech_classifier(x)
    
        # Optionally return embeddings
        if return_embeddings:
            return speech_out, transformer_output.last_hidden_state
    
        print('speech only classifier out...')
        return speech_out

    def speech_shap_inference(self, audio_path, segment_length=5, overlap=0.2, target_sr=16000, device='cuda'):
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
        segment_samples = int(segment_length * target_sr)
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
        segments_tensor = torch.stack(segments)  # Shape: [num_segments, seq_length]
    
        # Add batch dimension
        input_values = segments_tensor.unsqueeze(0).to(device)  # Shape: [1, num_segments, seq_length]
    
        with torch.no_grad():
            predictions, embeddings = self.speech_only_classification(input_values, return_embeddings=True)
    
        return {
            "predictions": predictions.cpu().numpy(),
            "embeddings": embeddings.cpu().numpy(),
            "segments_tensor": segments_tensor.cpu().numpy()
        }

    def calculate_and_visualize_speech_shap(
    self,
    audio_path,
    fig_save_path,
    segment_length=5,
    overlap=0.2,
    target_sr=16000,
    baseline_type='zeros',
    frame_length_ms=25,
    frame_step_ms=10,
    windowing_function="hamming",
    smooth=True,
    smooth_window=50,
    formants_to_plot=["F0", "F3"],
    merge_frame_duration=0.3
    ):
        """
        Calculate SHAP values for an audio file, visualize the spectrogram with SHAP-adjusted intensity,
        and plot the frequency Shannon entropy over time.
        """
    
        # Step 1: Perform speech SHAP inference
        result = self.speech_shap_inference(
            audio_path,
            segment_length=segment_length,
            overlap=overlap,
            target_sr=target_sr,
            device=self.device
        )
        segments_tensor = torch.tensor(result["segments_tensor"]).to(self.device)
        predictions = result["predictions"]
    
        # Step 2: Prepare baseline data
        if baseline_type == 'zeros':
            baseline_data = torch.zeros_like(segments_tensor)  # Zero baseline
        elif baseline_type == 'mean':
            baseline_data = torch.mean(segments_tensor, dim=0, keepdim=True).repeat(
                segments_tensor.size(0), 1, 1
            )  # Mean baseline
    
        # Ensure baseline_data has the correct shape
        baseline_data = baseline_data.unsqueeze(0) if baseline_data.dim() == 2 else baseline_data
        segments_tensor = segments_tensor.unsqueeze(0) if segments_tensor.dim() == 2 else segments_tensor
    
        # Step 3: Wrap the PyTorch model for compatibility with SHAP
        def model_wrapper(input_data):
            self.to(input_data.device)
            with torch.no_grad():
                predictions, _ = self.speech_only_classification(input_data, return_embeddings=True)
                return predictions.cpu().numpy()
    
        # Step 4: Initialize SHAP DeepExplainer
        explainer = shap.DeepExplainer(model_wrapper, baseline_data)
    
        # Step 5: Compute SHAP values
        shap_values = explainer.shap_values(segments_tensor, check_additivity=False)  # Disable additivity check
        shap_values_aggregated = [shap_val.sum(axis=-1) for shap_val in shap_values]
    
        # Step 6: Visualize the spectrogram with SHAP-adjusted intensity
        def visualize_shap_spectrogram(shap_values, label):
            name = os.path.splitext(os.path.basename(audio_path))[0]
            audio, _ = librosa.load(audio_path, sr=target_sr)
            S = librosa.feature.melspectrogram(y=audio, sr=target_sr, n_fft=2048, hop_length=512, power=2.0)
            log_S = librosa.power_to_db(S, ref=np.max)
    
            shap_values_label = shap_values[0, :, :, label]
            segment_samples = int(segment_length * target_sr)
            hop_samples = int(segment_samples * (1 - overlap))
            merge_samples = int(merge_frame_duration * target_sr)
            merged_shap_values = []
    
            for segment in shap_values_label:
                reshaped_segment = segment[: len(segment) // merge_samples * merge_samples]
                reshaped_segment = reshaped_segment.reshape(-1, merge_samples)
                merged_shap_values.append(reshaped_segment.mean(axis=1))
    
            merged_shap_values = np.concatenate(merged_shap_values)
            merged_shap_values_normalized = (merged_shap_values - np.percentile(merged_shap_values, 5)) / (
                np.percentile(merged_shap_values, 95) - np.percentile(merged_shap_values, 5)
            )
            merged_shap_values_normalized = np.clip(merged_shap_values_normalized, 0, 1)
            merged_shap_values_transformed = merged_shap_values_normalized**5
            merged_shap_values_transformed *= 10
            merged_shap_values_transformed += 0.01
    
            audio_duration = len(audio) / target_sr
            merged_frame_times = np.arange(0, len(merged_shap_values)) * merge_frame_duration
            time_bins = np.linspace(0, audio_duration, S.shape[1])
    
            for i, t in enumerate(merged_frame_times):
                idx_start = np.searchsorted(time_bins, t)
                idx_end = np.searchsorted(time_bins, t + merge_frame_duration)
                if idx_start < len(S[0]) and idx_end < len(S[0]):
                    S[:, idx_start:idx_end] *= merged_shap_values_transformed[i]
    
            modified_log_S = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots(figsize=(20, 4))
            img = librosa.display.specshow(modified_log_S, sr=target_sr, x_axis="time", y_axis="mel", cmap="viridis", ax=ax)
            ax.set_xlabel("")
            ax.set_ylabel("Frequency (Hz)", fontsize=16)
            ax.set_xlim(0, audio_duration)
            ax.set_xticks([])
    
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
    
                formant_colors = {"F0": 'red', "F1": 'cyan', "F2": 'white', "F3": 'orange'}
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
    
            return fig
    
        # Step 7: Plot frequency Shannon entropy
        def frequency_shannon_entropy():
            signal, sr = librosa.load(audio_path, sr=None)
            audio_duration = len(signal) / sr
            frame_length_samples = int(frame_length_ms * sr / 1000)
            frame_step_samples = int(frame_step_ms * sr / 1000)
    
            if windowing_function == "hamming":
                window = np.hamming(frame_length_samples)
            else:
                raise ValueError("Unsupported windowing function")
    
            num_frames = max(1, 1 + (len(signal) - frame_length_samples) // frame_step_samples)
            entropy_values = []
    
            for i in range(num_frames):
                start_idx = i * frame_step_samples
                end_idx = start_idx + frame_length_samples
                if end_idx > len(signal):
                    break
                frame = signal[start_idx:end_idx] * window
                frequencies, power_spectrum = welch(frame, fs=sr, nperseg=frame_length_samples)
                power_spectrum_prob_dist = power_spectrum / (np.sum(power_spectrum) + np.finfo(float).eps)
                entropy = -np.sum(power_spectrum_prob_dist * np.log2(power_spectrum_prob_dist + np.finfo(float).eps))
                entropy_values.append(entropy)
    
            entropy_values = np.array(entropy_values)
            if smooth:
                entropy_values = self.moving_average(entropy_values, window_size=smooth_window)
    
            time_axis = np.linspace(0, audio_duration, len(entropy_values))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_axis, entropy_values, label="Frequency Shannon Entropy")
            ax.set_xlim(0, audio_duration)
            ax.set_xticks(np.arange(0, audio_duration + 1, 1))
            ax.set_xlabel("Time (s)", fontsize=16)
            ax.set_ylabel("Entropy", fontsize=16)
            ax.grid(axis='x')
            ax.legend(loc='upper right')
            return fig
    
        # Step 8: Create the combined figure
        fig = plt.figure(figsize=(20, 5.5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.5])
    
        # Spectrogram subplot
        ax0 = plt.subplot(gs[0])
        spectrogram_fig = visualize_shap_spectrogram(shap_values, label=self.predicted_label)
        ax0 = spectrogram_fig.axes[0]
    
        # Frequency Shannon Entropy subplot
        ax1 = plt.subplot(gs[1])
        entropy_fig = frequency_shannon_entropy()
        ax1 = entropy_fig.axes[0]
    
        plt.tight_layout()
    
        # Save the figure
        os.makedirs(os.path.dirname(fig_save_path), exist_ok=True)
        plt.savefig(fig_save_path, dpi=600, bbox_inches="tight", transparent=True)
        plt.close()

    def moving_average(self, data, window_size=5):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')