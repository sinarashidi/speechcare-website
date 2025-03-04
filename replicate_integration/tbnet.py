import os
import json
import requests
import shap
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from transformers import (AutoModel, 
                          AutoTokenizer, 
                          AutoModelForSpeechSeq2Seq, 
                          AutoProcessor, pipeline)
from text_shap_visualization import text, unpack_shap_explanation_contents, process_shap_values
from speech_shap_visualization import visualize_shap_spectrogram, frequency_shannon_entropy


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
        self.text_explainer = shap.Explainer(self.calculate_text_shap_values, self.tokenizer, output_names=self.labels, hierarchical_values=True)

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

    def forward(self, input_values, input_ids, demography, attention_mask):
        """
        Forward pass of the TBNet model.

        Args:
            input_values (torch.Tensor): Audio embeddings of shape [batch_size, num_segments, seq_length].
            input_ids (torch.Tensor): Tokenized text input of shape [batch_size, max_seq_length].
            demography (torch.Tensor): Demographic information of shape [batch_size, 1].
            attention_mask (torch.Tensor): Attention mask for the text input of shape [batch_size, max_seq_length].

        Returns:
            tuple: A tuple containing:
                - logits (torch.Tensor): Output logits of shape [batch_size, num_labels].
                - probabilities (torch.Tensor): Probabilities for each class of shape [batch_size, num_labels].
        """
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
        txt_x = self.txt_head(txt_cls)          # Shape: [batch_size, hidden_size]

        # Ensure demography_x has the same shape as speech_x and txt_x
        demography_x = demography_x.unsqueeze(1)  # Shape: [batch_size, 1, demography_hidden_size]
        demography_x = demography_x.squeeze(1)   # Shape: [batch_size, demography_hidden_size]

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

        return fused_output, probabilities

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
        audio_path = str(audio_path)
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
    
    def calculate_text_shap_values(self, text):
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
    
    def get_text_shap_results(self):
        print('Running shap values...')
        input_text = [str(self.transcription)]
        print('input text:',input_text)
        
        shap_values = self.text_explainer(input_text)
        print('Values explained...')
        shap_html_code = text(shap_values[:,:,self.predicted_label], display=False)
        return shap_html_code

    def get_text_shap_dict(self, grouping_threshold=0.01, separator=" "):
        """
        Returns a dictionary of token -> SHAP value for the model's current transcription
        and predicted label.
    
        Parameters
        ----------
        grouping_threshold : float
            Merges tokens into a single 'group' if the interaction level is high compared
            to this threshold. A value near 0.01 is common for minimal grouping.
        separator : str
            Used to join subwords if they are merged due to hierarchical grouping. Typically
            ' ' for normal words, or '' if using GPT-style 'Ġ' tokens.
    
        Returns
        -------
        dict
            A dictionary mapping each (possibly merged) token to its SHAP value.
        """
    
        # If there's no transcription yet, run inference or make sure self.transcription is set
        if not self.transcription:
            raise ValueError("self.transcription is empty. Run inference(...) first or set it manually.")
    
        # Get SHAP values for the text (this uses calculate_text_shap_values under the hood)
        input_text = [str(self.transcription)]
        shap_explanations = self.text_explainer(input_text)  
        # shap_explanations typically has shape [batch, tokens, classes].
        # We'll pick the first example (index 0), then the predicted_label dimension.
        # i.e. shap_explanations[0, :, self.predicted_label].
    
        # Make sure we handle the predicted label dimension safely
        if self.predicted_label is None:
            raise ValueError("self.predicted_label is None. Inference may not have been run.")
    
        # The snippet below extracts the single (row, class) explanation:
        single_shap_values = shap_explanations[0, :, self.predicted_label]
        # 'single_shap_values' is now a shap.Explanation object representing the token-level SHAP
        # for whichever label is in self.predicted_label.
    
        # Decompose the hierarchical SHAP values into tokens/values
        values, clustering = unpack_shap_explanation_contents(single_shap_values)
        tokens, merged_values, group_sizes = process_shap_values(
            single_shap_values.data,
            values,
            grouping_threshold=grouping_threshold,
            separator=separator,
            clustering=clustering
        )
    
        # Build the dictionary: token -> SHAP value
        shap_dict = {}
        for token, val in zip(tokens, merged_values):
            # It’s good practice to cast the SHAP value to a plain float
            shap_dict[token] = float(val)
    
        return shap_dict, shap_explanations
    
    def get_llama_interpretation(self, llama_api_key):
        shap_dict, shap_values = self.get_text_shap_dict()
        
        message = f"""
            You are a specialized language model trained to detect linguistic cues of cognitive impairment. You will receive:
            1) A set of linguistic features to consider.
            2) A text passage to analyze.
            3) Token-level SHAP values from a pre-trained model.

            Your task is to:
            A. Identify which tokens are most influential in the classification, based on SHAP values.
            B. Map each influential token to one or more linguistic features (e.g., lexical richness, syntactic complexity).
            C. Explain how the token and its context may indicate healthy cognition or cognitive impairment.

            Please follow the steps below and provide a structured analysis.
            ---
            Linguistic Features to Consider:
            • **Lexical Richness**: Unusual or varied vocabulary, overuse of vague terms (e.g., “thing,” “stuff”).
            • **Syntactic Complexity**: Simple vs. complex sentence constructions, grammatical errors.
            • **Sentence Length and Structure**: Fragmented vs. compound/complex sentences.
            • **Repetition**: Repeated words, phrases, or clauses.
            • **Disfluencies and Fillers**: Terms like “um,” “uh,” “like.”
            • **Semantic Coherence and Content**: Logical flow of ideas, clarity of meaning.
            • **Additional Feature (XXX)**: Placeholder for any extra marker (e.g., specialized domain terms).
            ---
            Text to Analyze:
            {self.transcription}
            ---
            Token-level SHAP Values:
            {shap_dict}
            ---
            Analysis Format:
            1) **Token-Level Analysis**: For each token with a significant |SHAP| value, specify:
            - Token
            - SHAP Value
            - Linguistic Feature(s) Involved
            - Brief Interpretation
            2) **Overall Summary**: Synthesize the significance of these tokens/features to explain how they collectively point to healthy cognition or potential cognitive impairment.
            ---
            """
        output_format = """
            Example Output Format:
            {
            "Analysis": [
                {
                "Token": "um",
                "SHAP_Value": -0.87,
                "Linguistic_Feature": "Disfluency",
                "Interpretation": "Filler word commonly seen in cognitive impairment contexts."
                },
                {
                "Token": "patient",
                "SHAP_Value": 0.65,
                "Linguistic_Feature": "Lexical Richness",
                "Interpretation": "Use of domain-specific term suggests context awareness."
                },
                ...
            ],
            [sep_token]
            "Overall_Summary": "Multiple disfluencies and repetitive fragments are indicative of possible cognitive impairment."
            [sep_token]
            }
            ---
            Constraints and Guidelines:
            - Rely only on the provided text and SHAP values; do not infer from external or hidden knowledge.
            - Tie each token with high |SHAP| back to a specific linguistic feature and explain its clinical relevance.
            - Put separator token [sep_token] before and after the 'Overall_Summary' in the output.
            """

        message += output_format

        print(f"Getting LLaMA interpretation with api key: {llama_api_key}")
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {llama_api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "meta-llama/llama-3.3-70b-instruct:free",
                "messages": [
                {
                    "role": "user",
                    "content": message
                }
                ],
            })
            )
        print(response.json())
        shap_html_code = text(shap_values[:,:,self.predicted_label], display=False)
        return shap_html_code, response.json()['choices'][0]['message']['content']

    def speech_only_forward(self, input_values, return_embeddings=False):
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
        transformer_output = self.speech_transformer(input_values)
        output_embeddings = transformer_output.last_hidden_state

        output_embeddings = output_embeddings.view(batch_size, num_segments, -1, output_embeddings.size(-1))
        output_embeddings = output_embeddings.view(batch_size, num_segments * output_embeddings.size(2), -1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, output_embeddings), dim=1)
        embeddings += self.positional_encoding[:, :embeddings.size(1), :]

        for layer in self.layers:
            embeddings = layer(embeddings)

        cls = embeddings[:, 0, :]
        x = self.speech_head(cls)
        x = self.speech_classifier(x)
        
        if return_embeddings:
            return x, transformer_output.last_hidden_state
        return x
    
    def speech_only_inference(self, audio_path, segment_length=5, overlap=0.2, target_sr=16000, device='cuda'):
        """
        Inference method for the TBNet model. Processes an audio file, splits it, and returns predictions and embeddings.
        """
        self.eval()
        self.to(self.device)

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
            predictions, embeddings = self.speech_only_forward(input_values, return_embeddings=True)

        return {
            "predictions": predictions.cpu().numpy(),
            "embeddings": embeddings.cpu().numpy(),
            "segments_tensor": segments_tensor.cpu().numpy()
        }
        
    def calculate_speech_shap_values(
        self,
        audio_path,
        segment_length=5,
        overlap=0.2,
        target_sr=16000,
        baseline_type='zeros'
    ):
        result = self.speech_only_inference(
            audio_path,
            segment_length=segment_length,
            overlap=overlap,
            target_sr=target_sr,
            device=self.device
        )

        segments_tensor = torch.tensor(result["segments_tensor"]).to(self.device)  # Input tensor for SHAP
        predictions = result["predictions"]

        if baseline_type == 'zeros':
            baseline_data = torch.zeros_like(segments_tensor)  # Zero baseline
        elif baseline_type == 'mean':
            baseline_data = torch.mean(segments_tensor, dim=0, keepdim=True).repeat(
                segments_tensor.size(0), 1, 1
            )  # Mean baseline

        baseline_data = baseline_data.unsqueeze(0) if baseline_data.dim() == 2 else baseline_data
        segments_tensor = segments_tensor.unsqueeze(0) if segments_tensor.dim() == 2 else segments_tensor

        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
        
            def forward(self, x):
                # Instead of calling self.model.forward(x)
                return self.model.speech_only_forward(x)
        
        explainer = shap.DeepExplainer(ModelWrapper(self), baseline_data)

        shap_values = explainer.shap_values(segments_tensor, check_additivity=False)  # Disable additivity check

        shap_values_aggregated = [shap_val.sum(axis=-1) for shap_val in shap_values]

        return {
            "shap_values": shap_values,
            "shap_values_aggregated": shap_values_aggregated,
            "segments_tensor": segments_tensor.cpu().numpy(),
            "predictions": predictions
        }

    def get_speech_shap_results(
        self,
        audio_path,
        demography_info,
        config,
        frame_duration=0.3,
        formants_to_plot=["F0", "F3"],
        segment_length=5,
        overlap=0.2,
        target_sr=16000,
        baseline_type='zeros'
    ):
        """
        Calculates SHAP values for the given audio file, creates a figure with a spectrogram
        and frequency Shannon entropy subplots, saves the figure to fig_save_path, and returns the figure.
        """
        audio_path = str(audio_path)
        audio_label = self.inference(audio_path, demography_info, config)[0]

        shap_results = self.calculate_speech_shap_values(
            audio_path,
            segment_length=segment_length,
            overlap=overlap,
            target_sr=target_sr,
            baseline_type=baseline_type,
        )
        shap_values = shap_results["shap_values"]
        shap_values_aggregated = shap_results["shap_values_aggregated"]
        predictions = shap_results["predictions"]

        # Create the figure and grid
        fig = plt.figure(figsize=(20, 5.5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1.5])

        # Spectrogram subplot
        ax0 = plt.subplot(gs[0])
        _ = visualize_shap_spectrogram(
            audio_path,
            shap_values,
            audio_label,
            sr=target_sr,
            segment_length=segment_length,
            overlap=overlap,
            merge_frame_duration=frame_duration,
            formants_to_plot=formants_to_plot,
            fig_save_path=None,
            ax=ax0
        )

        # Frequency Shannon Entropy subplot
        ax1 = plt.subplot(gs[1])
        _ = frequency_shannon_entropy(
            audio_path,
            ax=ax1,
            smooth_window=50
        )

        plt.tight_layout()
        # Ensure the directory exists and save the figure
        fig_save_path = f"speech_shap_{os.path.basename(audio_path)}.png"
        plt.savefig(fig_save_path, dpi=600, bbox_inches="tight", transparent=True)
        return fig_save_path
    
    def get_speech_spectrogram(self,
        audio_path,
        demography_info,
        config,
        frame_duration=0.3,
        formants_to_plot=["F0", "F3"],
        segment_length=5,
        overlap=0.2,
        target_sr=16000,
        baseline_type='zeros'
    ):
        """
        Calculates SHAP values for the given audio file, then calculates the modified 
        spectrogram and frequency shannon entropy and returns them.
        """
        audio_path = str(audio_path)
        audio_label = self.inference(audio_path, demography_info, config)[0]

        shap_results = self.calculate_speech_shap_values(
            audio_path,
            segment_length=segment_length,
            overlap=overlap,
            target_sr=target_sr,
            baseline_type=baseline_type,
        )
        shap_values = shap_results["shap_values"]

        modified_spectrogram = visualize_shap_spectrogram(
            audio_path,
            shap_values,
            audio_label,
            sr=target_sr,
            segment_length=segment_length,
            overlap=overlap,
            merge_frame_duration=frame_duration,
            formants_to_plot=formants_to_plot,
            fig_save_path=None,
        )

        freq_shann_ent = frequency_shannon_entropy(
            audio_path,
            smooth_window=50
        )

        return modified_spectrogram, freq_shann_ent