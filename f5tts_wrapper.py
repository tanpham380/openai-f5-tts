import os
import torch
import torchaudio
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from importlib.resources import files
from pydub import AudioSegment, silence

from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    convert_char_to_pinyin,
    list_str_to_idx,
)
from f5_tts.infer.utils_infer import (
    chunk_text,
    load_vocoder,
    transcribe,
    initialize_asr_pipeline,
)


class F5TTSWrapper:
    """
    A wrapper class for F5-TTS that preprocesses reference audio once 
    and allows for repeated TTS generation.
    """
    
    def __init__(
        self, 
        model_name: str = "F5TTS_v1_Base", 
        ckpt_path: Optional[str] = None,
        vocab_file: Optional[str] = None,
        vocoder_name: str = "vocos",
        use_local_vocoder: bool = False,
        vocoder_path: Optional[str] = None,
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        target_sample_rate: int = 24000,
        n_mel_channels: int = 100,
        hop_length: int = 256,
        win_length: int = 1024,
        n_fft: int = 1024,
        ode_method: str = "euler",
        use_ema: bool = True,
        use_duration_predictor: bool = False,
    ):
        """
        Initialize the F5-TTS wrapper with model configuration.
        
        Args:
            model_name: Name of the F5-TTS model variant (e.g., "F5TTS_v1_Base")
            ckpt_path: Path to the model checkpoint file. If None, will use default path.
            vocab_file: Path to the vocab file. If None, will use default.
            vocoder_name: Name of the vocoder to use ("vocos" or "bigvgan")
            use_local_vocoder: Whether to use a local vocoder or download from HF
            vocoder_path: Path to the local vocoder. Only used if use_local_vocoder is True.
            device: Device to run the model on. If None, will automatically determine.
            hf_cache_dir: Directory to cache HuggingFace models
            target_sample_rate: Target sample rate for audio
            n_mel_channels: Number of mel channels
            hop_length: Hop length for the mel spectrogram
            win_length: Window length for the mel spectrogram
            n_fft: FFT size for the mel spectrogram
            ode_method: ODE method for sampling ("euler" or "midpoint")
            use_ema: Whether to use EMA weights from the checkpoint
            use_duration_predictor: Whether to use the duration predictor for inference
        """
        # Set device
        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available()
                else "xpu" if torch.xpu.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device
            
        # Audio processing parameters
        self.target_sample_rate = target_sample_rate
        self.n_mel_channels = n_mel_channels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.mel_spec_type = vocoder_name
        
        # Sampling parameters
        self.ode_method = ode_method
        
        # Duration predictor setting
        self.use_duration_predictor = use_duration_predictor
        
        # Initialize ASR for transcription if needed
        initialize_asr_pipeline(device=self.device)
        
        # Load model configuration
        if ckpt_path is None:
            # Mark that we need to check for EraX vocab later
            self._vocab_file_not_set = (vocab_file is None)
            
            # Check for EraX model first (local or auto-download)
            ckpt_path = self._get_erax_model_path(hf_cache_dir)
            
            # If not EraX model, use standard F5TTS models
            if ckpt_path is None:
                repo_name = "F5-TTS"
                ckpt_step = 1250000
                ckpt_type = "safetensors"
                
                # Adjust for previous models
                if model_name == "F5TTS_v1_Custom_Prune_14":
                    if vocoder_name == "vocos":
                        ckpt_step = 1200000
                elif model_name == "F5TTS_Base":
                    if vocoder_name == "vocos":
                        ckpt_step = 1200000
                    elif vocoder_name == "bigvgan":
                        model_name = "F5TTS_Base_bigvgan"
                        ckpt_type = "pt"
                elif model_name == "E2TTS_Base":
                    repo_name = "E2-TTS"
                    ckpt_step = 1200000
                else:
                     if vocoder_name == "vocos":
                        ckpt_step = 1200000
                    
                ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}"))
        
        # Load model configuration
        if "custom" not in model_name.lower(): 
            config_path = str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
        else:
             config_path = model_name
            
        model_cfg = OmegaConf.load(config_path)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch
        
        # Load tokenizer
        if vocab_file is None:
            # Check if we have EraX vocab from auto-download
            if hasattr(self, '_erax_vocab_file') and self._erax_vocab_file:
                vocab_file = self._erax_vocab_file
                print(f"✅ Using EraX vocab file: {vocab_file}")
            else:
                vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
                print(f"📁 Using default vocab file: {vocab_file}")
        else:
            print(f"📁 Using custom vocab file: {vocab_file}")
        
        tokenizer_type = "custom"
        self.vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer_type)
        
        # Create model
        self.model = CFM(
            transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
            mel_spec_kwargs=dict(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=vocoder_name,
            ),
            odeint_kwargs=dict(
                method=ode_method,
            ),
            vocab_char_map=self.vocab_char_map,
        ).to(self.device)
        
        # Load checkpoint
        dtype = torch.float32 if vocoder_name == "bigvgan" else None
        self._load_checkpoint(self.model, ckpt_path, dtype=dtype, use_ema=use_ema)
        
        # Check for duration predictor
        self.has_duration_predictor = hasattr(self.model, 'duration_predictor') and self.model.duration_predictor is not None
        if self.use_duration_predictor and not self.has_duration_predictor:
            print("Warning: Duration predictor requested but not found in model. Using fallback duration calculation.")
            self.use_duration_predictor = False
        elif self.has_duration_predictor:
            print("Duration predictor found in model.")
        
        # Load vocoder
        if vocoder_path is None:
            if vocoder_name == "vocos":
                vocoder_path = "../checkpoints/vocos-mel-24khz"
            elif vocoder_name == "bigvgan":
                vocoder_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
        
        self.vocoder = load_vocoder(
            vocoder_name=vocoder_name,
            is_local=use_local_vocoder,
            local_path=vocoder_path,
            device=self.device,
            hf_cache_dir=hf_cache_dir
        )
        
        # Storage for reference data
        self.ref_audio_processed = None
        self.ref_text = None
        self.ref_audio_len = None
        
        # Default inference parameters
        self.target_rms = 0.1
        self.cross_fade_duration = 0.15
        self.nfe_step = 32
        self.cfg_strength = 2.0
        self.sway_sampling_coef = -1.0
        self.speed = 1.0
        self.fix_duration = None

    def _get_erax_model_path(self, hf_cache_dir=None):
        """
        Get EraX model path, auto-downloading if not exists locally.
        
        Args:
            hf_cache_dir: HuggingFace cache directory
            
        Returns:
            Path to EraX model file, or None if not found/downloaded
        """
        # Check local path first
        local_model_dir = Path("./erax-ai_model")
        local_model_file = local_model_dir / "model_48000.safetensors"
        local_vocab_file = local_model_dir / "vocab.txt"
        
        if local_model_file.exists() and local_vocab_file.exists():
            print(f"✅ Found EraX model locally: {local_model_file}")
            # Also set vocab_file if not already set
            if getattr(self, '_vocab_file_not_set', False):
                self._erax_vocab_file = str(local_vocab_file)
            return str(local_model_file)
        
        # Try to auto-download from HuggingFace
        print("🔍 EraX model not found locally. Attempting to download from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            
            # Create local directory
            local_model_dir.mkdir(exist_ok=True)
            
            # Download model file
            if not local_model_file.exists():
                print("📥 Downloading model_48000.safetensors from erax-ai/EraX-Smile-UnixSex-F5...")
                downloaded_model = hf_hub_download(
                    repo_id="erax-ai/EraX-Smile-UnixSex-F5",
                    filename="models/model_48000.safetensors",
                    cache_dir=hf_cache_dir or "./hf_cache"
                )
                shutil.copy2(downloaded_model, local_model_file)
                print(f"✅ Model downloaded to: {local_model_file}")
            
            # Download vocab file
            if not local_vocab_file.exists():
                print("📥 Downloading vocab.txt from erax-ai/EraX-Smile-UnixSex-F5...")
                downloaded_vocab = hf_hub_download(
                    repo_id="erax-ai/EraX-Smile-UnixSex-F5", 
                    filename="models/vocab.txt",
                    cache_dir=hf_cache_dir or "./hf_cache"
                )
                shutil.copy2(downloaded_vocab, local_vocab_file)
                print(f"✅ Vocab downloaded to: {local_vocab_file}")
            
            # Set vocab file for later use
            self._erax_vocab_file = str(local_vocab_file)
            print("🎉 EraX model download completed successfully!")
            return str(local_model_file)
            
        except Exception as e:
            print(f"❌ Failed to download EraX model: {e}")
            print("💡 Please manually download from: https://huggingface.co/erax-ai/EraX-Smile-UnixSex-F5/tree/main/models")
            return None

    def _load_checkpoint(self, model, ckpt_path, dtype=None, use_ema=True):
        """
        Load model checkpoint with proper handling of different checkpoint formats.
        
        Args:
            model: The model to load weights into
            ckpt_path: Path to the checkpoint file
            dtype: Data type for model weights
            use_ema: Whether to use EMA weights from the checkpoint
        
        Returns:
            Loaded model
        """
        if dtype is None:
            dtype = (
                torch.float16
                if "cuda" in self.device
                and torch.cuda.get_device_properties(self.device).major >= 7
                and not torch.cuda.get_device_name().endswith("[ZLUDA]")
                else torch.float32
            )
        model = model.to(dtype)

        ckpt_type = ckpt_path.split(".")[-1]
        if ckpt_type == "safetensors":
            from safetensors.torch import load_file
            checkpoint = load_file(ckpt_path, device=self.device)
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=True)

        if use_ema:
            if ckpt_type == "safetensors":
                checkpoint = {"ema_model_state_dict": checkpoint}
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }

            # patch for backward compatibility
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            if ckpt_type == "safetensors":
                checkpoint = {"model_state_dict": checkpoint}
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        del checkpoint
        torch.cuda.empty_cache()

        return model.to(self.device)
    
    def preprocess_reference(self, ref_audio_path: str, ref_text: str = "", clip_short: bool = True):
        """
        Preprocess the reference audio and text, storing them for later use.
        
        Args:
            ref_audio_path: Path to the reference audio file
            ref_text: Text transcript of reference audio. If empty, will auto-transcribe.
            clip_short: Whether to clip long audio to shorter segments
            
        Returns:
            Tuple of processed audio and text
        """
        print("Converting audio...")
        # Load audio file
        aseg = AudioSegment.from_file(ref_audio_path)
        
        if clip_short:
            # 1. try to find long silence for clipping
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                    print("Audio is over 12s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg
                
            # 2. try to find short silence for clipping if 1. failed
            if len(non_silent_wave) > 12000:
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                        print("Audio is over 12s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg
                    
            aseg = non_silent_wave
                
            # 3. if no proper silence found for clipping
            if len(aseg) > 12000:
                aseg = aseg[:12000]
                print("Audio is over 12s, clipping short. (3)")
            
        # Remove silence edges
        aseg = self._remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
        
        # Export to temporary file and load as tensor
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            aseg.export(tmp_file.name, format="wav")
            processed_audio_path = tmp_file.name
            
        # Transcribe if needed
        if not ref_text.strip():
            print("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(processed_audio_path)
        else:
            print("Using custom reference text...")
            
        # Ensure ref_text ends with proper punctuation
        if not ref_text.endswith(". ") and not ref_text.endswith("。"):
            if ref_text.endswith("."):
                ref_text += " "
            else:
                ref_text += ". "
                
        print("\nReference text:", ref_text)
        
        # Load and process audio
        audio, sr = torchaudio.load(processed_audio_path)
        if audio.shape[0] > 1:  # Convert stereo to mono
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Normalize volume
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < self.target_rms:
            audio = audio * self.target_rms / rms
            
        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)
            
        # Move to device
        audio = audio.to(self.device)
        
        # Store reference data
        self.ref_audio_processed = audio
        self.ref_text = ref_text
        self.ref_audio_len = audio.shape[-1] // self.hop_length
        
        # Remove temporary file
        os.unlink(processed_audio_path)
        
        return audio, ref_text
    
    def _remove_silence_edges(self, audio, silence_threshold=-42):
        """
        Remove silence from the start and end of audio.

        Args:
            audio: AudioSegment to process
            silence_threshold: dB threshold to consider as silence

        Returns:
            Processed AudioSegment
        """
        # Remove silence from the start
        non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
        audio = audio[non_silent_start_idx:]

        # Remove silence from the end
        non_silent_end_duration = audio.duration_seconds
        for ms in reversed(audio):
            if ms.dBFS > silence_threshold:
                break
            non_silent_end_duration -= 0.001
        trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

        return trimmed_audio
    
    def calculate_duration_with_predictor(self, text_tokens, text_lengths, local_speed=1.0):
        """
        Calculate duration using the model's duration predictor.
        
        Args:
            text_tokens: Tokenized text batch
            text_lengths: Lengths of each text in the batch
            local_speed: Speed factor for duration
            
        Returns:
            Predicted duration in frames
        """
        # Create mask for text tokens
        b, nt = text_tokens.shape
        range_tensor = torch.arange(nt, device=self.device).unsqueeze(0)
        text_tokens_mask = (range_tensor < text_lengths.unsqueeze(1)).int()
        
        # Get duration predictions
        with torch.inference_mode():
            log_durations = self.model.duration_predictor(text_tokens, text_tokens_mask)
            # Sum across token dimension (dim=1) to get total duration
            durations = torch.exp(log_durations).squeeze(-1).sum(dim=1)
            
        # Calculate duration with reference offset and speed adjustment
        duration = self.ref_audio_len + int(durations[0].item() / local_speed)
        return duration
    
    def generate(
        self, 
        text: str,
        output_path: Optional[str] = None,
        nfe_step: Optional[int] = None,
        cfg_strength: Optional[float] = None,
        sway_sampling_coef: Optional[float] = None,
        speed: Optional[float] = None,
        fix_duration: Optional[float] = None,
        cross_fade_duration: Optional[float] = None,
        use_duration_predictor: Optional[bool] = None,
        return_numpy: bool = False,
        return_spectrogram: bool = False,
    ) -> Union[str, Tuple[np.ndarray, int], Tuple[np.ndarray, int, np.ndarray]]:
        """
        Generate speech for the given text using the stored reference audio.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the generated audio. If None, won't save.
            nfe_step: Number of function evaluation steps
            cfg_strength: Classifier-free guidance strength
            sway_sampling_coef: Sway sampling coefficient
            speed: Speed of generated audio
            fix_duration: Fixed duration in seconds
            cross_fade_duration: Duration of cross-fade between segments
            use_duration_predictor: Override default setting for using duration predictor
            return_numpy: If True, returns the audio as a numpy array
            return_spectrogram: If True, also returns the spectrogram
            
        Returns:
            If output_path provided: path to output file
            If return_numpy=True: tuple of (audio_array, sample_rate)
            If return_spectrogram=True: tuple of (audio_array, sample_rate, spectrogram)
        """
        if self.ref_audio_processed is None or self.ref_text is None:
            raise ValueError("Reference audio not preprocessed. Call preprocess_reference() first.")
            
        # Use default values if not specified
        nfe_step = nfe_step if nfe_step is not None else self.nfe_step
        cfg_strength = cfg_strength if cfg_strength is not None else self.cfg_strength
        sway_sampling_coef = sway_sampling_coef if sway_sampling_coef is not None else self.sway_sampling_coef
        speed = speed if speed is not None else self.speed
        fix_duration = fix_duration if fix_duration is not None else self.fix_duration
        cross_fade_duration = cross_fade_duration if cross_fade_duration is not None else self.cross_fade_duration
        use_predictor = use_duration_predictor if use_duration_predictor is not None else self.use_duration_predictor
        
        # Check if we can use the duration predictor
        can_use_predictor = use_predictor and self.has_duration_predictor
        
        # Split the input text into batches
        audio_len = self.ref_audio_processed.shape[-1] / self.target_sample_rate
        max_chars = int(len(self.ref_text.encode("utf-8")) / audio_len * (22 - audio_len))
        text_batches = chunk_text(text, max_chars=max_chars)
        
        for i, text_batch in enumerate(text_batches):
            print(f"Text batch {i}: {text_batch}")
        print("\n")
        
        # Generate audio for each batch
        generated_waves = []
        spectrograms = []
        
        for text_batch in text_batches:
            # Adjust speed for very short texts
            local_speed = speed
            if len(text_batch.encode("utf-8")) < 10:
                local_speed = speed * 0.8
                
            # Prepare the text
            text_list = [self.ref_text + text_batch]
            final_text_list = convert_char_to_pinyin(text_list)
            
            # Calculate duration
            if fix_duration is not None:
                duration = int(fix_duration * self.target_sample_rate / self.hop_length)
            elif can_use_predictor:
                text_tokens = list_str_to_idx(final_text_list, self.vocab_char_map)[0]
                text_lengths = torch.tensor([len(text_tokens)], device=self.device)
                text_tokens = torch.tensor([text_tokens], device=self.device)
                duration = self.calculate_duration_with_predictor(text_tokens, text_lengths, local_speed)
            else:
                duration = self.ref_audio_len + int(len(text_batch) / len(self.ref_text) * self.ref_audio_len / local_speed)
                
            # Generate audio
            with torch.inference_mode():
                generated, _ = self.model.sample(
                    cond=self.ref_audio_processed,
                    text=final_text_list,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )
                
            generated_waves.append(generated[:, self.ref_audio_len:, :])
            if return_spectrogram:
                spectrograms.append(generated[0, self.ref_audio_len:, :].cpu().numpy())
        
        # Combine all segments
        if generated_waves:
            if cross_fade_duration <= 0:
                generated_wave = torch.cat(generated_waves, dim=1)
            else:
                generated_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = generated_wave
                    next_wave = generated_waves[i]
                    cross_fade_samples = int(cross_fade_duration * self.target_sample_rate / self.hop_length)
                    if prev_wave.shape[1] > cross_fade_samples and next_wave.shape[1] > cross_fade_samples:
                        fade_out_prev = prev_wave[:, -cross_fade_samples:, :] * torch.linspace(1, 0, cross_fade_samples, device=self.device).unsqueeze(0).unsqueeze(-1)
                        fade_in_next = next_wave[:, :cross_fade_samples, :] * torch.linspace(0, 1, cross_fade_samples, device=self.device).unsqueeze(0).unsqueeze(-1)
                        cross_faded = fade_out_prev + fade_in_next
                        generated_wave = torch.cat([prev_wave[:, :-cross_fade_samples, :], cross_faded, next_wave[:, cross_fade_samples:, :]], dim=1)
                    else:
                        generated_wave = torch.cat([generated_wave, next_wave], dim=1)
            
            # Combine spectrograms if needed
            if return_spectrogram or output_path is not None:
                final_spectrogram = np.concatenate(spectrograms, axis=0) if spectrograms else None
                
            # Save to file if path provided
            if output_path is not None:
                generated_mel_spec = generated_wave.permute(0, 2, 1).float()
                generated_wave = self.vocoder.decode(generated_mel_spec)
                if generated_wave.dim() > 1:
                    generated_wave = generated_wave.squeeze()
                torchaudio.save(output_path, generated_wave.unsqueeze(0).cpu(), self.target_sample_rate)
                if return_spectrogram:
                    spec_path = output_path.replace('.wav', '_spectrogram.png')
                    self._save_spectrogram(final_spectrogram, spec_path)
                return output_path
            
            # Return numpy array if requested
            if return_numpy:
                generated_mel_spec = generated_wave.permute(0, 2, 1).float()
                generated_wave = self.vocoder.decode(generated_mel_spec)
                if generated_wave.dim() > 1:
                    generated_wave = generated_wave.squeeze()
                audio_array = generated_wave.cpu().numpy()
                if return_spectrogram:
                    return audio_array, self.target_sample_rate, final_spectrogram
                return audio_array, self.target_sample_rate
        
        else:
            raise RuntimeError("No audio was generated")
    
    def _save_spectrogram(self, spectrogram, path):
        """Save spectrogram as image"""
        plt.figure(figsize=(12, 4))
        plt.imshow(spectrogram, origin="lower", aspect="auto")
        plt.colorbar()
        plt.savefig(path)
        plt.close()
        
    def get_current_audio_length(self):
        """Get the length of the reference audio in seconds"""
        if self.ref_audio_processed is None:
            return 0
        return self.ref_audio_processed.shape[-1] / self.target_sample_rate