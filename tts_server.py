import os
import json
import time
import numpy as np
import argparse
from typing import List, Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Form, Request, Depends, Security, APIRouter
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import asyncio
import io
import wave
from vinorm import TTSnorm
import torch
import traceback
import re
from pathlib import Path
from contextlib import asynccontextmanager
import gc
from pydub import AudioSegment

# Import F5TTS wrapper
from f5tts_wrapper import F5TTSWrapper
import os
import requests
import time
import io
import wave
import asyncio
import gc
import traceback
import uvicorn
import argparse
import tempfile
import numpy as np
import torch
import torchaudio



# --- Configuration ---
MODEL_CONFIG = {
    "model_name": "F5TTS_v1_Base",  # Use F5TTS_v1_Base for EraX models
    "vocoder_name": "vocos",  # Using Vocos vocoder as in example
    "ckpt_path": None,  # Will be set after auto-download
    "vocab_file": None,  # Will be set after auto-download
    "use_ema": True,  # Set to True for better quality (corrected from example)
    "target_sample_rate": 24000,  # Output sample rate (24kHz is standard)
    "use_duration_predictor": False,  # Standard setting
    # Additional parameters from example code for better compatibility
    "n_mel_channels": 100,  # Number of mel-spectrogram channels
    "hop_length": 256,      # Hop length for audio processing
    "win_length": 1024,     # Window length for STFT
    "n_fft": 1024,         # FFT size
    "ode_method": "euler",   # ODE solver method
    "hf_cache_dir": "./hf_cache"  # Cache directory for HuggingFace downloads
}

# Separate flag for dtype fixing
FORCE_FLOAT32 = True  # Fix dtype mismatches

DEFAULT_REFERENCES = {
    "default_vi": {
        "audio": "./ref_audios/default_vi.wav",
        "text": "./ref_audios/default_vi.txt",
        "name": "Default Vietnamese"
    }
}

CUSTOM_REF_PATH = "./ref_audios"
TEXT_SPLITTER_CHUNK_SIZE = 200
TEXT_SPLITTER_CHUNK_OVERLAP = 20

# Global variables
DEFAULT_SAMPLE_RATE = 24000
tts_model: Optional[F5TTSWrapper] = None
reference_cache: Dict[str, Dict[str, Any]] = {}
os.makedirs(CUSTOM_REF_PATH, exist_ok=True)

# --- Pydantic Models ---
class TTSRequest(BaseModel):
    text: str
    speaker: Optional[str] = "male"
    nfe_step: int = 32
    cfg_strength: float = 2.0
    speed: float = 1.0
    cross_fade_duration: float = 0.15
    sway_sampling_coef: float = -1.0

class OpenAITTSRequest(BaseModel):
    model: str
    input: str
    voice: str
    instructions: Optional[str] = None
    response_format: Optional[str] = "wav"

# --- Authentication ---
API_KEY_NAME = "Authorization"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
SERVER_API_KEY = os.getenv("API_KEY")

async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Dependency to validate the API key from the Authorization header."""
    if not SERVER_API_KEY:
        print("FATAL: Server is not configured with an API_KEY.")
        raise HTTPException(
            status_code=500,
            detail="Server is not configured for authentication. Please set the API_KEY environment variable."
        )

    if api_key_header is None:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is missing"
        )

    parts = api_key_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format. Must be 'Bearer <key>'"
        )

    client_api_key = parts[1]
    if client_api_key != SERVER_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )

PROTECTED = Depends(get_api_key)

# --- FastAPI App and Router ---
app = FastAPI(title="Unified F5TTS + OpenAI Compatible TTS Server")
api_router_v1 = APIRouter(prefix="/v1", dependencies=[PROTECTED])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utility Functions ---
def split_text_into_chunks(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.
    Based on viF5TTS implementation.

    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def create_wave_header(sample_rate, num_channels=1, bits_per_sample=16, data_size=0):
    """Create a wave header for streaming. data_size=0 means unknown size."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(bits_per_sample // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(b'')
    header_bytes = buffer.getvalue()
    if data_size > 0:
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(bits_per_sample // 8)
            wf.setframerate(sample_rate)
            wf.setnframes(data_size // (num_channels * (bits_per_sample // 8)))
            wf.writeframes(b'')
        header_bytes = buffer.getvalue()
    return header_bytes

def convert_wav_to_mp3(wav_data: bytes, sample_rate: int = 24000, bitrate: str = "128k") -> bytes:
    """Convert WAV audio data to MP3 format using pydub."""
    try:
        # Create AudioSegment from WAV bytes
        wav_io = io.BytesIO(wav_data)
        audio = AudioSegment.from_wav(wav_io)
        
        # Convert to MP3
        mp3_io = io.BytesIO()
        audio.export(mp3_io, format="mp3", bitrate=bitrate)
        mp3_io.seek(0)
        
        return mp3_io.read()
    except Exception as e:
        print(f"Error converting WAV to MP3: {e}")
        raise e

def process_chunk(chunk_text: str, model: F5TTSWrapper, request: TTSRequest) -> Optional[bytes]:
    """Process a single text chunk and return raw audio bytes (int16)"""
    # Clean and normalize the text
    chunk_text = TTSnorm(chunk_text).strip()
    
    # Skip empty chunks
    if not chunk_text or len(chunk_text.strip()) < 3:
        return None

    print(f"  Synthesizing chunk: '{chunk_text}'")
    try:
        audio_array, _ = model.generate(
            text=chunk_text, return_numpy=True, nfe_step=request.nfe_step,
            cfg_strength=request.cfg_strength, speed=request.speed,
            cross_fade_duration=request.cross_fade_duration,
            sway_sampling_coef=request.sway_sampling_coef,
            use_duration_predictor=model.use_duration_predictor
        )
        if audio_array is None or audio_array.size == 0:
            print(f"Warning: Model generated empty audio for chunk: '{chunk_text}'")
            return None
        audio_int16 = (audio_array * 32767).astype(np.int16)
        return audio_int16.tobytes()
    except Exception as e:
        print(f"Error generating audio for chunk '{chunk_text}': {e}")
        traceback.print_exc()
        return None

@asynccontextmanager
async def model_context(ref_id: str):
    """Context manager for safely managing model state"""
    global tts_model, reference_cache
    async with model_lock:
        try:
            cached_ref_data = reference_cache.get(ref_id)
            if not cached_ref_data or cached_ref_data.get("loaded") != True:
                raise ValueError(f"Reference '{ref_id}' not ready")
            
            # Set model state from cache (similar to viF5TTS approach)
            tts_model.ref_audio_processed = cached_ref_data["processed_mel"]
            tts_model.ref_text = cached_ref_data["processed_text"]
            tts_model.ref_audio_len = cached_ref_data["processed_mel_len"]
            
            # Ensure tensor device and dtype compatibility
            if tts_model.ref_audio_processed.device != tts_model.device:
                print(f"Warning: Moving cached mel tensor from {tts_model.ref_audio_processed.device} to {tts_model.device}")
                tts_model.ref_audio_processed = tts_model.ref_audio_processed.to(tts_model.device)
            
            # Force dtype consistency to avoid half/float mismatch errors
            if hasattr(tts_model, 'model') and hasattr(tts_model.model, 'dtype'):
                target_dtype = tts_model.model.dtype
                if tts_model.ref_audio_processed.dtype != target_dtype:
                    print(f"Warning: Converting mel tensor from {tts_model.ref_audio_processed.dtype} to {target_dtype}")
                    tts_model.ref_audio_processed = tts_model.ref_audio_processed.to(dtype=target_dtype)
            
            # **CORRECTED** Ensure vocoder is consistently float32 to prevent runtime errors
            if hasattr(tts_model, 'vocoder'):
                tts_model.vocoder = tts_model.vocoder.float()
            
            print(f"[Model Context] Set reference state for '{ref_id}'. Mel shape: {tts_model.ref_audio_processed.shape}, dtype: {tts_model.ref_audio_processed.dtype}")
            yield tts_model
        finally:
            # Always reset model state after use (like viF5TTS)
            if tts_model:
                tts_model.ref_audio_processed = None
                tts_model.ref_text = None
                tts_model.ref_audio_len = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print("[Model Context] Cleaned up model state and GPU memory")

# --- Thread safety and resource management ---
model_lock = asyncio.Lock()
reference_processing_lock = asyncio.Lock()
request_semaphore = asyncio.Semaphore(3)
background_semaphore = asyncio.Semaphore(2)

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize the F5TTS model on startup"""
    global tts_model, reference_cache, DEFAULT_SAMPLE_RATE
    print("Starting up F5TTS Server...")
    
    if not SERVER_API_KEY:
        print("="*80)
        print("WARNING: API_KEY environment variable not set!")
        print("The server will fail on protected endpoints until the key is configured.")
        print("="*80)
    
    try:
        print("🔍 Initializing F5TTS with automatic EraX model detection...")
        print(f"Loading F5TTS model with config: {MODEL_CONFIG}")
        tts_model = F5TTSWrapper(**MODEL_CONFIG)
        
        # Fix dtype mismatches if FORCE_FLOAT32 is enabled
        if FORCE_FLOAT32:
            print("🔧 Applying float32 dtype fix to the vocoder...")
            if hasattr(tts_model, 'vocoder'):
                # **CORRECTED** More robustly cast the entire vocoder module to float32
                tts_model.vocoder = tts_model.vocoder.float()
                print("✅ Vocoder dtype robustly fixed to float32")
        
        print(f"F5TTS model loaded successfully. Device: {tts_model.device}")
        await load_default_references()
        DEFAULT_SAMPLE_RATE = MODEL_CONFIG.get("target_sample_rate", 24000)
        print(f"F5TTS Server ready! Available voices: {list(reference_cache.keys())}")
    except Exception as e:
        print(f"FATAL: Failed to load F5TTS model during startup: {e}")
        traceback.print_exc()
        tts_model = None

async def load_default_references():
    """Load default reference audio files and cache processed data."""
    if tts_model is None: return
    print("Loading and processing default references...")
    async with reference_processing_lock:
        for ref_id, ref_data in DEFAULT_REFERENCES.items():
            try:
                audio_path = ref_data["audio"]
                text_path = ref_data["text"]
                
                if os.path.exists(audio_path):
                    # Read text from file if text_path is provided
                    ref_text = ""
                    if os.path.exists(text_path):
                        with open(text_path, 'r', encoding='utf-8') as f:
                            ref_text = f.read().strip()
                    
                    async with model_lock:
                        try:
                            _, _ = tts_model.preprocess_reference(
                                ref_audio_path=audio_path,
                                ref_text=TTSnorm(ref_text).strip(),
                                clip_short=False
                            )
                            reference_cache[ref_id] = {
                                "ref_audio_path": audio_path,
                                "ref_text_original": ref_text, "loaded": True,
                                "name": ref_data.get("name", ref_id),
                                "processed_mel": tts_model.ref_audio_processed.clone().detach(),
                                "processed_text": tts_model.ref_text,
                                "processed_mel_len": tts_model.ref_audio_len, "error": None
                            }
                            print(f"Successfully cached reference '{ref_id}'.")
                        finally:
                            if tts_model:
                                tts_model.ref_audio_processed = None
                                tts_model.ref_text = None
                                tts_model.ref_audio_len = None
                                if torch.cuda.is_available(): torch.cuda.empty_cache()
                                gc.collect()
                else:
                    reference_cache[ref_id] = {"loaded": False, "name": ref_data.get("name", ref_id), "error": "Audio file not found"}
            except Exception as e:
                reference_cache[ref_id] = {"loaded": False, "name": ref_data.get("name", ref_id), "error": str(e)}
    print("Default reference processing complete.")

# --- Audio Generation Functions ---
async def stream_audio_generator(request: TTSRequest) -> AsyncGenerator[bytes, None]:
    """F5TTS audio stream generator with resource management."""
    start_time = time.time()
    print(f"[stream_audio_generator] Request received at {start_time:.2f}")
    
    async with request_semaphore:
        if tts_model is None: 
            print("Error: F5TTS model is not initialized.")
            raise HTTPException(503, "TTS model is not ready.")
        if not request.text or not request.text.strip(): 
            print("Error: No text provided in the request.")
            raise HTTPException(400, "Input text cannot be empty.")
        
        print(f"[stream_audio_generator] Speaker selected: '{request.speaker or 'default_vi'}'")
        
        try:
            normalized_text = TTSnorm(request.text).strip()
            print(f"[stream_audio_generator] Normalized text (first 100 chars): '{normalized_text[:100]}...'")
        except Exception: 
            print(f"Warning: Text normalization failed. Proceeding with original text.")
            normalized_text = request.text.strip()
        
        # Use viF5TTS chunking method
        text_chunks = split_text_into_chunks(normalized_text, max_chars=135)
        num_chunks = len(text_chunks)
        print(f"[stream_audio_generator] Text split into {num_chunks} chunks.")
        
        if not text_chunks:
            print("Warning: Text resulted in zero chunks after splitting.")
            yield create_wave_header(tts_model.target_sample_rate)
            return

        try:
            async with model_context(request.speaker or "default_vi") as model:
                sample_rate = model.target_sample_rate
                print(f"[stream_audio_generator] Starting audio stream generation at {sample_rate} Hz...")
                
                yield create_wave_header(sample_rate=sample_rate, data_size=0)
                print("[stream_audio_generator] WAV header yielded.")
                
                total_bytes_yielded = 0
                for i, chunk_text in enumerate(text_chunks):
                    chunk_num = i + 1
                    chunk_start_time = time.time()
                    print(f"[stream_audio_generator] Processing chunk {chunk_num}/{num_chunks}...")
                    
                    audio_bytes = process_chunk(chunk_text, model, request)
                    if audio_bytes and len(audio_bytes) > 0:
                        yield audio_bytes
                        bytes_yielded = len(audio_bytes)
                        total_bytes_yielded += bytes_yielded
                        chunk_duration = time.time() - chunk_start_time
                        print(f"  [Chunk {chunk_num}] Yielded {bytes_yielded} bytes. Time: {chunk_duration:.3f}s")
                    else:
                        print(f"  [Chunk {chunk_num}] Skipped yielding (no audio data generated).")
                        
        except ValueError as ve: 
            print(f"Error: Required processed data not found: {ve}")
            raise HTTPException(404, str(ve))
        except Exception as e: 
            print(f"Internal error during audio generation: {e}")
            traceback.print_exc()
            raise HTTPException(500, f"Internal error during audio generation: {e}")
        
        await asyncio.sleep(0.05)
        total_duration = time.time() - start_time
        print(f"[stream_audio_generator] Stream generation complete.")
        print(f"  Total audio bytes yielded: {total_bytes_yielded}")
        print(f"  Total request processing time: {total_duration:.3f} seconds.")

async def stream_mp3_generator(request: TTSRequest) -> AsyncGenerator[bytes, None]:
    """Generate MP3 audio stream by collecting WAV chunks and converting to MP3."""
    print("[stream_mp3_generator] Starting MP3 generation...")
    
    # Collect all WAV audio data first
    wav_chunks = []
    is_header = True
    
    async for audio_bytes in stream_audio_generator(request):
        if is_header:
            # Skip WAV header for MP3 conversion
            is_header = False
            continue
        wav_chunks.append(audio_bytes)
    
    if not wav_chunks:
        print("[stream_mp3_generator] No audio data to convert.")
        return
    
    # Combine all WAV chunks
    combined_audio = b''.join(wav_chunks)
    
    # Create a complete WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(DEFAULT_SAMPLE_RATE)
        wf.writeframes(combined_audio)
    
    wav_data = buffer.getvalue()
    
    # Convert to MP3
    try:
        mp3_data = convert_wav_to_mp3(wav_data, DEFAULT_SAMPLE_RATE)
        print(f"[stream_mp3_generator] Converted {len(wav_data)} bytes WAV to {len(mp3_data)} bytes MP3")
        yield mp3_data
    except Exception as e:
        print(f"[stream_mp3_generator] Error converting to MP3: {e}")
        raise HTTPException(500, f"Error converting audio to MP3: {e}")

async def process_and_cache_reference(file_path: str, text: Optional[str], ref_id: str):
    """Process a reference audio file in the background."""
    async with background_semaphore:
        print(f"Background task: Processing reference {ref_id}")
        if tts_model is None:
            async with reference_processing_lock:
                reference_cache[ref_id].update({"loaded": False, "error": "TTS Model not available."})
            return
        try:
            normalized_text = TTSnorm(text).strip() if text else ""
            async with model_lock:
                try:
                    _, _ = tts_model.preprocess_reference(
                        ref_audio_path=file_path, ref_text=normalized_text, clip_short=True
                    )
                    async with reference_processing_lock:
                        reference_cache[ref_id].update({
                            "loaded": True, "error": None,
                            "processed_mel": tts_model.ref_audio_processed.clone().detach(),
                            "processed_text": tts_model.ref_text,
                            "processed_mel_len": tts_model.ref_audio_len,
                        })
                    print(f"Background task: Successfully cached reference '{ref_id}'.")
                finally:
                    if tts_model:
                        tts_model.ref_audio_processed, tts_model.ref_text, tts_model.ref_audio_len = None, None, None
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        gc.collect()
        except Exception as e:
            async with reference_processing_lock:
                reference_cache[ref_id].update({"loaded": False, "error": str(e)})

# --- API Endpoints (V1) ---
@api_router_v1.post("/tts/stream")
async def tts_stream(request: TTSRequest):
    """F5TTS streaming endpoint"""
    return StreamingResponse(stream_audio_generator(request), media_type="audio/wav")

@api_router_v1.post("/audio/speech")
async def generate_speech_openai(request_data: OpenAITTSRequest):
    """OpenAI-compatible TTS endpoint using F5TTS directly"""
    try:
        # Validate voice reference
        if request_data.voice not in reference_cache:
            available_voices = list(reference_cache.keys())
            raise HTTPException(400, f"Voice '{request_data.voice}' not found. Available voices: {available_voices}")
        
        cached_ref_data = reference_cache[request_data.voice]
        if cached_ref_data.get("loaded") != True:
            status = cached_ref_data.get('loaded', 'Not Found')
            error_msg = cached_ref_data.get('error', 'N/A')
            error_detail = f"Reference voice '{request_data.voice}' is not ready. Status: {status}"
            if status == 'processing':
                error_detail += " Still processing, please wait."
            elif error_msg != 'N/A':
                error_detail += f" Error: {error_msg}"
            raise HTTPException(503, error_detail)

        # Create F5TTS request
        f5tts_request = TTSRequest(text=request_data.input, speaker=request_data.voice)
        
        # Handle response format
        response_format = request_data.response_format.lower() if request_data.response_format else "wav"
        
        if response_format == "wav":
            headers = {
                "Content-Type": "audio/wav",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
            return StreamingResponse(
                stream_audio_generator(f5tts_request), 
                media_type="audio/wav",
                headers=headers
            )
        elif response_format == "mp3":
            headers = {
                "Content-Type": "audio/mpeg",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
            return StreamingResponse(
                stream_mp3_generator(f5tts_request),
                media_type="audio/mpeg",
                headers=headers
            )
        elif response_format == "pcm":
            media_type = f"audio/L16;rate={DEFAULT_SAMPLE_RATE};channels=1"
            async def pcm_gen():
                is_header = True
                async for audio_bytes in stream_audio_generator(f5tts_request):
                    if is_header:
                        is_header = False
                        continue
                    yield audio_bytes
            return StreamingResponse(pcm_gen(), media_type=media_type)
        else:
            raise HTTPException(400, f"Unsupported response format: {response_format}. Supported: wav, mp3, pcm")
            
    except HTTPException:
        # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        print(f"Unhandled error in /v1/audio/speech endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Internal server error generating audio: {str(e)}")


@api_router_v1.post("/upload_reference")
async def upload_reference(
    background_tasks: BackgroundTasks, file: UploadFile = File(...), text: Optional[str] = Form(None)
):
    """Upload a custom reference audio file for processing."""
    if not file.filename: raise HTTPException(400, "No file provided")
    
    timestamp = int(time.time())
    ref_id = f"custom_{timestamp}"
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.wav', '.mp3']: raise HTTPException(400, "Unsupported audio format.")
    
    ref_path = os.path.join(CUSTOM_REF_PATH, f"{ref_id}{file_extension}")
    try:
        with open(ref_path, "wb") as buffer: buffer.write(await file.read())
        
        async with reference_processing_lock:
            reference_cache[ref_id] = {
                "ref_audio_path": ref_path, "loaded": "processing",
                "name": f"Custom {timestamp} (Processing...)"
            }
        
        background_tasks.add_task(process_and_cache_reference, ref_path, text, ref_id)
        
        return {"status": "processing", "ref_id": ref_id, "name": reference_cache[ref_id]["name"]}
    except Exception as e:
        raise HTTPException(500, f"Error saving reference file: {e}")

@api_router_v1.get("/references", dependencies=None) # Make this endpoint public
async def get_references():
    """Get available reference presets (default and custom)."""
    return {
        ref_id: {"name": data.get("name", ref_id), "status": data.get("loaded", "error")}
        for ref_id, data in reference_cache.items()
        if data.get("loaded") is True or data.get("loaded") == "processing"
    }

@api_router_v1.get("/health", dependencies=None) # Make this endpoint public
async def health_check():
    """Health check endpoint."""
    if tts_model is None: return {"status": "error", "message": "F5TTS model not initialized"}
    loaded_refs = sum(1 for data in reference_cache.values() if data.get("loaded") is True)
    if loaded_refs == 0: return {"status": "warning", "message": "Model loaded, but no reference voices are ready."}
    return {"status": "ok", "message": "F5TTS model ready.", "loaded_references": loaded_refs}

@api_router_v1.get("/model")
async def get_model_name():
    """Endpoint to get the name of the current model."""
    model_name = MODEL_CONFIG.get("model_name", "Unknown")
    return {"model_name": model_name}

# Include the v1 router in the main app
app.include_router(api_router_v1)

# --- Web UI Endpoint (at root) ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_client():
    """Serve the HTML client page."""
    try:
        with open("client.html", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Client HTML file not found.")

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="F5TTS Server with OpenAI Compatible API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 F5TTS SERVER WITH OPENAI COMPATIBLE V1 API")
    print("=" * 80)
    print(f"🔑 API Key Required: Set the 'API_KEY' environment variable.")
    print("-" * 80)
    print(f"🌐 Web Interface:       http://{args.host}:{args.port}/")
    print(f"🎤 F5TTS Streaming:     http://{args.host}:{args.port}/v1/tts/stream")
    print(f"🤖 OpenAI Compatible:   http://{args.host}:{args.port}/v1/audio/speech")
    print(f"📁 Reference Upload:    http://{args.host}:{args.port}/v1/upload_reference")
    print(f"❤️  Health Check (Public): http://{args.host}:{args.port}/v1/health")
    print(f"📋 Available Refs (Public): http://{args.host}:{args.port}/v1/references")
    print("=" * 80)
    
    uvicorn.run(app, host=args.host, port=args.port)