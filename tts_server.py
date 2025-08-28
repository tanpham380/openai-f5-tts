import os
import json
import time
import numpy as np
import argparse
from typing import List, Optional, Dict, Any, AsyncGenerator
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File, Form, Request, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import asyncio
import io
import wave
from vinorm import TTSnorm
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
import gc

# Import F5TTS wrapper
from f5tts_wrapper import F5TTSWrapper

# Text normalization import (optional)


# --- Configuration ---
MODEL_CONFIG = {
    "model_name": "F5TTS_v1_Base",  # Use F5TTS_v1_Base for EraX models
    "vocoder_name": "vocos",  # Using Vocos vocoder as in example
    "ckpt_path": "./erax-ai_model/model_48000.safetensors",  # Path to your EraX model
    "vocab_file": "./erax-ai_model/vocab.txt",  # Path to vocab file in your folder
    "use_ema": True,  # Set to True for better quality (corrected from example)
    "target_sample_rate": 24000,  # Output sample rate (24kHz is standard)
    "use_duration_predictor": False,  # Standard setting
    # Additional parameters from example code for better compatibility
    "n_mel_channels": 100,  # Number of mel-spectrogram channels
    "hop_length": 256,      # Hop length for audio processing
    "win_length": 1024,     # Window length for STFT
    "n_fft": 1024,         # FFT size
    "ode_method": "euler"   # ODE solver method
}

DEFAULT_REFERENCES = {
    "male": {
        "audio": "./male_south_TEACH_chunk_0_segment_684.wav",
        "text": "Người người hô hào thay đổi phương pháp giảng dạy. Bộ giáo dục và đào tạo Việt Nam không thiếu những dự án nhằm thay đổi diện mạo giáo dục nước nhà. Nhưng trong khi những thành quả đổi mới còn chưa kịp thu về, thì những ví dụ điển hình về bước lùi của giáo dục ngày càng hiện rõ.",
        "name": "Male Voice (South)"
    },
    "female": {
        "audio": "./female-vts.wav",
        "text": "Ai đã đến Hàng Dương, đều không thể cầm lòng về những nấm mộ chen nhau, nhấp nhô trải khắp một vùng đồi. Những nấm mộ có tên và không tên, nhưng nấm mộ lấp ló trong lùm cây, bụi cỏ.",
        "name": "Female Voice (VTS)"
    }
}

CUSTOM_REF_PATH = "./references"
TEXT_SPLITTER_CHUNK_SIZE = 100
TEXT_SPLITTER_CHUNK_OVERLAP = 0

# Global variables (will be initialized during startup)
DEFAULT_SAMPLE_RATE = 24000  # Updated to match EraX model

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

# --- Utility Functions ---




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

def process_chunk(chunk_text: str, model: F5TTSWrapper, request: TTSRequest) -> Optional[bytes]:
    """Process a single text chunk and return raw audio bytes (int16)"""
    chunk_text = TTSnorm(chunk_text).strip()
    chunk_text = chunk_text.strip()
    
    if not chunk_text:
        return None
    if chunk_text.endswith(".."):
        chunk_text = chunk_text[:-1].strip()

    if not chunk_text:
        return None

    print(f"  Synthesizing chunk: '{chunk_text}'")

    try:
        audio_array, sample_rate = model.generate(
            text=chunk_text,
            return_numpy=True,
            nfe_step=request.nfe_step,
            cfg_strength=request.cfg_strength,
            speed=request.speed,
            cross_fade_duration=request.cross_fade_duration,
            sway_sampling_coef=request.sway_sampling_coef,
            use_duration_predictor=model.use_duration_predictor
        )

        if audio_array is None or audio_array.size == 0:
            print(f"Warning: Model generated empty audio for chunk: '{chunk_text}'")
            return None

        audio_int16 = (audio_array * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes
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
            
            # Set model state
            tts_model.ref_audio_processed = cached_ref_data["processed_mel"]
            tts_model.ref_text = cached_ref_data["processed_text"]
            tts_model.ref_audio_len = cached_ref_data["processed_mel_len"]
            
            # Move to correct device if needed
            if tts_model.ref_audio_processed.device != tts_model.device:
                print(f"Moving cached tensor from {tts_model.ref_audio_processed.device} to {tts_model.device}")
                tts_model.ref_audio_processed = tts_model.ref_audio_processed.to(tts_model.device)
            
            yield tts_model
            
        finally:
            # Always cleanup model state
            if tts_model:
                tts_model.ref_audio_processed = None
                tts_model.ref_text = None
                tts_model.ref_audio_len = None
                
                # Force GPU memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                print("[Model Context] Cleaned up model state and GPU memory")

# --- FastAPI Setup ---
app = FastAPI(title="Unified F5TTS + OpenAI Compatible TTS Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
API_KEY_NAME = "Authorization"
API_KEY_HEADER = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
SERVER_API_KEY = os.getenv("API_KEY")

async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Dependency to validate the API key from the Authorization header."""
    # Nếu server không cấu hình API key, cho phép truy cập (chỉ nên dùng cho local dev)
    # Để bảo mật hơn trong production, bạn nên throw lỗi nếu SERVER_API_KEY không được set
    if not SERVER_API_KEY:
        print("WARNING: No API key set. Server is running in unprotected mode.")
        return

    if api_key_header is None:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is missing"
        )

    # Key phải có dạng "Bearer YOUR_KEY"
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

# Tạo m


PROTECTED = Depends(get_api_key)

# Global variables
tts_model: Optional[F5TTSWrapper] = None
reference_cache: Dict[str, Dict[str, Any]] = {}
os.makedirs(CUSTOM_REF_PATH, exist_ok=True)

# Thread safety and resource management
model_lock = asyncio.Lock()  # Prevents concurrent model access
reference_processing_lock = asyncio.Lock()  # For reference cache operations
request_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent TTS requests
background_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent background tasks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
    chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    keep_separator=True
)

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Initialize the F5TTS model on startup"""
    global tts_model, reference_cache, DEFAULT_SAMPLE_RATE
    print("Starting up F5TTS Server...")
    try:
        # Initialize F5TTS wrapper
        print(f"Loading F5TTS model with config: {MODEL_CONFIG}")
        tts_model = F5TTSWrapper(**MODEL_CONFIG)
        print(f"F5TTS model loaded successfully. Device: {tts_model.device}")

        # Load default references for F5TTS
        await load_default_references()
        
        # Set default sample rate
        DEFAULT_SAMPLE_RATE = MODEL_CONFIG.get("target_sample_rate", 24000)
        
        print(f"F5TTS Server ready!")
        print(f"Available voices: {list(reference_cache.keys())}")
        print(f"Sample rate: {DEFAULT_SAMPLE_RATE}")

    except Exception as e:
        print(f"FATAL: Failed to load F5TTS model during startup: {e}")
        traceback.print_exc()
        tts_model = None

async def load_default_references():
    """Load default reference audio files and cache processed data with proper resource management."""
    global reference_cache, tts_model

    if tts_model is None:
        print("Skipping default reference loading: Model not initialized.")
        return

    print("Loading and processing default references for caching...")
    
    async with reference_processing_lock:
        for ref_id, ref_data in DEFAULT_REFERENCES.items():
            try:
                audio_path = ref_data["audio"]
                if os.path.exists(audio_path):
                    print(f"Processing default reference: {ref_id} from {audio_path}")

                    # Use async context to ensure proper cleanup
                    async with model_lock:
                        try:
                            _, processed_ref_text = tts_model.preprocess_reference(
                                ref_audio_path=audio_path,
                                ref_text=TTSnorm(ref_data.get("text", "")).strip(),
                                clip_short=False
                            )

                            cached_mel = tts_model.ref_audio_processed.clone().detach()
                            cached_mel_len = tts_model.ref_audio_len
                            cached_text = tts_model.ref_text

                            reference_cache[ref_id] = {
                                "ref_audio_path": audio_path,
                                "ref_text_original": ref_data.get("text", ""),
                                "loaded": True,
                                "name": ref_data.get("name", ref_id),
                                "processed_mel": cached_mel,
                                "processed_text": cached_text,
                                "processed_mel_len": cached_mel_len,
                                "error": None
                            }
                            print(f"Successfully processed and cached reference '{ref_id}'. Mel shape: {cached_mel.shape}, Len: {cached_mel_len}")

                        finally:
                            # Always cleanup model state
                            if tts_model:
                                tts_model.ref_audio_processed = None
                                tts_model.ref_text = None
                                tts_model.ref_audio_len = None
                                
                                # Force GPU memory cleanup
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                gc.collect()

                else:
                    print(f"Warning: Default reference audio '{ref_id}' not found at {audio_path}")
                    reference_cache[ref_id] = {"loaded": False, "name": ref_data.get("name", ref_id), "error": "File not found"}

            except Exception as e:
                print(f"Error processing reference '{ref_id}' for caching: {e}")
                traceback.print_exc()
                reference_cache[ref_id] = {"loaded": False, "name": ref_data.get("name", ref_id), "error": str(e)}

    print("Default reference processing and caching complete.")

# --- Audio Generation Functions ---
async def stream_audio_generator(request: TTSRequest) -> AsyncGenerator[bytes, None]:
    """F5TTS audio stream generator using cached reference data with proper resource management."""
    global tts_model, reference_cache, text_splitter

    start_time = time.time()
    print(f"[F5TTS Stream] Request received at {start_time:.2f}")

    # Request queuing - wait for available slot
    async with request_semaphore:
        print(f"[F5TTS Stream] Acquired request slot")

        if tts_model is None:
            print("Error: F5TTS model is not initialized.")
            raise HTTPException(status_code=503, detail="TTS model is not ready. Please try again later.")

        selected_ref_id = request.speaker or "male"
        print(f"[F5TTS Stream] Speaker selected: '{selected_ref_id}'")

        input_text = request.text
        if not input_text or not input_text.strip():
            print("Error: No text provided in the request.")
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        print(f"[F5TTS Stream] Normalizing input text (first 100 chars): '{input_text[:100]}...'")
        try:
            normalized_text = TTSnorm(input_text).strip()
            print(f"[F5TTS Stream] Normalized text (first 100 chars): '{normalized_text[:100]}...'")
        except Exception as e:
            print(f"Warning: Text normalization failed: {e}. Proceeding with original text.")
            traceback.print_exc()
            normalized_text = input_text.strip()

        text_chunks = text_splitter.split_text(normalized_text)
        num_chunks = len(text_chunks)
        print(f"[F5TTS Stream] Text split into {num_chunks} chunks.")

        if num_chunks == 0:
            print("Warning: Text resulted in zero chunks after splitting.")
            yield create_wave_header(tts_model.target_sample_rate)
            return

        try:
            # Use context manager for safe model access
            async with model_context(selected_ref_id) as model:
                sample_rate = model.target_sample_rate
                print(f"[F5TTS Stream] Starting audio stream generation at {sample_rate} Hz...")
                
                # Yield WAV header
                yield create_wave_header(sample_rate=sample_rate, data_size=0)
                print("[F5TTS Stream] WAV header yielded.")

                total_bytes_yielded = 0
                for i, chunk_text in enumerate(text_chunks):
                    chunk_num = i + 1
                    chunk_start_time = time.time()
                    print(f"[F5TTS Stream] Processing chunk {chunk_num}/{num_chunks}...")
                    
                    audio_bytes = process_chunk(chunk_text, model, request)

                    if audio_bytes and len(audio_bytes) > 0:
                        try:
                            yield audio_bytes
                            bytes_yielded = len(audio_bytes)
                            total_bytes_yielded += bytes_yielded
                            chunk_duration = time.time() - chunk_start_time
                            print(f"  [Chunk {chunk_num}] Yielded {bytes_yielded} bytes. Time: {chunk_duration:.3f}s")
                        except Exception as yield_e:
                            print(f"Error yielding audio bytes for chunk {chunk_num}: {yield_e}")
                            break
                    else:
                        print(f"  [Chunk {chunk_num}] Skipped yielding (no audio data generated).")

        except ValueError as ve:
            # Reference not ready error
            print(f"Error: {ve}")
            raise HTTPException(status_code=404, detail=str(ve))
        except Exception as e:
            print(f"Error in stream generation: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal error during audio generation: {str(e)}")

        final_delay = 0.05
        print(f"[F5TTS Stream] Finished processing all chunks. Adding final delay of {final_delay}s...")
        await asyncio.sleep(final_delay)

        end_time = time.time()
        total_duration = end_time - start_time
        print(f"[F5TTS Stream] Stream generation complete.")
        print(f"  Total audio bytes yielded (excluding header): {total_bytes_yielded}")
        print(f"  Total request processing time: {total_duration:.3f} seconds.")



async def process_and_cache_reference(file_path: str, text: Optional[str], ref_id: str):
    """Process a reference audio file in background and cache processed data with proper resource management."""
    global reference_cache, tts_model
    
    # Use background semaphore to limit concurrent background tasks
    async with background_semaphore:
        print(f"Background task: Processing reference {ref_id} from {file_path}")

        if tts_model is None:
            print(f"Background task error: Model not loaded. Cannot process reference {ref_id}.")
            async with reference_processing_lock:
                reference_cache[ref_id] = {
                    "ref_audio_path": file_path,
                    "loaded": False,
                    "error": "TTS Model not available during processing.",
                    "name": f"Custom {ref_id.split('_')[-1]} (Error)"
                }
            return

        try:
            normalized_text = TTSnorm(text).strip() if text else ""

            # Use model lock for thread safety
            async with model_lock:
                try:
                    _, processed_ref_text = tts_model.preprocess_reference(
                        ref_audio_path=file_path,
                        ref_text=normalized_text,
                        clip_short=True
                    )

                    cached_mel = tts_model.ref_audio_processed.clone().detach()
                    cached_mel_len = tts_model.ref_audio_len
                    cached_text = tts_model.ref_text

                    # Update cache with lock protection
                    async with reference_processing_lock:
                        reference_cache[ref_id].update({
                            "ref_text_original": text,
                            "loaded": True,
                            "processed_mel": cached_mel,
                            "processed_text": cached_text,
                            "processed_mel_len": cached_mel_len,
                            "error": None,
                        })
                    
                    print(f"Background task: Successfully processed and cached reference '{ref_id}'. Mel shape: {cached_mel.shape}, Len: {cached_mel_len}")

                finally:
                    # Always cleanup model state
                    if tts_model:
                        tts_model.ref_audio_processed = None
                        tts_model.ref_text = None
                        tts_model.ref_audio_len = None
                        
                        # Force GPU memory cleanup
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

        except Exception as e:
            print(f"Background task error: Processing reference {ref_id} failed: {e}")
            traceback.print_exc()
            
            async with reference_processing_lock:
                reference_cache[ref_id].update({
                    "loaded": False,
                    "error": str(e),
                    "name": f"{reference_cache[ref_id].get('name', ref_id)} (Error)"
                })

# --- API Endpoints ---
@app.post("/tts/stream", dependencies=[PROTECTED])
async def tts_stream(request: TTSRequest):
    """F5TTS streaming endpoint"""
    try:
        headers = {
            "Content-Type": "audio/wav",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        return StreamingResponse(
            stream_audio_generator(request),
            media_type="audio/wav",
            headers=headers
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unhandled error in /tts/stream endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error generating audio stream: {str(e)}"
        )

@app.post("/tts/stream", dependencies=[PROTECTED])
async def generate_speech_openai(request_data: OpenAITTSRequest):
    """OpenAI-compatible TTS endpoint using F5TTS directly"""
    global tts_model, reference_cache, DEFAULT_SAMPLE_RATE
    
    # Check if voice exists in our F5TTS reference cache
    if request_data.voice not in reference_cache:
        raise HTTPException(status_code=400, detail=f"Voice '{request_data.voice}' not found in F5TTS references.")
    
    # Check if the voice is ready
    cached_ref_data = reference_cache.get(request_data.voice)
    if not cached_ref_data or cached_ref_data.get("loaded") != True:
        status = cached_ref_data.get('loaded', 'Not Found') if cached_ref_data else 'Not Found'
        error_msg = cached_ref_data.get('error') if cached_ref_data else 'N/A'
        error_detail = f"Reference voice '{request_data.voice}' is not ready. Status: {status}."
        if status == 'processing':
            error_detail += " Still processing, please wait."
        elif error_msg:
            error_detail += f" Error: {error_msg}"
        raise HTTPException(status_code=503, detail=error_detail)

    if tts_model is None:
        raise HTTPException(status_code=503, detail="F5TTS model is not initialized.")

    input_text = request_data.input
    if not input_text or not input_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    response_format = request_data.response_format.lower() if request_data.response_format else "wav"
    if response_format not in ["pcm", "wav"]:
        response_format = "wav"

    print(f"[OpenAI API] Processing request with voice '{request_data.voice}' for text: '{input_text[:100]}...'")

    try:
        # Create a TTSRequest object for F5TTS processing
        f5tts_request = TTSRequest(
            text=input_text,
            speaker=request_data.voice,
            nfe_step=32,  # Default values - could be made configurable
            cfg_strength=2.0,
            speed=1.0,
            cross_fade_duration=0.15,
            sway_sampling_coef=-1.0
        )

        if response_format == "pcm":
            print("[OpenAI API] Streaming raw PCM audio using F5TTS.")
            media_type = f"audio/L16;rate={DEFAULT_SAMPLE_RATE};channels=1"
            
            async def pcm_byte_stream_generator():
                async for audio_bytes in stream_audio_generator(f5tts_request):
                    # Skip WAV header and yield raw PCM data
                    if len(audio_bytes) == 44:  # WAV header size
                        continue
                    yield audio_bytes
                print("[OpenAI API] Finished streaming PCM bytes.")
            
            return StreamingResponse(pcm_byte_stream_generator(), media_type=media_type)

        elif response_format == "wav":
            print("[OpenAI API] Generating WAV file using F5TTS.")
            
            # Use the existing stream_audio_generator which returns WAV format
            return StreamingResponse(
                stream_audio_generator(f5tts_request),
                media_type="audio/wav"
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in OpenAI-compatible endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/upload_reference", dependencies=[PROTECTED])
async def upload_reference(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    text: Optional[str] = Form(None)
):
    """Upload a custom reference audio file for processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    timestamp = int(time.time())
    ref_id = f"custom_{timestamp}"
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {file_extension}")

    ref_path = os.path.join(CUSTOM_REF_PATH, f"{ref_id}{file_extension}")
    print(f"Receiving file upload: {file.filename}, saving as {ref_path}")

    try:
        file_content = await file.read()
        with open(ref_path, "wb") as buffer:
            buffer.write(file_content)
        print(f"File saved successfully: {ref_path}")

        background_tasks.add_task(
            process_and_cache_reference,
            ref_path,
            text,
            ref_id
        )

        reference_cache[ref_id] = {
            "ref_audio_path": ref_path,
            "loaded": "processing",
            "name": f"Custom {timestamp} (Processing...)"
        }

        return {
            "status": "processing",
            "message": "Reference audio uploaded. Processing in background.",
            "ref_id": ref_id,
             "estimated_name": reference_cache[ref_id]["name"]
        }
    except Exception as e:
        print(f"Error saving/uploading reference file: {e}")
        traceback.print_exc()

        print(f"Attempting cleanup of potentially partial file: {ref_path}")
        try:
            if os.path.exists(ref_path):
                os.remove(ref_path)
                print(f"Successfully removed potentially partial file: {ref_path}")
            else:
                print(f"File {ref_path} did not exist, no cleanup needed.")
        except OSError as remove_error:
            print(f"Warning: Failed to remove partially saved file {ref_path} during cleanup: {remove_error}")

        raise HTTPException(status_code=500, detail=f"Error saving reference file: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_client():
    """Serve the HTML client page."""
    try:
        with open("client.html", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
         raise HTTPException(status_code=404, detail="Client HTML file not found.")

@app.get("/references")
async def get_references():
    """Get available reference presets (default and custom)."""
    available_refs = {
        ref_id: {"name": data.get("name", ref_id), "status": data.get("loaded", "error")}
        for ref_id, data in reference_cache.items()
        if data.get("loaded") == True or data.get("loaded") == "processing"
    }
    return available_refs

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if tts_model is None:
        return {"status": "error", "message": "F5TTS model not initialized"}
    
    loaded_refs = sum(1 for data in reference_cache.values() if data.get("loaded") == True)
    if loaded_refs == 0:
         return {"status": "warning", "message": "Model loaded, but no reference voices are ready."}
    
    return {
        "status": "ok", 
        "message": "F5TTS model ready.", 
        "loaded_references": loaded_refs,
        "available_voices": list(reference_cache.keys())
    }

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="F5TTS Server with OpenAI Compatible API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=6008, help="Port to bind the server to")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 F5TTS SERVER WITH OPENAI COMPATIBLE API")
    print("=" * 80)
    print(f"🌐 Web Interface:       http://{args.host}:{args.port}/")
    print(f"🎤 F5TTS Streaming:     http://{args.host}:{args.port}/tts/stream")
    print(f"🤖 OpenAI Compatible:   http://{args.host}:{args.port}/audio/speech")
    print(f"📁 Reference Upload:    http://{args.host}:{args.port}/upload_reference")
    print(f"❤️  Health Check:       http://{args.host}:{args.port}/health")
    print(f"📋 Available Refs:      http://{args.host}:{args.port}/references")
    print("=" * 80)
    
    uvicorn.run(app, host=args.host, port=args.port)
