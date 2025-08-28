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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
import gc

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
    if not chunk_text or chunk_text == ".":
        return None
    if chunk_text.endswith(".."):
        chunk_text = chunk_text[:-1].strip()
    if not chunk_text:
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
            
            tts_model.ref_audio_processed = cached_ref_data["processed_mel"]
            tts_model.ref_text = cached_ref_data["processed_text"]
            tts_model.ref_audio_len = cached_ref_data["processed_mel_len"]
            
            if tts_model.ref_audio_processed.device != tts_model.device:
                tts_model.ref_audio_processed = tts_model.ref_audio_processed.to(tts_model.device)
            
            yield tts_model
        finally:
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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=TEXT_SPLITTER_CHUNK_SIZE, chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP,
    length_function=len, separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    keep_separator=True
)

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
                if os.path.exists(audio_path):
                    async with model_lock:
                        try:
                            _, _ = tts_model.preprocess_reference(
                                ref_audio_path=audio_path,
                                ref_text=TTSnorm(ref_data.get("text", "")).strip(),
                                clip_short=False
                            )
                            reference_cache[ref_id] = {
                                "ref_audio_path": audio_path,
                                "ref_text_original": ref_data.get("text", ""), "loaded": True,
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
                    reference_cache[ref_id] = {"loaded": False, "name": ref_data.get("name", ref_id), "error": "File not found"}
            except Exception as e:
                reference_cache[ref_id] = {"loaded": False, "name": ref_data.get("name", ref_id), "error": str(e)}
    print("Default reference processing complete.")

# --- Audio Generation Functions ---
async def stream_audio_generator(request: TTSRequest) -> AsyncGenerator[bytes, None]:
    """F5TTS audio stream generator with resource management."""
    start_time = time.time()
    async with request_semaphore:
        if tts_model is None: raise HTTPException(503, "TTS model is not ready.")
        if not request.text or not request.text.strip(): raise HTTPException(400, "Input text cannot be empty.")
        
        try:
            normalized_text = TTSnorm(request.text).strip()
        except Exception: normalized_text = request.text.strip()
        
        text_chunks = text_splitter.split_text(normalized_text)
        if not text_chunks:
            yield create_wave_header(tts_model.target_sample_rate)
            return

        try:
            async with model_context(request.speaker or "male") as model:
                yield create_wave_header(sample_rate=model.target_sample_rate, data_size=0)
                for chunk_text in text_chunks:
                    audio_bytes = process_chunk(chunk_text, model, request)
                    if audio_bytes: yield audio_bytes
        except ValueError as ve: raise HTTPException(404, str(ve))
        except Exception as e: raise HTTPException(500, f"Internal error during audio generation: {e}")
        
        await asyncio.sleep(0.05)
        print(f"[F5TTS Stream] Request processed in {time.time() - start_time:.3f}s.")

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
    if request_data.voice not in reference_cache:
        raise HTTPException(400, f"Voice '{request_data.voice}' not found.")
    
    cached_ref_data = reference_cache[request_data.voice]
    if cached_ref_data.get("loaded") != True:
        raise HTTPException(503, f"Reference voice '{request_data.voice}' is not ready.")

    f5tts_request = TTSRequest(text=request_data.input, speaker=request_data.voice)
    
    response_format = request_data.response_format.lower() if request_data.response_format else "wav"
    
    if response_format == "wav":
        return StreamingResponse(stream_audio_generator(f5tts_request), media_type="audio/wav")
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
        raise HTTPException(400, f"Unsupported response format: {response_format}")


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