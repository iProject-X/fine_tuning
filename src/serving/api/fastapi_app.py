"""
FastAPI application for serving multilingual Whisper ASR
Supports real-time streaming, batch processing, and WebSocket connections
"""

import asyncio
import io
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, AsyncGenerator
from pathlib import Path
import tempfile
import os

import torch
import numpy as np
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis
from pydantic import BaseModel, Field
import uvicorn

from ...models.architectures.whisper_multilingual import MultilingualWhisperForConditionalGeneration
from ...data.processors.code_switch_detector import CodeSwitchDetector
from ...utils.audio_utils import AudioProcessor
from ...utils.monitoring import MetricsCollector, track_request
from ...utils.config import ServingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class TranscriptionRequest(BaseModel):
    """Request model for transcription"""
    language_hint: Optional[str] = Field(None, description="Language hint: uz, ru, or mixed")
    enable_language_detection: bool = Field(True, description="Enable automatic language detection")
    enable_code_switch_detection: bool = Field(True, description="Enable code-switching detection")
    return_segments: bool = Field(False, description="Return segment-level results")
    temperature: float = Field(0.0, description="Sampling temperature")
    beam_size: int = Field(5, description="Beam search size")

class TranscriptionResponse(BaseModel):
    """Response model for transcription"""
    text: str
    language_detected: str
    confidence: float
    processing_time: float
    segments: Optional[List[Dict]] = None
    code_switch_analysis: Optional[Dict] = None

class StreamingTranscriptionChunk(BaseModel):
    """Streaming transcription chunk"""
    type: str  # "interim", "final", "error"
    text: str
    timestamp: float
    language: Optional[str] = None
    confidence: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, str]
    uptime: float

# Global objects
app = FastAPI(
    title="Uzbek Multilingual Whisper ASR",
    description="Production-ready ASR service for Uzbek-Russian code-switching",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

class ASRService:
    """
    Main ASR service class handling model inference and caching
    """

    def __init__(self, config: ServingConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.code_switch_detector = CodeSwitchDetector()
        self.audio_processor = AudioProcessor()
        self.metrics = MetricsCollector()

        # Redis for caching
        self.redis_client = None

        # WebSocket connections
        self.active_connections: Dict[str, WebSocket] = {}
        self.streaming_buffers: Dict[str, List[bytes]] = {}

        # Performance tracking
        self.start_time = time.time()

    async def initialize(self):
        """Initialize the ASR service"""
        logger.info("Initializing ASR service...")

        # Load model
        await self._load_model()

        # Setup Redis
        await self._setup_redis()

        logger.info("ASR service initialized successfully")

    async def _load_model(self):
        """Load the multilingual Whisper model"""
        try:
            model_path = self.config.model_path
            device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading model from {model_path} on {device}")

            # Load model
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                # Load PyTorch model
                self.model = torch.jit.load(model_path, map_location=device)
            else:
                # For demo, use standard Whisper model
                from transformers import WhisperForConditionalGeneration
                self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
                self.model.to(device)

            self.model.eval()

            # Load processor
            from transformers import WhisperProcessor
            self.processor = WhisperProcessor.from_pretrained(
                self.config.processor_path or "openai/whisper-base"
            )

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def _setup_redis(self):
        """Setup Redis connection for caching"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                decode_responses=True,
                max_connections=self.config.redis_max_connections
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis_client = None

    @track_request("transcribe_audio")
    async def transcribe_audio(
        self,
        audio_data: bytes,
        request: TranscriptionRequest
    ) -> TranscriptionResponse:
        """
        Transcribe audio file
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._get_cache_key(audio_data, request)
            if self.redis_client and cache_key:
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    logger.info("Cache hit for transcription request")
                    return TranscriptionResponse(**json.loads(cached_result))

            # Process audio
            audio_array = await self.audio_processor.process_audio_bytes(audio_data)

            # Run inference
            with torch.no_grad():
                # Prepare input
                inputs = self.processor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt",
                    truncation=True
                )

                # Generate transcription
                generated_ids = self.model.generate(
                    inputs.input_features.to(self.model.device),
                    language=request.language_hint,
                    task="transcribe",
                    temperature=request.temperature,
                    num_beams=request.beam_size,
                    max_length=448,
                    do_sample=request.temperature > 0
                )

                # Decode result
                transcription = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]

            # Language detection and code-switching analysis
            language_detected = "unknown"
            code_switch_analysis = None
            confidence = 1.0

            if request.enable_language_detection or request.enable_code_switch_detection:
                segments = self.code_switch_detector.detect_segments(transcription)
                if segments:
                    # Determine primary language
                    language_counts = {}
                    for seg in segments:
                        language_counts[seg.language] = language_counts.get(seg.language, 0) + seg.word_count

                    if language_counts:
                        language_detected = max(language_counts, key=language_counts.get)

                    # Calculate average confidence
                    confidence = sum(seg.confidence for seg in segments) / len(segments)

                    # Code-switching analysis
                    if request.enable_code_switch_detection:
                        code_switch_analysis = self.code_switch_detector.analyze_text_statistics(transcription)

            # Prepare response
            processing_time = time.time() - start_time

            response = TranscriptionResponse(
                text=transcription,
                language_detected=language_detected,
                confidence=confidence,
                processing_time=processing_time,
                code_switch_analysis=code_switch_analysis
            )

            # Segment-level results
            if request.return_segments and 'segments' in locals():
                response.segments = [
                    {
                        'text': seg.text,
                        'language': seg.language,
                        'confidence': seg.confidence,
                        'start_idx': seg.start_idx,
                        'end_idx': seg.end_idx
                    }
                    for seg in segments
                ]

            # Cache result
            if self.redis_client and cache_key:
                await self.redis_client.setex(
                    cache_key,
                    self.config.cache_ttl,
                    response.json()
                )

            # Update metrics
            self.metrics.record_transcription(
                processing_time=processing_time,
                language=language_detected,
                audio_duration=len(audio_array) / 16000
            )

            return response

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    async def transcribe_streaming(
        self,
        websocket: WebSocket,
        session_id: str
    ):
        """
        Handle streaming transcription via WebSocket
        """
        try:
            await websocket.accept()
            self.active_connections[session_id] = websocket
            self.streaming_buffers[session_id] = []

            logger.info(f"WebSocket connection established: {session_id}")

            while True:
                # Receive audio chunk
                try:
                    data = await websocket.receive_bytes()
                except Exception:
                    break

                # Add to buffer
                self.streaming_buffers[session_id].append(data)

                # Process when we have enough data (approximate 1 second)
                if len(self.streaming_buffers[session_id]) >= 10:
                    audio_chunk = b''.join(self.streaming_buffers[session_id])

                    try:
                        # Process chunk
                        audio_array = await self.audio_processor.process_audio_bytes(audio_chunk)

                        # Quick transcription for interim results
                        with torch.no_grad():
                            inputs = self.processor(
                                audio_array,
                                sampling_rate=16000,
                                return_tensors="pt",
                                truncation=True
                            )

                            generated_ids = self.model.generate(
                                inputs.input_features.to(self.model.device),
                                max_length=448,
                                num_beams=1,  # Faster for streaming
                                temperature=0.0,
                                do_sample=False
                            )

                            transcription = self.processor.batch_decode(
                                generated_ids,
                                skip_special_tokens=True
                            )[0]

                        # Send interim result
                        chunk_response = StreamingTranscriptionChunk(
                            type="interim",
                            text=transcription,
                            timestamp=time.time()
                        )

                        await websocket.send_text(chunk_response.json())

                        # Clear buffer (keep some overlap)
                        self.streaming_buffers[session_id] = self.streaming_buffers[session_id][-3:]

                    except Exception as e:
                        logger.error(f"Streaming transcription error: {e}")
                        error_response = StreamingTranscriptionChunk(
                            type="error",
                            text=f"Processing error: {str(e)}",
                            timestamp=time.time()
                        )
                        await websocket.send_text(error_response.json())

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Cleanup
            if session_id in self.active_connections:
                del self.active_connections[session_id]
            if session_id in self.streaming_buffers:
                del self.streaming_buffers[session_id]
            logger.info(f"WebSocket connection closed: {session_id}")

    def _get_cache_key(self, audio_data: bytes, request: TranscriptionRequest) -> Optional[str]:
        """Generate cache key for audio transcription"""
        try:
            import hashlib
            audio_hash = hashlib.md5(audio_data).hexdigest()
            request_hash = hashlib.md5(request.json().encode()).hexdigest()
            return f"transcription:{audio_hash}:{request_hash}"
        except Exception:
            return None

    async def get_health(self) -> HealthResponse:
        """Get service health status"""
        gpu_available = torch.cuda.is_available()
        model_loaded = self.model is not None

        memory_usage = {}
        if gpu_available:
            memory_usage["gpu_memory"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            memory_usage["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"

        import psutil
        memory_usage["system_memory"] = f"{psutil.virtual_memory().percent}%"

        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            gpu_available=gpu_available,
            memory_usage=memory_usage,
            uptime=time.time() - self.start_time
        )

# Initialize service
config = ServingConfig()  # Load from environment or config file
asr_service = ASRService(config)

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    await asr_service.initialize()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return await asr_service.get_health()

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    language_hint: Optional[str] = None,
    enable_language_detection: bool = True,
    enable_code_switch_detection: bool = True,
    return_segments: bool = False,
    temperature: float = 0.0,
    beam_size: int = 5,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Transcribe uploaded audio file
    """
    # Validate file
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be audio format")

    # Read file
    audio_data = await file.read()

    # Validate file size
    if len(audio_data) > config.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")

    # Create request
    request = TranscriptionRequest(
        language_hint=language_hint,
        enable_language_detection=enable_language_detection,
        enable_code_switch_detection=enable_code_switch_detection,
        return_segments=return_segments,
        temperature=temperature,
        beam_size=beam_size
    )

    return await asr_service.transcribe_audio(audio_data, request)

@app.post("/transcribe/batch")
async def transcribe_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Batch transcription endpoint
    """
    if len(files) > config.max_batch_size:
        raise HTTPException(status_code=400, detail=f"Batch size exceeds limit of {config.max_batch_size}")

    batch_id = str(uuid.uuid4())
    results = []

    for i, file in enumerate(files):
        if not file.content_type.startswith('audio/'):
            results.append({"error": f"File {i} is not audio format"})
            continue

        audio_data = await file.read()
        request = TranscriptionRequest()

        try:
            result = await asr_service.transcribe_audio(audio_data, request)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e)})

    return {
        "batch_id": batch_id,
        "results": results,
        "processed_count": len([r for r in results if "error" not in r])
    }

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming transcription
    """
    session_id = str(uuid.uuid4())
    await asr_service.transcribe_streaming(websocket, session_id)

@app.get("/models/info")
async def model_info():
    """Get information about loaded models"""
    return {
        "model_loaded": asr_service.model is not None,
        "processor_loaded": asr_service.processor is not None,
        "supported_languages": ["uz", "ru", "mixed"],
        "model_type": "multilingual_whisper",
        "features": [
            "code_switching_detection",
            "language_detection",
            "streaming_transcription",
            "batch_processing"
        ]
    }

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    return asr_service.metrics.get_metrics()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Uzbek Multilingual Whisper ASR",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "transcribe": "/transcribe",
            "batch": "/transcribe/batch",
            "streaming": "/ws/transcribe",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

def main():
    """Run the FastAPI server"""
    uvicorn.run(
        "src.serving.api.fastapi_app:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        reload=config.debug,
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()