"""
MyAI FastAPI Backend - Main Application

High-performance inference server with:
- REST API for batch inference
- WebSocket for streaming inference
- Authentication and authorization
- Content moderation
- Prometheus metrics
- Health checks
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List
from datetime import timedelta

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import make_asgi_app
import redis

from config import settings
from model_loader import model_loader
from worker import batch_worker, streaming_worker, InferenceRequest
from auth import (
    authenticate_user, create_access_token, get_current_user,
    get_current_active_user, Token, User
)
from safety import content_moderator, check_rate_limit
from middleware import setup_middleware, INFERENCE_COUNT, INFERENCE_DURATION, INFERENCE_TOKENS, MODEL_LOADED

logger = logging.getLogger(__name__)


# Request/Response Models
class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(100, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(50, ge=0, le=1000, description="Top-k sampling parameter")
    stream: bool = Field(False, description="Enable streaming response")


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    request_id: str
    text: str
    tokens: int
    latency: float
    model: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_info: dict


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting MyAI backend...")
    
    # Initialize Redis
    try:
        app.state.redis = redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True
        )
        app.state.redis.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        app.state.redis = None
    
    # Start batch worker
    await batch_worker.start()
    
    # Load model (lazy loading - will load on first request)
    logger.info("Model will be loaded on first request")
    
    logger.info("MyAI backend started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MyAI backend...")
    await batch_worker.stop()
    
    if app.state.redis:
        app.state.redis.close()
    
    logger.info("MyAI backend stopped")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="High-performance AI assistant inference server",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_info = model_loader.get_model_info()
    model_loaded = model_info.get("status") == "loaded"
    
    # Update Prometheus gauge
    MODEL_LOADED.labels(model=settings.model_name).set(1 if model_loaded else 0)
    
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        model_info=model_info
    )


@app.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    model_info = model_loader.get_model_info()
    if model_info.get("status") != "loaded":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return {"status": "ready"}


@app.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"status": "alive"}


# Authentication endpoints
@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint to get access token."""
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.username},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )


@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return current_user


# Inference endpoints
@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Generate text from a prompt.
    
    This endpoint supports both batch and streaming inference.
    For streaming, use the WebSocket endpoint instead.
    """
    # Check rate limit
    user_id = current_user.username if current_user else "anonymous"
    is_allowed, remaining = check_rate_limit(user_id, app.state.redis)
    
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Validate generation parameters
    is_valid, error_msg = content_moderator.validate_generation_params(
        request.max_new_tokens,
        request.temperature,
        request.top_p,
        request.top_k
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    
    # Sanitize and moderate input
    sanitized_prompt = content_moderator.sanitize_input(request.prompt)
    moderation_result = content_moderator.moderate_input(sanitized_prompt)
    
    if not moderation_result.is_safe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=moderation_result.reason
        )
    
    # Create inference request
    request_id = str(uuid.uuid4())
    inference_request = InferenceRequest(
        request_id=request_id,
        prompt=sanitized_prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream=request.stream
    )
    
    # Submit to worker
    try:
        response = await batch_worker.submit(inference_request)
        
        if response.error:
            INFERENCE_COUNT.labels(model=settings.model_name, status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=response.error
            )
        
        # Moderate output
        output_moderation = content_moderator.moderate_output(response.text)
        final_text = output_moderation.filtered_text or response.text
        
        # Record metrics
        INFERENCE_COUNT.labels(model=settings.model_name, status="success").inc()
        INFERENCE_DURATION.labels(model=settings.model_name).observe(response.latency)
        INFERENCE_TOKENS.labels(model=settings.model_name).observe(response.tokens)
        
        return GenerateResponse(
            request_id=response.request_id,
            text=final_text,
            tokens=response.tokens,
            latency=response.latency,
            model=settings.model_name
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        INFERENCE_COUNT.labels(model=settings.model_name, status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket endpoint for streaming text generation.
    
    Send JSON: {"prompt": "...", "max_new_tokens": 100, ...}
    Receive: Stream of generated tokens
    """
    await websocket.accept()
    
    try:
        # Receive request
        data = await websocket.receive_json()
        
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", 100)
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)
        top_k = data.get("top_k", 50)
        
        # Validate parameters
        is_valid, error_msg = content_moderator.validate_generation_params(
            max_new_tokens, temperature, top_p, top_k
        )
        
        if not is_valid:
            await websocket.send_json({"error": error_msg})
            await websocket.close()
            return
        
        # Sanitize and moderate input
        sanitized_prompt = content_moderator.sanitize_input(prompt)
        moderation_result = content_moderator.moderate_input(sanitized_prompt)
        
        if not moderation_result.is_safe:
            await websocket.send_json({"error": moderation_result.reason})
            await websocket.close()
            return
        
        # Stream generation
        async for token in streaming_worker.generate_stream(
            sanitized_prompt,
            max_new_tokens,
            temperature,
            top_p,
            top_k
        ):
            await websocket.send_json({"token": token})
        
        # Send completion signal
        await websocket.send_json({"done": True})
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


# Model management endpoints (admin only)
@app.post("/admin/model/load")
async def load_model(current_user: User = Depends(get_current_active_user)):
    """Load the model into memory."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        model_loader.load_model()
        return {"status": "success", "message": "Model loaded"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/admin/model/unload")
async def unload_model(current_user: User = Depends(get_current_active_user)):
    """Unload the model from memory."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    try:
        model_loader.unload_model()
        return {"status": "success", "message": "Model unloaded"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/admin/model/info")
async def get_model_info(current_user: User = Depends(get_current_active_user)):
    """Get detailed model information."""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return model_loader.get_model_info()


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )
