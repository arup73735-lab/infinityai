"""
Configuration management for MyAI backend.

This module handles all configuration settings using Pydantic Settings,
supporting environment variables and .env files.
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "MyAI"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Model
    model_name: str = Field(
        default="facebook/opt-125m",
        description="HuggingFace model identifier"
    )
    model_cache_dir: str = Field(
        default="/root/.cache/huggingface",
        description="Directory to cache downloaded models"
    )
    device: str = Field(
        default="cpu",
        description="Device to run inference on: cpu, cuda, cuda:0, etc."
    )
    max_length: int = Field(
        default=512,
        description="Maximum sequence length for generation"
    )
    
    # Performance
    enable_quantization: bool = Field(
        default=False,
        description="Enable INT8 quantization for faster inference"
    )
    batch_size: int = Field(
        default=8,
        description="Maximum batch size for inference"
    )
    batch_timeout: float = Field(
        default=0.1,
        description="Maximum time to wait for batch to fill (seconds)"
    )
    num_workers: int = Field(
        default=1,
        description="Number of worker processes"
    )
    
    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    
    # Security
    jwt_secret_key: str = Field(
        default="change-this-in-production-use-openssl-rand-hex-32",
        description="Secret key for JWT token generation"
    )
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Rate limiting
    rate_limit_requests: int = Field(
        default=100,
        description="Maximum requests per window"
    )
    rate_limit_window: int = Field(
        default=60,
        description="Rate limit window in seconds"
    )
    
    # Safety & Moderation
    enable_strict_moderation: bool = Field(
        default=True,
        description="Enable content moderation filters"
    )
    max_input_length: int = Field(
        default=2048,
        description="Maximum input length in characters"
    )
    blocked_words: List[str] = Field(
        default_factory=list,
        description="List of blocked words for content filtering"
    )
    
    # Monitoring
    prometheus_enabled: bool = True
    sentry_dsn: Optional[str] = None
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
