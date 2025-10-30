"""
Model loader with caching, quantization, and device management.

Design rationale:
- Lazy loading: Models are loaded on first request to reduce startup time
- Quantization support: INT8 quantization for 2-4x speedup with minimal quality loss
- Device management: Automatic GPU detection with CPU fallback
- Caching: Models are cached in memory to avoid repeated loading

Performance tradeoffs:
- Quantization: 2-4x faster inference, ~1-2% quality degradation
- Memory: Full precision ~4GB for 1B params, INT8 ~1GB
- Startup: Lazy loading adds 5-10s to first request but reduces startup time
"""

import os
import logging
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """Manages model loading, caching, and device placement."""
    
    def __init__(self):
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._device: str = self._detect_device()
        self._generation_config: Optional[GenerationConfig] = None
        
    def _detect_device(self) -> str:
        """Detect available device (CUDA, MPS, or CPU)."""
        if settings.device != "auto":
            return settings.device
            
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("MPS (Apple Silicon) available")
        else:
            device = "cpu"
            logger.info("Using CPU for inference")
            
        return device
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration if enabled."""
        if not settings.enable_quantization:
            return None
            
        if self._device == "cpu":
            logger.warning("Quantization not supported on CPU, disabling")
            return None
            
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    
    def load_model(self) -> None:
        """Load model and tokenizer with optimizations."""
        if self._model is not None:
            logger.info("Model already loaded")
            return
            
        logger.info(f"Loading model: {settings.model_name}")
        logger.info(f"Device: {self._device}")
        logger.info(f"Quantization: {settings.enable_quantization}")
        
        try:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.model_name,
                cache_dir=settings.model_cache_dir,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Get quantization config
            quantization_config = self._get_quantization_config()
            
            # Load model with optimizations
            model_kwargs = {
                "cache_dir": settings.model_cache_dir,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float16 if self._device != "cpu" else torch.float32
            
            self._model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if not quantization_config:
                self._model = self._model.to(self._device)
            
            # Set to eval mode
            self._model.eval()
            
            # Create generation config
            self._generation_config = GenerationConfig(
                max_length=settings.max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @property
    def model(self):
        """Get model, loading if necessary."""
        if self._model is None:
            self.load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Get tokenizer, loading if necessary."""
        if self._tokenizer is None:
            self.load_model()
        return self._tokenizer
    
    @property
    def device(self) -> str:
        """Get current device."""
        return self._device
    
    @property
    def generation_config(self) -> GenerationConfig:
        """Get generation configuration."""
        if self._generation_config is None:
            self.load_model()
        return self._generation_config
    
    def update_generation_config(self, **kwargs) -> None:
        """Update generation configuration parameters."""
        if self._generation_config is None:
            self.load_model()
        
        for key, value in kwargs.items():
            if hasattr(self._generation_config, key):
                setattr(self._generation_config, key, value)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        if self._model is None:
            return {"status": "not_loaded"}
        
        num_params = sum(p.numel() for p in self._model.parameters())
        num_trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        
        return {
            "status": "loaded",
            "model_name": settings.model_name,
            "device": self._device,
            "quantized": settings.enable_quantization,
            "num_parameters": num_params,
            "num_trainable_parameters": num_trainable,
            "dtype": str(next(self._model.parameters()).dtype),
            "memory_footprint_mb": num_params * 4 / (1024 ** 2),  # Approximate
        }
    
    def unload_model(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Model unloaded")


# Global model loader instance
model_loader = ModelLoader()
