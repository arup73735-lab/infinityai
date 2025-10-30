"""
Model loader and worker tests.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch

from model_loader import ModelLoader
from safety import ContentModerator, ModerationResult


class TestModelLoader:
    """Test model loader functionality."""
    
    def test_detect_device_cuda(self):
        """Test CUDA device detection."""
        with patch('torch.cuda.is_available', return_value=True):
            loader = ModelLoader()
            # Device detection happens in __init__
            assert loader._device in ["cuda", "cpu", "auto"]
    
    def test_detect_device_cpu(self):
        """Test CPU fallback."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            loader = ModelLoader()
            assert loader._device == "cpu"
    
    def test_get_model_info_not_loaded(self):
        """Test getting info when model not loaded."""
        loader = ModelLoader()
        info = loader.get_model_info()
        assert info["status"] == "not_loaded"
    
    @patch('model_loader.AutoModelForCausalLM')
    @patch('model_loader.AutoTokenizer')
    def test_load_model(self, mock_tokenizer, mock_model):
        """Test model loading."""
        # Mock tokenizer
        mock_tok = Mock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "[EOS]"
        mock_tok.pad_token_id = 0
        mock_tok.eos_token_id = 1
        mock_tokenizer.from_pretrained.return_value = mock_tok
        
        # Mock model
        mock_mdl = Mock()
        mock_mdl.parameters.return_value = [torch.zeros(100)]
        mock_mdl.eval.return_value = None
        mock_mdl.to.return_value = mock_mdl
        mock_model.from_pretrained.return_value = mock_mdl
        
        loader = ModelLoader()
        loader.load_model()
        
        assert loader._model is not None
        assert loader._tokenizer is not None
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()


class TestContentModerator:
    """Test content moderation."""
    
    def test_moderate_input_safe(self):
        """Test safe input moderation."""
        moderator = ContentModerator()
        result = moderator.moderate_input("Hello, how are you?")
        assert result.is_safe
    
    def test_moderate_input_empty(self):
        """Test empty input."""
        moderator = ContentModerator()
        result = moderator.moderate_input("")
        assert not result.is_safe
        assert "empty" in result.reason.lower()
    
    def test_moderate_input_too_long(self):
        """Test input that's too long."""
        moderator = ContentModerator()
        long_text = "a" * 10000
        result = moderator.moderate_input(long_text)
        assert not result.is_safe
        assert "too long" in result.reason.lower()
    
    def test_moderate_input_toxic(self):
        """Test toxic content detection."""
        moderator = ContentModerator()
        result = moderator.moderate_input("I want to harm someone")
        assert not result.is_safe
        assert "toxic" in result.reason.lower()
    
    def test_sanitize_input(self):
        """Test input sanitization."""
        moderator = ContentModerator()
        text = "Hello!!!!!!   World\x00\x01"
        sanitized = moderator.sanitize_input(text)
        assert "\x00" not in sanitized
        assert "\x01" not in sanitized
        assert "!!!" in sanitized  # Should reduce but not remove
    
    def test_validate_generation_params_valid(self):
        """Test valid generation parameters."""
        moderator = ContentModerator()
        is_valid, error = moderator.validate_generation_params(
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        assert is_valid
        assert error is None
    
    def test_validate_generation_params_invalid_tokens(self):
        """Test invalid max_new_tokens."""
        moderator = ContentModerator()
        is_valid, error = moderator.validate_generation_params(
            max_new_tokens=5000,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        assert not is_valid
        assert "max_new_tokens" in error
    
    def test_validate_generation_params_invalid_temperature(self):
        """Test invalid temperature."""
        moderator = ContentModerator()
        is_valid, error = moderator.validate_generation_params(
            max_new_tokens=100,
            temperature=5.0,
            top_p=0.9,
            top_k=50
        )
        assert not is_valid
        assert "temperature" in error
    
    def test_moderate_output_safe(self):
        """Test safe output moderation."""
        moderator = ContentModerator()
        result = moderator.moderate_output("This is a safe response.")
        assert result.is_safe
        assert result.filtered_text == "This is a safe response."
    
    def test_moderate_output_toxic(self):
        """Test toxic output filtering."""
        moderator = ContentModerator()
        result = moderator.moderate_output("I want to harm you")
        assert not result.is_safe
        assert "filtered" in result.filtered_text.lower()
    
    def test_detect_pii_email(self):
        """Test PII detection for email."""
        moderator = ContentModerator()
        pii = moderator._detect_pii("Contact me at test@example.com")
        assert "email" in pii
    
    def test_detect_pii_phone(self):
        """Test PII detection for phone."""
        moderator = ContentModerator()
        pii = moderator._detect_pii("Call me at 555-123-4567")
        assert "phone" in pii
    
    def test_filter_pii(self):
        """Test PII filtering."""
        moderator = ContentModerator()
        filtered = moderator._filter_pii("Email: test@example.com, Phone: 555-123-4567")
        assert "test@example.com" not in filtered
        assert "555-123-4567" not in filtered
        assert "EMAIL_REDACTED" in filtered
        assert "PHONE_REDACTED" in filtered


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_first_request(self):
        """Test first request within window."""
        from safety import check_rate_limit
        
        mock_redis = Mock()
        mock_redis.get.return_value = None
        
        is_allowed, remaining = check_rate_limit("user123", mock_redis)
        
        assert is_allowed
        assert remaining > 0
        mock_redis.setex.assert_called_once()
    
    def test_rate_limit_within_limit(self):
        """Test request within rate limit."""
        from safety import check_rate_limit
        
        mock_redis = Mock()
        mock_redis.get.return_value = "5"
        
        is_allowed, remaining = check_rate_limit("user123", mock_redis)
        
        assert is_allowed
        mock_redis.incr.assert_called_once()
    
    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded."""
        from safety import check_rate_limit
        from config import settings
        
        mock_redis = Mock()
        mock_redis.get.return_value = str(settings.rate_limit_requests)
        
        is_allowed, remaining = check_rate_limit("user123", mock_redis)
        
        assert not is_allowed
        assert remaining == 0
    
    def test_rate_limit_no_redis(self):
        """Test rate limiting with no Redis."""
        from safety import check_rate_limit
        
        is_allowed, remaining = check_rate_limit("user123", None)
        
        # Should fail open
        assert is_allowed
