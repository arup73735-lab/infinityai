"""
Content moderation and safety filters.

Implements multiple layers of safety:
- Input validation and sanitization
- Toxic content detection
- Blocked word filtering
- Output post-processing
- Rate limiting per user

Design rationale:
- Defense in depth: Multiple layers of protection
- Configurable: Can be enabled/disabled via settings
- Extensible: Easy to add new filters or integrate external APIs
"""

import re
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModerationResult:
    """Result of content moderation."""
    is_safe: bool
    reason: Optional[str] = None
    filtered_text: Optional[str] = None


class ContentModerator:
    """Handles content moderation and safety checks."""
    
    # Common toxic patterns (simplified - use proper toxicity classifier in production)
    TOXIC_PATTERNS = [
        r'\b(hate|kill|attack|harm|violence)\b',
        r'\b(racist|sexist|homophobic)\b',
        r'\b(illegal|crime|weapon|drug)\b',
    ]
    
    # PII patterns
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }
    
    def __init__(self):
        self.enabled = settings.enable_strict_moderation
        self.blocked_words = set(settings.blocked_words)
        
    def moderate_input(self, text: str) -> ModerationResult:
        """Moderate user input before processing."""
        if not self.enabled:
            return ModerationResult(is_safe=True)
        
        # Check length
        if len(text) > settings.max_input_length:
            return ModerationResult(
                is_safe=False,
                reason=f"Input too long (max {settings.max_input_length} characters)"
            )
        
        # Check for empty input
        if not text.strip():
            return ModerationResult(
                is_safe=False,
                reason="Empty input"
            )
        
        # Check for blocked words
        text_lower = text.lower()
        for word in self.blocked_words:
            if word.lower() in text_lower:
                return ModerationResult(
                    is_safe=False,
                    reason=f"Blocked word detected: {word}"
                )
        
        # Check for toxic patterns
        for pattern in self.TOXIC_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Toxic pattern detected: {pattern}")
                return ModerationResult(
                    is_safe=False,
                    reason="Potentially toxic content detected"
                )
        
        # Check for PII
        pii_found = self._detect_pii(text)
        if pii_found:
            logger.warning(f"PII detected: {pii_found}")
            return ModerationResult(
                is_safe=False,
                reason=f"Personal information detected: {', '.join(pii_found)}"
            )
        
        return ModerationResult(is_safe=True)
    
    def moderate_output(self, text: str) -> ModerationResult:
        """Moderate model output before returning to user."""
        if not self.enabled:
            return ModerationResult(is_safe=True, filtered_text=text)
        
        # Check for toxic patterns
        for pattern in self.TOXIC_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning(f"Toxic output detected: {pattern}")
                return ModerationResult(
                    is_safe=False,
                    reason="Model generated potentially toxic content",
                    filtered_text="[Content filtered due to safety policies]"
                )
        
        # Filter PII from output
        filtered_text = self._filter_pii(text)
        
        return ModerationResult(
            is_safe=True,
            filtered_text=filtered_text
        )
    
    def _detect_pii(self, text: str) -> List[str]:
        """Detect personally identifiable information."""
        found = []
        
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, text):
                found.append(pii_type)
        
        return found
    
    def _filter_pii(self, text: str) -> str:
        """Filter PII from text."""
        filtered = text
        
        for pii_type, pattern in self.PII_PATTERNS.items():
            filtered = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]', filtered)
        
        return filtered
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input."""
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){4,}', r'\1\1\1', text)
        
        return text.strip()
    
    def validate_generation_params(
        self,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> Tuple[bool, Optional[str]]:
        """Validate generation parameters."""
        if max_new_tokens < 1 or max_new_tokens > 2048:
            return False, "max_new_tokens must be between 1 and 2048"
        
        if temperature < 0.0 or temperature > 2.0:
            return False, "temperature must be between 0.0 and 2.0"
        
        if top_p < 0.0 or top_p > 1.0:
            return False, "top_p must be between 0.0 and 1.0"
        
        if top_k < 0 or top_k > 1000:
            return False, "top_k must be between 0 and 1000"
        
        return True, None


# Global moderator instance
content_moderator = ContentModerator()


def check_rate_limit(user_id: str, redis_client) -> Tuple[bool, int]:
    """
    Check if user has exceeded rate limit.
    
    Returns:
        Tuple of (is_allowed, remaining_requests)
    """
    if redis_client is None:
        return True, settings.rate_limit_requests
    
    key = f"rate_limit:{user_id}"
    
    try:
        # Get current count
        count = redis_client.get(key)
        
        if count is None:
            # First request in window
            redis_client.setex(
                key,
                settings.rate_limit_window,
                1
            )
            return True, settings.rate_limit_requests - 1
        
        count = int(count)
        
        if count >= settings.rate_limit_requests:
            return False, 0
        
        # Increment count
        redis_client.incr(key)
        return True, settings.rate_limit_requests - count - 1
        
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        # Fail open - allow request if rate limiting fails
        return True, settings.rate_limit_requests
