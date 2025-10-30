"""
Middleware for logging, monitoring, and request processing.

Implements:
- Structured JSON logging
- Request/response logging
- Prometheus metrics
- Error tracking with Sentry
- CORS handling
"""

import time
import logging
import json
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge
import sentry_sdk

from config import settings

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(message)s'
)


class JSONFormatter(logging.Formatter):
    """Format logs as JSON."""
    
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        return json.dumps(log_data)


# Configure JSON logging for production
if not settings.debug:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.root.handlers = [handler]


# Prometheus metrics
REQUEST_COUNT = Counter(
    'myai_requests_total',
    'Total request count',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'myai_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

INFERENCE_COUNT = Counter(
    'myai_inference_total',
    'Total inference count',
    ['model', 'status']
)

INFERENCE_DURATION = Histogram(
    'myai_inference_duration_seconds',
    'Inference duration in seconds',
    ['model']
)

INFERENCE_TOKENS = Histogram(
    'myai_inference_tokens',
    'Number of tokens generated',
    ['model']
)

ACTIVE_REQUESTS = Gauge(
    'myai_active_requests',
    'Number of active requests'
)

MODEL_LOADED = Gauge(
    'myai_model_loaded',
    'Whether model is loaded',
    ['model']
)


# Initialize Sentry if DSN is provided
if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment="production" if not settings.debug else "development",
        traces_sample_rate=0.1,
    )


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = request.headers.get('X-Request-ID', str(time.time()))
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        logger = logging.getLogger(__name__)
        logger.info(
            f"Request started",
            extra={
                'request_id': request_id,
                'method': request.method,
                'path': request.url.path,
                'client': request.client.host if request.client else None,
            }
        )
        
        # Track active requests
        ACTIVE_REQUESTS.inc()
        
        # Process request
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            # Log response
            logger.info(
                f"Request completed",
                extra={
                    'request_id': request_id,
                    'status': response.status_code,
                    'duration': duration,
                }
            )
            
            # Add request ID to response headers
            response.headers['X-Request-ID'] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {str(e)}",
                extra={
                    'request_id': request_id,
                    'duration': duration,
                },
                exc_info=True
            )
            
            # Record error metric
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            
            raise
            
        finally:
            ACTIVE_REQUESTS.dec()


def setup_cors(app):
    """Setup CORS middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_middleware(app):
    """Setup all middleware."""
    app.add_middleware(LoggingMiddleware)
    setup_cors(app)
