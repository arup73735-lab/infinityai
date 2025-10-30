"""
Pytest configuration and fixtures for backend tests.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from main import app
from config import settings


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = Mock()
    mock.get.return_value = None
    mock.setex.return_value = True
    mock.incr.return_value = 1
    mock.ping.return_value = True
    return mock


@pytest.fixture
def mock_model_loader():
    """Mock model loader."""
    with patch('main.model_loader') as mock:
        mock.get_model_info.return_value = {
            "status": "loaded",
            "model_name": "test-model",
            "device": "cpu",
            "quantized": False,
            "num_parameters": 1000000,
        }
        yield mock


@pytest.fixture
def auth_headers(client):
    """Get authentication headers."""
    response = client.post(
        "/token",
        data={"username": "demo", "password": "demo123"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(client):
    """Get admin authentication headers."""
    response = client.post(
        "/token",
        data={"username": "admin", "password": "admin123"}
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
