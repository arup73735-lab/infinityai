"""
API endpoint tests.
"""

import pytest
from fastapi import status
from unittest.mock import patch, AsyncMock


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client, mock_model_loader):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "model_info" in data
    
    def test_liveness_check(self, client):
        """Test liveness check."""
        response = client.get("/health/live")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "alive"
    
    def test_readiness_check(self, client, mock_model_loader):
        """Test readiness check."""
        response = client.get("/health/ready")
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "ready"


class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    def test_login_success(self, client):
        """Test successful login."""
        response = client.post(
            "/token",
            data={"username": "demo", "password": "demo123"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = client.post(
            "/token",
            data={"username": "demo", "password": "wrong"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_current_user(self, client, auth_headers):
        """Test getting current user info."""
        response = client.get("/users/me", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["username"] == "demo"
    
    def test_get_current_user_unauthorized(self, client):
        """Test getting user info without auth."""
        response = client.get("/users/me")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestGenerateEndpoint:
    """Test text generation endpoint."""
    
    @pytest.mark.asyncio
    async def test_generate_success(self, client, auth_headers, mock_redis):
        """Test successful text generation."""
        with patch('main.app.state.redis', mock_redis), \
             patch('main.batch_worker.submit', new_callable=AsyncMock) as mock_submit:
            
            # Mock response
            mock_response = AsyncMock()
            mock_response.request_id = "test-123"
            mock_response.text = "Generated text"
            mock_response.tokens = 10
            mock_response.latency = 0.5
            mock_response.error = None
            mock_submit.return_value = mock_response
            
            response = client.post(
                "/generate",
                json={
                    "prompt": "Test prompt",
                    "max_new_tokens": 50,
                    "temperature": 0.7
                },
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "request_id" in data
            assert "text" in data
            assert "tokens" in data
            assert "latency" in data
    
    def test_generate_invalid_params(self, client, auth_headers):
        """Test generation with invalid parameters."""
        response = client.post(
            "/generate",
            json={
                "prompt": "Test",
                "max_new_tokens": 5000,  # Too large
                "temperature": 0.7
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_generate_empty_prompt(self, client, auth_headers):
        """Test generation with empty prompt."""
        response = client.post(
            "/generate",
            json={
                "prompt": "",
                "max_new_tokens": 50
            },
            headers=auth_headers
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_generate_rate_limit(self, client, auth_headers, mock_redis):
        """Test rate limiting."""
        # Mock rate limit exceeded
        mock_redis.get.return_value = "1000"
        
        with patch('main.app.state.redis', mock_redis):
            response = client.post(
                "/generate",
                json={
                    "prompt": "Test prompt",
                    "max_new_tokens": 50
                },
                headers=auth_headers
            )
            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS


class TestAdminEndpoints:
    """Test admin endpoints."""
    
    def test_load_model_admin(self, client, admin_headers):
        """Test loading model as admin."""
        with patch('main.model_loader.load_model'):
            response = client.post("/admin/model/load", headers=admin_headers)
            assert response.status_code == status.HTTP_200_OK
    
    def test_load_model_non_admin(self, client, auth_headers):
        """Test loading model as non-admin."""
        response = client.post("/admin/model/load", headers=auth_headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_get_model_info_admin(self, client, admin_headers, mock_model_loader):
        """Test getting model info as admin."""
        response = client.get("/admin/model/info", headers=admin_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
