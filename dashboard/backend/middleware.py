"""
Security and Rate Limiting Middleware
API protection and request throttling
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional
import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware

    Implements token bucket algorithm for rate limiting
    """

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)
        self.cleanup_interval = timedelta(minutes=5)
        self.last_cleanup = datetime.now()

    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP address or API key)
        client_id = self._get_client_id(request)

        # Clean old requests periodically
        if datetime.now() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests()

        # Check rate limit
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Remove requests older than 1 minute
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > minute_ago
        ]

        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute",
                    "retry_after": 60
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0"
                }
            )

        # Add current request
        self.requests[client_id].append(now)

        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(self.requests[client_id])
        )

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier"""
        # Check for API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key_{api_key}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host

        return f"ip_{ip}"

    def _cleanup_old_requests(self):
        """Remove old request records"""
        cutoff = datetime.now() - timedelta(minutes=5)

        for client_id in list(self.requests.keys()):
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > cutoff
            ]

            if not self.requests[client_id]:
                del self.requests[client_id]

        self.last_cleanup = datetime.now()


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    API Key authentication middleware

    Validates API keys for protected endpoints
    """

    def __init__(self, app, api_keys: Optional[Dict[str, Dict]] = None):
        super().__init__(app)
        # api_keys: Dict[key -> {name, permissions, rate_limit}]
        self.api_keys = api_keys or {}

        # Public endpoints that don't require API key
        self.public_paths = [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/health"
        ]

    async def dispatch(self, request: Request, call_next):
        # Check if path is public
        if any(request.url.path.startswith(path) for path in self.public_paths):
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Missing API key",
                    "message": "Please provide X-API-Key header"
                }
            )

        # Validate API key
        if api_key not in self.api_keys:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "Invalid API key",
                    "message": "The provided API key is invalid"
                }
            )

        # Add API key info to request state
        request.state.api_key_info = self.api_keys[api_key]

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests"""

    async def dispatch(self, request: Request, call_next):
        start_time = datetime.now()

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host}"
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Log response
        logger.info(
            f"Response: {response.status_code} "
            f"in {duration:.3f}s"
        )

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration:.3f}s"

        return response


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    IP whitelist middleware

    Only allows requests from whitelisted IPs
    """

    def __init__(self, app, whitelist: Optional[list] = None):
        super().__init__(app)
        self.whitelist = set(whitelist or [])

    async def dispatch(self, request: Request, call_next):
        if not self.whitelist:
            # No whitelist configured, allow all
            return await call_next(request)

        # Get client IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host

        # Check whitelist
        if client_ip not in self.whitelist:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "Access denied",
                    "message": "Your IP address is not whitelisted"
                }
            )

        return await call_next(request)


def generate_api_key(name: str) -> str:
    """
    Generate a secure API key

    Args:
        name: Name/identifier for the API key

    Returns:
        Secure API key string
    """
    # Generate random bytes
    random_bytes = secrets.token_bytes(32)

    # Create hash with name
    combined = f"{name}:{random_bytes.hex()}".encode()
    hash_obj = hashlib.sha256(combined)

    # Format as API key
    api_key = f"aidao_{hash_obj.hexdigest()[:40]}"

    return api_key


class CORSConfig:
    """CORS configuration for different environments"""

    @staticmethod
    def development():
        return {
            "allow_origins": ["http://localhost:3000", "http://localhost:5173"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }

    @staticmethod
    def production(allowed_origins: list):
        return {
            "allow_origins": allowed_origins,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["Content-Type", "Authorization", "X-API-Key"],
            "expose_headers": ["X-RateLimit-Limit", "X-RateLimit-Remaining"],
        }


if __name__ == "__main__":
    # Example: Generate API keys
    print("=== API Key Generation ===\n")

    api_keys = {}

    for user in ["admin", "dashboard", "mobile_app"]:
        key = generate_api_key(user)
        api_keys[key] = {
            "name": user,
            "created_at": datetime.now().isoformat(),
            "permissions": ["read", "write"],
            "rate_limit": 100 if user == "admin" else 60
        }
        print(f"{user}: {key}")

    print(f"\n Generated {len(api_keys)} API keys")

    # Example rate limits
    print("\n=== Rate Limit Configuration ===\n")
    print("Standard: 60 requests/minute")
    print("Premium: 100 requests/minute")
    print("Admin: 1000 requests/minute")
