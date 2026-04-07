"""
Authentication utilities for Spend Analyzer.
Provides decorators and helper functions for API authentication.
"""

import logging
from functools import wraps
from flask import request, jsonify
from config.settings import settings

logger = logging.getLogger(__name__)


def require_api_key(f):
    """
    Decorator to require API Key authentication on Flask routes.
    
    Usage:
        @app.route('/protected')
        @require_api_key
        def protected_route():
            return jsonify({"message": "authenticated"})
    
    Client usage:
        curl -H "X-API-Key: your-api-key" http://localhost:5000/protected
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not settings.ENABLE_AUTH:
            return f(*args, **kwargs)
        
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning(f"API request without API key from {request.remote_addr}")
            return jsonify({"error": "API key required"}), 401
        
        if api_key != settings.API_KEY:
            logger.warning(f"Invalid API key from {request.remote_addr}")
            return jsonify({"error": "Invalid API key"}), 403
        
        logger.info(f"Authenticated request from {request.remote_addr}")
        return f(*args, **kwargs)
    
    return decorated_function


def require_basic_auth(f):
    """
    Decorator to require HTTP Basic Authentication.
    
    Usage:
        @app.route('/admin')
        @require_basic_auth
        def admin_route():
            return jsonify({"message": "authenticated"})
    
    Client usage:
        curl -u username:password http://localhost:5000/admin
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not settings.ENABLE_AUTH:
            return f(*args, **kwargs)
        
        auth = request.authorization
        
        if not auth:
            logger.warning(f"Request without auth from {request.remote_addr}")
            return jsonify({"error": "Authentication required"}), 401
        
        if auth.username != settings.API_USERNAME or auth.password != settings.API_PASSWORD:
            logger.warning(f"Invalid credentials from {request.remote_addr}")
            return jsonify({"error": "Invalid credentials"}), 403
        
        logger.info(f"Basic auth successful for user '{auth.username}' from {request.remote_addr}")
        return f(*args, **kwargs)
    
    return decorated_function


def get_auth_header_example():
    """Get example auth header for documentation."""
    if not settings.ENABLE_AUTH:
        return "No authentication required"
    
    return {
        "api_key_method": f"Header: X-API-Key: {settings.API_KEY}",
        "basic_auth_method": f"Authorization: Basic <base64({settings.API_USERNAME}:{settings.API_PASSWORD})>",
        "example_curl_api_key": f'curl -H "X-API-Key: {settings.API_KEY}" http://localhost:{settings.FLASK_PORT}/health',
        "example_curl_basic": f'curl -u {settings.API_USERNAME}:{settings.API_PASSWORD} http://localhost:{settings.FLASK_PORT}/health'
    }