"""Optional authentication support for DROMA MCP Server (FastMCP 2.13+).

This module provides enterprise-grade authentication capabilities for production deployments.
Authentication is OPTIONAL and only needed for web deployments requiring access control.

Supported Providers:
- API Key (simplest, for internal services)
- Google OAuth
- GitHub OAuth  
- Microsoft Azure AD
- Custom JWT

Usage:
    from droma_mcp.auth import create_auth_provider
    
    # API Key authentication (simplest)
    auth = create_auth_provider("api_key", api_keys=["your-secret-key"])
    
    # Google OAuth (for user authentication)
    auth = create_auth_provider(
        "google",
        client_id="your-client-id",
        client_secret="your-secret",
        base_url="https://your-server.com"
    )
    
    # Then use with FastMCP:
    droma_mcp = FastMCP("DROMA-MCP-Server", auth=auth)
"""

import os
from typing import Optional, List, Dict, Any
from enum import Enum


class AuthProvider(str, Enum):
    """Supported authentication providers."""
    NONE = "none"
    API_KEY = "api_key"
    GOOGLE = "google"
    GITHUB = "github"
    AZURE = "azure"
    JWT = "jwt"


def create_auth_provider(
    provider: str,
    api_keys: Optional[List[str]] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    base_url: Optional[str] = None,
    jwt_secret: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Create an authentication provider for FastMCP 2.13+.
    
    Args:
        provider: Authentication provider type
        api_keys: List of valid API keys (for api_key provider)
        client_id: OAuth client ID (for OAuth providers)
        client_secret: OAuth client secret (for OAuth providers)
        base_url: Base URL for OAuth callbacks
        jwt_secret: Secret for JWT validation (for jwt provider)
        **kwargs: Additional provider-specific configuration
    
    Returns:
        Authentication provider instance compatible with FastMCP
    
    Raises:
        ImportError: If fastmcp.server.auth is not available
        ValueError: If required parameters are missing
    
    Example:
        >>> # Simple API key authentication
        >>> auth = create_auth_provider("api_key", api_keys=["secret123"])
        >>> 
        >>> # Google OAuth
        >>> auth = create_auth_provider(
        ...     "google",
        ...     client_id=os.getenv("GOOGLE_CLIENT_ID"),
        ...     client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        ...     base_url="https://droma-mcp.example.com"
        ... )
    """
    try:
        from fastmcp.server.auth import providers
    except ImportError:
        raise ImportError(
            "FastMCP authentication not available. "
            "Ensure you have FastMCP 2.13+ installed: pip install 'fastmcp>=2.13.0'"
        )
    
    provider = provider.lower()
    
    if provider == "none":
        return None
    
    elif provider == "api_key":
        if not api_keys:
            raise ValueError("API keys required for api_key provider")
        
        # Simple API key validation function
        def validate_api_key(key: str) -> bool:
            return key in api_keys
        
        # Return a simple auth validator
        # Note: FastMCP 2.13+ has specific API key provider
        try:
            return providers.APIKeyProvider(api_keys=api_keys)
        except AttributeError:
            # Fallback for different FastMCP versions
            return validate_api_key
    
    elif provider == "google":
        if not all([client_id, client_secret, base_url]):
            raise ValueError("client_id, client_secret, and base_url required for Google OAuth")
        
        return providers.GoogleProvider(
            client_id=client_id,
            client_secret=client_secret,
            base_url=base_url,
            **kwargs
        )
    
    elif provider == "github":
        if not all([client_id, client_secret, base_url]):
            raise ValueError("client_id, client_secret, and base_url required for GitHub OAuth")
        
        return providers.GitHubProvider(
            client_id=client_id,
            client_secret=client_secret,
            base_url=base_url,
            **kwargs
        )
    
    elif provider == "azure":
        if not all([client_id, client_secret, base_url]):
            raise ValueError("client_id, client_secret, and base_url required for Azure AD")
        
        tenant_id = kwargs.get("tenant_id", "common")
        
        return providers.AzureProvider(
            client_id=client_id,
            client_secret=client_secret,
            base_url=base_url,
            tenant_id=tenant_id,
            **kwargs
        )
    
    elif provider == "jwt":
        if not jwt_secret:
            raise ValueError("jwt_secret required for JWT authentication")
        
        return providers.JWTProvider(
            jwt_secret=jwt_secret,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unsupported authentication provider: {provider}")


def load_auth_from_env() -> Optional[Any]:
    """
    Load authentication configuration from environment variables.
    
    Environment Variables:
        DROMA_MCP_AUTH_PROVIDER: Provider type (none, api_key, google, github, azure, jwt)
        DROMA_MCP_API_KEYS: Comma-separated API keys (for api_key)
        DROMA_MCP_CLIENT_ID: OAuth client ID
        DROMA_MCP_CLIENT_SECRET: OAuth client secret
        DROMA_MCP_BASE_URL: Base URL for OAuth callbacks
        DROMA_MCP_JWT_SECRET: JWT secret (for jwt)
        DROMA_MCP_AZURE_TENANT: Azure tenant ID (for azure)
    
    Returns:
        Authentication provider or None
    
    Example:
        >>> # Set environment variables:
        >>> # export DROMA_MCP_AUTH_PROVIDER=api_key
        >>> # export DROMA_MCP_API_KEYS=key1,key2,key3
        >>> auth = load_auth_from_env()
    """
    provider = os.getenv("DROMA_MCP_AUTH_PROVIDER", "none").lower()
    
    if provider == "none":
        return None
    
    config = {
        "provider": provider,
        "api_keys": os.getenv("DROMA_MCP_API_KEYS", "").split(",") if os.getenv("DROMA_MCP_API_KEYS") else None,
        "client_id": os.getenv("DROMA_MCP_CLIENT_ID"),
        "client_secret": os.getenv("DROMA_MCP_CLIENT_SECRET"),
        "base_url": os.getenv("DROMA_MCP_BASE_URL"),
        "jwt_secret": os.getenv("DROMA_MCP_JWT_SECRET"),
        "tenant_id": os.getenv("DROMA_MCP_AZURE_TENANT", "common"),
    }
    
    # Remove None values
    config = {k: v for k, v in config.items() if v}
    
    try:
        return create_auth_provider(**config)
    except Exception as e:
        print(f"Warning: Failed to load authentication from environment: {e}")
        return None


__all__ = [
    "AuthProvider",
    "create_auth_provider",
    "load_auth_from_env"
]

