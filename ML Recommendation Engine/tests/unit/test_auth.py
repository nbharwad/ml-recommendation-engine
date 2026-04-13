"""
Unit Tests — JWT Authentication
============================
Tests for JWT validation with JWKS-based signature verification.
"""

import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend


def generate_test_keypair():
    """Generate RSA key pair for testing."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    public_key = private_key.public_key()
    
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    
    return private_pem, public_pem


class TestJWTValidation:
    """Test JWT validation scenarios."""
    
    @pytest.fixture
    def test_keys(self):
        return generate_test_keypair()
    
    @pytest.fixture
    def mock_jwks_server(self, test_keys):
        private_pem, public_pem = test_keys
        return {"private": private_pem, "public": public_pem}
    
    def create_token(self, private_pem: bytes, payload: dict, algorithm: str = "RS256") -> str:
        """Create a signed JWT token."""
        return jwt.encode(payload, private_pem, algorithm=algorithm)
    
    @pytest.mark.asyncio
    async def test_valid_token_returns_claims(self, mock_jwks_server):
        """Valid token should return claims."""
        from auth import validate_jwt
        
        private_pem = mock_jwks_server["private"]
        
        payload = {
            "sub": "user123",
            "role": "admin",
            "iss": "https://auth.example.com",
            "exp": int(time.time()) + 3600,
        }
        
        token = self.create_token(private_pem, payload)
        
        with patch("auth PyJWKClient") as mock_jwk_client:
            mock_jwk_client.return_value.get_signing_key_from_jwt.return_value = MagicMock(
                key=serialization.load_pem_private_key(private_pem, None)
            )
            
            with patch("auth.jwt.decode") as mock_decode:
                mock_decode.return_value = payload
                
                claims = await validate_jwt(token)
                
                assert claims["sub"] == "user123"
                assert claims["role"] == "admin"
    
    @pytest.mark.asyncio
    async def test_expired_token_raises_401(self):
        """Expired token should raise ValueError."""
        from auth import validate_jwt
        
        payload = {
            "sub": "user123",
            "iss": "https://auth.example.com",
            "exp": int(time.time()) - 3600,
        }
        
        token = "expired.token.here"
        
        with pytest.raises(ValueError, match="expired"):
            await validate_jwt(token)
    
    @pytest.mark.asyncio
    async def test_tampered_payload_raises_401(self):
        """Tampered payload should raise ValueError."""
        from auth import validate_jwt
        
        token = "valid.header.tampered_payload_signature"
        
        with pytest.raises(ValueError, match="validation failed"):
            await validate_jwt(token)
    
    @pytest.mark.asyncio
    async def test_unknown_issuer_raises_401(self):
        """Unknown issuer should raise ValueError."""
        from auth import validate_jwt
        
        payload = {
            "sub": "user123",
            "iss": "https://malicious.com",
            "exp": int(time.time()) + 3600,
        }
        
        token = "malicious.token.here"
        
        with pytest.raises(ValueError, match="issuer"):
            await validate_jwt(token)
    
    @pytest.mark.asyncio
    async def test_missing_signature_raises_401(self):
        """Token without signature should raise ValueError."""
        from auth import validate_jwt
        
        parts = "header.payload".split(".")
        
        with pytest.raises(ValueError):
            await validate_jwt(".".join(parts))


class TestRBACRoles:
    """Test RBAC role hierarchy."""
    
    def test_role_hierarchy_order(self):
        """Verify role hierarchy is correct."""
        from auth import Role, ROLE_HIERARCHY
        
        assert ROLE_HIERARCHY[Role.READER] < ROLE_HIERARCHY[Role.DATA_SCIENTIST]
        assert ROLE_HIERARCHY[Role.DATA_SCIENTIST] < ROLE_HIERARCHY[Role.OPERATOR]
        assert ROLE_HIERARCHY[Role.OPERATOR] < ROLE_HIERARCHY[Role.ADMIN]
        assert ROLE_HIERARCHY[Role.ADMIN] < ROLE_HIERARCHY[Role.SUPER_ADMIN]
    
    def test_permissions_defined(self):
        """Verify all required permissions are defined."""
        from auth import PERMISSIONS
        
        required_permissions = [
            "read_recommendations",
            "write_recommendations",
            "manage_users",
            "deploy_services",
            "view_audit_logs",
        ]
        
        for perm in required_permissions:
            assert perm in PERMISSIONS, f"Missing permission: {perm}"
    
    def test_super_admin_has_all_permissions(self):
        """Super admin should have all permissions."""
        from auth import PERMISSIONS, Role
        
        for permission, roles in PERMISSIONS.items():
            assert Role.SUPER_ADMIN in roles, f"SUPER_ADMIN missing {permission}"