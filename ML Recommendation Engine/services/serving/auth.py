"""
RBAC (Role-Based Access Control) Implementation
========================================
API-level authorization service using JWT claims.

Roles defined in implementation.md Section 6.4:
- reader: Read-only access to public endpoints
- admin: Full administrative access
- operator: Operational access (deployments, monitoring)
- data_scientist: ML pipeline access
- super_admin: Emergency break-glass access

Architecture:
- JWT token validated at API gateway
- Role claims embedded in JWT payload
- Service-level authorization via this module
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import structlog
from fastapi import Depends, HTTPException, Request, status
from pydantic import BaseModel

logger = structlog.get_logger(component="rbac")


class Role(str, Enum):
    """Five roles as defined in implementation.md Section 6.4"""
    READER = "reader"
    ADMIN = "admin"
    OPERATOR = "operator"
    DATA_SCIENTIST = "data_scientist"
    SUPER_ADMIN = "super_admin"


ROLE_HIERARCHY = {
    Role.READER: 1,
    Role.DATA_SCIENTIST: 2,
    Role.OPERATOR: 3,
    Role.ADMIN: 4,
    Role.SUPER_ADMIN: 5,
}


PERMISSIONS = {
    "read_recommendations": [Role.READER, Role.OPERATOR, Role.ADMIN, Role.DATA_SCIENTIST, Role.SUPER_ADMIN],
    "write_recommendations": [Role.OPERATOR, Role.ADMIN, Role.SUPER_ADMIN],
    "read_metrics": [Role.READER, Role.OPERATOR, Role.ADMIN, Role.DATA_SCIENTIST, Role.SUPER_ADMIN],
    "write_metrics": [Role.OPERATOR, Role.ADMIN, Role.SUPER_ADMIN],
    "read_models": [Role.DATA_SCIENTIST, Role.ADMIN, Role.SUPER_ADMIN],
    "write_models": [Role.DATA_SCIENTIST, Role.ADMIN, Role.SUPER_ADMIN],
    "deploy_services": [Role.OPERATOR, Role.ADMIN, Role.SUPER_ADMIN],
    "manage_users": [Role.ADMIN, Role.SUPER_ADMIN],
    "view_audit_logs": [Role.OPERATOR, Role.ADMIN, Role.SUPER_ADMIN],
    "system_config": [Role.SUPER_ADMIN],
}


@dataclass(frozen=True)
class RBACConfig:
    """RBAC configuration"""
    jwt_issuer: str = os.getenv("JWT_ISSUER", "https://auth.recommendation.internal/oauth2/default")
    jwt_jwks_uri: str = os.getenv("JWT_JWKS_URI", "https://auth.recommendation.internal/oauth2/default/v1/keys")
    required_role: Role = Role.READER
    enable_rbac: bool = os.getenv("ENABLE_RBAC", "true").lower() == "true"


class UserContext:
    """Current user context extracted from JWT"""
    
    def __init__(
        self,
        user_id: str,
        role: Role,
        email: Optional[str] = None,
        claims: Optional[dict] = None,
    ):
        self.user_id = user_id
        self.role = role
        self.email = email
        self.claims = claims or {}
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        allowed_roles = PERMISSIONS.get(permission, [])
        return self.role in allowed_roles
    
    def has_minimum_role(self, minimum_role: Role) -> bool:
        """Check if user has minimum role level"""
        return ROLE_HIERARCHY.get(self.role, 0) >= ROLE_HIERARCHY.get(minimum_role, 0)


def require_permission(permission: str) -> Callable:
    """
    Dependency for FastAPI endpoints requiring specific permission.
    
    Usage:
        @app.get("/admin/users")
        async def list_users(user: UserContext = Depends(require_permission("manage_users"))):
            ...
    """
    async def check_permission(request: Request) -> UserContext:
        user_context = await get_current_user(request)
        
        if not user_context.has_permission(permission):
            logger.warning(
                "permission_denied",
                user_id=user_context.user_id,
                role=user_context.role.value,
                permission=permission,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission}",
            )
        
        return user_context
    
    return check_permission


def require_role(minimum_role: Role) -> Callable:
    """
    Dependency for FastAPI endpoints requiring minimum role level.
    
    Usage:
        @app.get("/deploy")
        async def deploy(user: UserContext = Depends(require_role(Role.OPERATOR))):
            ...
    """
    async def check_role(request: Request) -> UserContext:
        user_context = await get_current_user(request)
        
        if not user_context.has_minimum_role(minimum_role):
            logger.warning(
                "role_denied",
                user_id=user_context.user_id,
                role=user_context.role.value,
                required_role=minimum_role.value,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{minimum_role.value}' or higher required",
            )
        
        return user_context
    
    return check_role


async def get_current_user(request: Request) -> UserContext:
    """
    Extract and validate current user from JWT token.
    
    Expected JWT payload:
        {
            "sub": "user_id",
            "email": "user@example.com",
            "role": "admin",
            "iss": "https://auth.example.com",
            "exp": 1234567890
        }
    """
    auth_header = request.headers.get("authorization", "")
    
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid authorization header",
        )
    
    token = auth_header[7:]
    
    try:
        claims = await validate_jwt(token)
    except Exception as e:
        logger.warning("jwt_validation_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    
    user_id = claims.get("sub")
    role_str = claims.get("role", "reader")
    email = claims.get("email")
    
    try:
        role = Role(role_str)
    except ValueError:
        role = Role.READER
    
    return UserContext(
        user_id=user_id,
        role=role,
        email=email,
        claims=claims,
    )


async def validate_jwt(token: str) -> dict:
    """Validate JWT token using JWKS signature verification."""
    from jose import jwt, JWTError
    from jose.exceptions import ExpiredSignatureError
    from jwt import PyJWKClient
    from .config import get_app_settings
    from fastapi import HTTPException
    
    settings = get_app_settings()
    try:
        # Get JWKS from identity provider
        jwks_client = PyJWKClient(settings.jwt_jwks_uri)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
        return payload
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Validation error: {e}")


async def get_audit_log(permission: str, user_id: str, result: str) -> None:
    """Log access for audit trail"""
    logger.info(
        "rbac_audit",
        permission=permission,
        user_id=user_id,
        result=result,
        timestamp_ms=int(time.time() * 1000),
    )


def create_emergency_access(user_id: str) -> UserContext:
    """Emergency break-glass access (implementation.md Section 6.4 Rollback Plan)"""
    return UserContext(
        user_id=user_id,
        role=Role.SUPER_ADMIN,
        claims={"emergency": True},
    )