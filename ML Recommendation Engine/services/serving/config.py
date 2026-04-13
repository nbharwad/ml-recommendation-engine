"""
Configuration Validation — Startup Validation
========================================
Validates required environment variables at startup.
Uses pydantic_settings for proper validation.
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application settings with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    
    feature_service_host: str = "feature-service"
    feature_service_port: int = 50051
    retrieval_service_host: str = "retrieval-service"
    retrieval_service_port: int = 50052
    ranking_service_host: str = "ranking-service"
    ranking_service_port: int = 50053
    reranking_service_host: str = "reranking-service"
    reranking_service_port: int = 50054
    experiment_service_host: str = "experiment-service"
    experiment_service_port: int = 50055
    
    jwt_issuer: str = "https://auth.example.com"
    jwt_jwks_uri: str = "https://auth.example.com/.well-known/jwks.json"
    
    cors_allowed_origins: str = "http://localhost:3000"
    
    pseudonymization_salt: str = "default_salt_change_in_production"
    
    @model_validator(mode="after")
    def validate_critical(self) -> AppSettings:
        """Validate critical configuration at startup."""
        if len(self.pseudonymization_salt) < 32:
            raise ValueError(
                "PSEUDONYMIZATION_SALT must be >= 32 characters. "
                f"Current length: {len(self.pseudonymization_salt)}"
            )
        return self


def get_app_settings() -> AppSettings:
    """Get validated application settings."""
    return AppSettings()


def verify_startup_config() -> None:
    """
    Verify required configuration at startup.
    Exits with code 1 if validation fails.
    """
    try:
        settings = get_app_settings()
        print(f"✓ Configuration validated")
        print(f"  JWT Issuer: {settings.jwt_issuer}")
        print(f"  CORS Origins: {settings.cors_allowed_origins}")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    verify_startup_config()