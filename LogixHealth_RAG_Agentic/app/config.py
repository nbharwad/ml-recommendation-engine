"""Application configuration using Pydantic Settings."""

from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Azure OpenAI Configuration
    # =========================================================================
    azure_openai_endpoint: str = Field(
        default="",
        description="Azure OpenAI endpoint URL",
    )
    azure_openai_api_key: str = Field(
        default="",
        description="Azure OpenAI API key",
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview",
        description="Azure OpenAI API version",
    )
    azure_openai_deployment_name: str = Field(
        default="gpt-4o",
        description="GPT-4o deployment name",
    )
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-3-large",
        description="Text embedding deployment name",
    )
    azure_openai_cheap_deployment_name: Optional[str] = Field(
        default=None,
        description="Cheaper model for classification tasks",
    )

    # =========================================================================
    # Azure AI Search Configuration
    # =========================================================================
    azure_search_endpoint: str = Field(
        default="",
        description="Azure AI Search endpoint URL",
    )
    azure_search_api_key: str = Field(
        default="",
        description="Azure AI Search API key",
    )
    azure_search_index_name: str = Field(
        default="logixhealth-docs",
        description="Azure AI Search index name",
    )
    azure_search_semantic_config: str = Field(
        default="default",
        description="Semantic configuration name",
    )

    # =========================================================================
    # Redis Configuration
    # =========================================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis password (if auth enabled)",
    )

    # =========================================================================
    # Cache TTL Configuration (in seconds)
    # =========================================================================
    cache_ttl_query: int = Field(
        default=3600,
        description="Query cache TTL (1 hour)",
    )
    cache_ttl_retrieval: int = Field(
        default=14400,
        description="Retrieval cache TTL (4 hours)",
    )
    cache_ttl_llm: int = Field(
        default=1800,
        description="LLM response cache TTL (30 minutes)",
    )

    # =========================================================================
    # Memory Configuration
    # =========================================================================
    session_memory_max_turns: int = Field(
        default=10,
        description="Maximum conversation turns in session memory",
    )
    session_memory_ttl: int = Field(
        default=86400,
        description="Session memory TTL (24 hours)",
    )
    long_term_memory_ttl: int = Field(
        default=2592000,
        description="Long-term memory TTL (30 days)",
    )
    long_term_memory_max_summaries: int = Field(
        default=20,
        description="Maximum long-term memory summaries before merging",
    )

    # =========================================================================
    # Retrieval Configuration
    # =========================================================================
    retrieval_top_k: int = Field(
        default=5,
        description="Number of documents to return from search",
    )
    retrieval_top_n_candidates: int = Field(
        default=50,
        description="Number of candidates before reranking",
    )
    reranker_semantic_weight: float = Field(
        default=0.70,
        description="Semantic score weight in reranking",
    )
    reranker_recency_weight: float = Field(
        default=0.15,
        description="Recency score weight in reranking",
    )
    reranker_authority_weight: float = Field(
        default=0.10,
        description="Authority score weight in reranking",
    )
    reranker_exact_match_weight: float = Field(
        default=0.05,
        description="Exact match score weight in reranking",
    )

    # =========================================================================
    # Agent Configuration
    # =========================================================================
    max_audit_retries: int = Field(
        default=2,
        description="Maximum audit retry cycles",
    )
    confidence_high_threshold: float = Field(
        default=0.85,
        description="High confidence threshold",
    )
    confidence_low_threshold: float = Field(
        default=0.60,
        description="Low confidence threshold",
    )

    # =========================================================================
    # API Configuration
    # =========================================================================
    api_rate_limit_per_minute: int = Field(
        default=60,
        description="Rate limit per minute per IP",
    )
    api_host: str = Field(
        default="0.0.0.0",
        description="API host",
    )
    api_port: int = Field(
        default=8000,
        description="API port",
    )
    cors_origins: str = Field(
        default="http://localhost:5173",
        description="CORS origins (comma-separated)",
    )

    # =========================================================================
    # Application Configuration
    # =========================================================================
    environment: str = Field(
        default="development",
        description="Environment: development | staging | production",
    )
    log_level: str = Field(
        default="INFO",
        description="Log level: DEBUG | INFO | WARNING | ERROR",
    )
    debug: bool = Field(
        default=False,
        description="Debug mode",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )

    # =========================================================================
    # Observability Configuration
    # =========================================================================
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangChain tracing",
    )
    langchain_api_key: Optional[str] = Field(
        default=None,
        description="LangChain API key",
    )
    log_format: str = Field(
        default="json",
        description="Log format: json | text",
    )

    # =========================================================================
    # Security Configuration
    # =========================================================================
    api_auth_enabled: bool = Field(
        default=False,
        description="Enable API authentication",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication",
    )
    jwt_secret: Optional[str] = Field(
        default=None,
        description="JWT secret for tokens",
    )

    # =========================================================================
    # Validators
    # =========================================================================
    @field_validator("azure_openai_endpoint")
    @classmethod
    def validate_azure_openai_endpoint(cls, v: str) -> str:
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("azure_openai_endpoint must be a valid URL")
        return v

    @field_validator("azure_search_endpoint")
    @classmethod
    def validate_azure_search_endpoint(cls, v: str) -> str:
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("azure_search_endpoint must be a valid URL")
        return v

    @field_validator(
        "reranker_semantic_weight",
        "reranker_recency_weight",
        "reranker_authority_weight",
        "reranker_exact_match_weight",
    )
    @classmethod
    def validate_reranker_weights(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Reranker weights must be between 0.0 and 1.0")
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"environment must be one of: {valid_envs}")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of: {valid_levels}")
        return v.upper()


# Global settings instance
settings = Settings()
