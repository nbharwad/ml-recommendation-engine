"""FastAPI dependency injection and service providers."""

from functools import lru_cache
from typing import Optional

from langchain_openai import AzureChatOpenAI

from app.config import settings
from app.retrieval.azure_search import AzureSearchRetriever
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import Reranker


@lru_cache()
def get_llm(
    deployment_name: Optional[str] = None,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> AzureChatOpenAI:
    """Get a configured Azure OpenAI LLM instance."""
    deployment = deployment_name or settings.azure_openai_deployment_name
    
    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        openai_api_key=settings.azure_openai_api_key,
        openai_api_version=settings.azure_openai_api_version,
        deployment_name=deployment,
        temperature=temperature,
        model_kwargs=model_kwargs,
    )


@lru_cache()
def get_retriever() -> AzureSearchRetriever:
    """Get a configured AzureSearchRetriever instance."""
    return AzureSearchRetriever(
        endpoint=settings.azure_search_endpoint,
        api_key=settings.azure_search_api_key,
        index_name=settings.azure_search_index_name,
    )


@lru_cache()
def get_rewriter() -> QueryRewriter:
    """Get a configured QueryRewriter instance."""
    return QueryRewriter(llm_client=get_llm())


@lru_cache()
def get_reranker() -> Reranker:
    """Get a configured Reranker instance."""
    return Reranker(
        semantic_weight=settings.reranker_semantic_weight,
        recency_weight=settings.reranker_recency_weight,
        authority_weight=settings.reranker_authority_weight,
        exact_match_weight=settings.reranker_exact_match_weight,
    )


# TODO: Implement Redis connection pool (Phase 3)
# TODO: Implement agent routing dependencies (Phase 10)
