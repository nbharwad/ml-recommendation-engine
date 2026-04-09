"""Azure AI Search retriever with hybrid search capabilities."""

from dataclasses import dataclass
from typing import Any, Optional

from azure.core.exceptions import AzureError, HttpResponseError
from azure.search.documents import SearchClient
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.observability.logger import get_logger


class SearchDocument(BaseModel):
    """Search result document model."""

    id: str = Field(description="Document ID")
    content: str = Field(description="Document content")
    score: float = Field(default=0.0, description="Relevance score")
    payer: Optional[str] = Field(default=None, description="Payer name")
    cpt_code: Optional[str] = Field(default=None, description="CPT code")
    denial_code: Optional[str] = Field(default=None, description="Denial code")
    authority_score: float = Field(default=0.5, description="Authority score (0-1)")
    last_updated: Optional[str] = Field(default=None, description="Last updated date")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchFilters(BaseModel):
    """Search filter parameters."""

    payer: Optional[str] = None
    cpt_code: Optional[str] = None
    denial_code: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def to_odata_filter(self) -> str:
        """Convert filters to OData filter string."""
        filters = []

        if self.payer:
            filters.append(f"payer eq '{self.payer}'")
        if self.cpt_code:
            filters.append(f"cpt_code eq '{self.cpt_code}'")
        if self.denial_code:
            filters.append(f"denial_code eq '{self.denial_code}'")
        if self.start_date:
            filters.append(f"last_updated ge '{self.start_date}'")
        if self.end_date:
            filters.append(f"last_updated le '{self.end_date}'")

        return " and ".join(filters) if filters else ""


@dataclass
class SearchResult:
    """Search results container."""

    documents: list[SearchDocument]
    total_count: Optional[int] = None
    query: str = ""


class AzureSearchRetriever:
    """Azure AI Search retriever with hybrid search and retry logic."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
    ):
        """Initialize Azure Search client."""
        self.endpoint = endpoint or settings.azure_search_endpoint
        self.api_key = api_key or settings.azure_search_api_key
        self.index_name = index_name or settings.azure_search_index_name
        self._logger = get_logger(__name__)

        if not self.endpoint or not self.api_key:
            raise ValueError("Azure Search endpoint and API key are required")

        self._client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self._create_credential(),
        )

    def _create_credential(self) -> "AzureKeyCredential":
        """Create API key credential."""
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential(self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        top_k: Optional[int] = None,
        embedding: Optional[list[float]] = None,
    ) -> SearchResult:
        """
        Perform hybrid search (vector + keyword) with optional filters.

        Args:
            query: Search query string
            filters: Optional search filters
            top_k: Number of results to return
            embedding: Optional vector embedding for semantic search

        Returns:
            SearchResult with ranked documents
        """
        top_k = top_k or settings.retrieval_top_k

        odata_filter = filters.to_odata_filter() if filters else ""

        search_options: dict[str, Any] = {
            "top": top_k,
            "query_type": "simple",
        }

        if odata_filter:
            search_options["filter"] = odata_filter

        if embedding:
            if isinstance(embedding, str):
                from azure.search.documents.models import VectorizableTextQuery

                vector_query = VectorizableTextQuery(
                    text=embedding,
                    k_nearest_neighbors=top_k * 2,
                    fields="content_vector",
                )
            else:
                from azure.search.documents.models import VectorizedQuery

                vector_query = VectorizedQuery(
                    vector=embedding,
                    k_nearest=top_k * 2,
                    fields="content_vector",
                )
            search_options["vector_queries"] = [vector_query]
            search_options["semantic_configuration_name"] = settings.azure_search_semantic_config

        try:
            results = self._client.search(
                search_text=query if not embedding else "*",
                **search_options,
            )

            documents = []
            for result in results:
                doc = SearchDocument(
                    id=result.get("id", ""),
                    content=result.get("content", result.get("chunk", "")),
                    score=result.get("@search_score", 0.0),
                    payer=result.get("payer"),
                    cpt_code=result.get("cpt_code"),
                    denial_code=result.get("denial_code"),
                    authority_score=result.get("authority_score", 0.5),
                    last_updated=result.get("last_updated"),
                    metadata=result.get("metadata", {}),
                )
                documents.append(doc)

            return SearchResult(
                documents=documents,
                total_count=len(documents),
                query=query,
            )

        except HttpResponseError as e:
            status = e.status_code
            if status == 400:
                self._logger.warning(
                    "Azure Search bad request (400)",
                    extra={"query": query, "error": str(e)},
                )
                return SearchResult(documents=[], query=query)
            elif status in (401, 403):
                self._logger.error(
                    f"Azure Search authentication failed ({status})",
                    extra={"status": status, "error": str(e)},
                )
                raise
            raise
        except AzureError as e:
            raise RuntimeError(f"Azure Search error: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Search failed: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def get_document(self, document_id: str) -> Optional[SearchDocument]:
        """Retrieve a single document by ID."""
        try:
            result = self._client.get_document(key=document_id)
            return SearchDocument(
                id=result.get("id", ""),
                content=result.get("content", result.get("chunk", "")),
                score=1.0,
                payer=result.get("payer"),
                cpt_code=result.get("cpt_code"),
                denial_code=result.get("denial_code"),
                authority_score=result.get("authority_score", 0.5),
                last_updated=result.get("last_updated"),
                metadata=result.get("metadata", {}),
            )
        except Exception:
            return None


def create_retriever() -> AzureSearchRetriever:
    """Factory function to create retriever from settings."""
    return AzureSearchRetriever()
