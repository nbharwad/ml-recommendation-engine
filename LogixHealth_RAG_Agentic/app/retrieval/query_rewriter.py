"""LLM-based query rewriting for improved retrieval."""

from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings


class ExtractedEntity(BaseModel):
    """Extracted entity from query."""

    type: str = Field(description="Entity type (payer, cpt_code, denial_code, policy)")
    value: str = Field(description="Entity value")


class QueryRewriteResult(BaseModel):
    """Result of query rewrite operation."""

    rewritten_query: str = Field(description="Expanded query")
    sub_queries: list[str] = Field(
        default_factory=list, description="Sub-queries for multi-hop questions"
    )
    entities: list[ExtractedEntity] = Field(default_factory=list, description="Extracted entities")
    original_query: str = Field(description="Original query")


QUERY_EXPANSION_PROMPT = """You are a query expansion assistant for a healthcare RCM (Revenue Cycle Management) system.

Your task is to expand user queries to improve retrieval from a knowledge base containing:
- Denial codes and their explanations
- CPT code information
- Payer policies
- Appeal letter templates

Rules:
1. Expand abbreviations (e.g., "UHC" -> "UnitedHealthcare")
2. Add relevant medical/RCM terminology
3. Include common variations of terms
4. Keep the expanded query concise but comprehensive
5. Focus on terms that would help find relevant documents
6. If the query requires multi-hop reasoning (e.g., "what is denial CO-4 and how do I appeal it?"), generate sub-queries

Output the expanded query, optional sub-queries for multi-hop questions, and extract any entities (payer names, CPT codes, denial codes, policy names).

Examples:
- Input: "denial 27" -> Output: "denial code 27 missing information", sub_queries: []
- Input: "UHC policy" -> Output: "UnitedHealthcare policy prior authorization requirements", sub_queries: []
- Input: "99213 reimbursement" -> Output: "CPT 99213 office visit reimbursement rates", sub_queries: []
- Input: "what is CO-4 and how do I appeal?" -> Output: "denial code CO-4 explanation and appeal process", sub_queries: ["denial code CO-4 explanation", "denial code CO-4 appeal process"]

Now expand this query:
Original: {query}

Respond in JSON format:
{{
    "rewritten_query": "expanded query",
    "sub_queries": ["sub-query 1", "sub-query 2"],
    "entities": [
        {{"type": "cpt_code", "value": "99213"}}
    ]
}}"""


class QueryRewriter:
    """LLM-based query rewriter for retrieval improvement."""

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.0):
        """Initialize query rewriter with LLM."""
        cheap_deployment = settings.azure_openai_cheap_deployment_name
        deployment = model_name or cheap_deployment or settings.azure_openai_deployment_name

        self._llm = ChatOpenAI(
            model=deployment,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            temperature=temperature,
            max_tokens=500,
        )

    async def rewrite(
        self, query: str, session_context: Optional[str] = None
    ) -> QueryRewriteResult:
        """
        Rewrite and expand a query for better retrieval.

        Args:
            query: Original user query
            session_context: Optional session context for resolving ambiguous references

        Returns:
            QueryRewriteResult with expanded query, sub-queries, and extracted entities
        """
        if not query or not query.strip():
            return QueryRewriteResult(
                rewritten_query=query,
                sub_queries=[],
                entities=[],
                original_query=query,
            )

        context_section = f"\n\nSession Context:\n{session_context}" if session_context else ""
        prompt = QUERY_EXPANSION_PROMPT.format(query=query) + context_section

        try:
            response = self._llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            import json

            parsed = json.loads(content)

            entities = [
                ExtractedEntity(type=e["type"], value=e["value"])
                for e in parsed.get("entities", [])
            ]

            return QueryRewriteResult(
                rewritten_query=parsed.get("rewritten_query", query),
                sub_queries=parsed.get("sub_queries", []),
                entities=entities,
                original_query=query,
            )

        except json.JSONDecodeError:
            return QueryRewriteResult(
                rewritten_query=query,
                sub_queries=[],
                entities=[],
                original_query=query,
            )
        except Exception:
            return QueryRewriteResult(
                rewritten_query=query,
                sub_queries=[],
                entities=[],
                original_query=query,
            )


def create_rewriter() -> QueryRewriter:
    """Factory function to create query rewriter from settings."""
    return QueryRewriter()
