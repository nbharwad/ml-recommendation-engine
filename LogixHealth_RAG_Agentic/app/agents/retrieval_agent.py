"""Retrieval Agent - orchestrates query rewrite, search, and rerank."""

import time
from typing import Any

from app.agents.state import AgentState
from app.config import settings
from app.dependencies import get_reranker, get_retriever, get_rewriter
from app.observability.logger import get_logger, log_agent_step

logger = get_logger(__name__)


async def retrieval_agent(state: AgentState) -> dict[str, Any]:
    """Orchestrates the retrieval pipeline.
    
    Args:
        state: The current agent state.
        
    Returns:
        A dictionary containing the retrieved_docs and rewritten_query.
    """
    start_time = time.perf_counter()
    step_name = "retrieval"
    
    try:
        rewriter = get_rewriter()
        retriever = get_retriever()
        reranker = get_reranker()
        
        # 1. Query Rewriting
        memory_context = state.get("memory_context", "")
        rewritten_result = await rewriter.rewrite(
            query=state["query"],
            session_context=memory_context
        )
        
        rewritten_query = rewritten_result.rewritten_query
        sub_queries = rewritten_result.sub_queries
        entities = rewritten_result.entities
        
        # 1b. Map entities to SearchFilters
        from app.retrieval.azure_search import SearchFilters
        search_filters = SearchFilters()
        for entity in entities:
            if entity.type == "payer":
                search_filters.payer = entity.value
            elif entity.type == "cpt_code":
                search_filters.cpt_code = entity.value
            elif entity.type == "denial_code":
                search_filters.denial_code = entity.value

        # 2. Hybrid Search (including sub-queries)
        all_docs = []
        queries_to_run = [rewritten_query] + sub_queries
        
        # In a production system, these would be run in parallel
        for q in queries_to_run:
            search_result = await retriever.search(
                query=q,
                filters=search_filters,
                top_k=settings.retrieval_top_n_candidates
            )
            all_docs.extend(search_result.documents)
            
        # 3. Application-level Reranking
        # Remove duplicates based on id
        unique_docs = {doc.id: doc for doc in all_docs}.values()
        
        reranked_docs = reranker.rerank(
            documents=list(unique_docs),
            query=rewritten_query,
            top_k=settings.retrieval_top_k
        )
        
        # Convert Pydantic models to dicts for LangGraph state
        serializable_docs = [doc.model_dump() for doc in reranked_docs]
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_agent_step(
            logger,
            step_name,
            f"Retrieved {len(serializable_docs)} relevant documents",
            duration_ms=duration_ms,
            rewritten_query=rewritten_query,
            num_docs=len(serializable_docs)
        )
        
        return {
            "retrieved_docs": serializable_docs,
            "rewritten_query": rewritten_query
        }
        
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Retrieval error: {str(e)}", exc_info=True)
        return {
            "error": f"Retrieval error: {str(e)}"
        }
