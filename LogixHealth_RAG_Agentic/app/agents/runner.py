"""Unified entry point for running the agentic workflow with integrated memory."""

import uuid
from typing import Any, Optional

from app.agents.graph import build_graph
from app.agents.state import AgentState
from app.cache.redis_cache import create_cache
from app.dependencies import get_llm
from app.memory.memory_manager import create_memory_manager
from app.observability.logger import get_logger

logger = get_logger(__name__)


async def run_agent_workflow(
    query: str,
    session_id: str,
    user_id: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run the agentic workflow with full memory integration.

    Args:
        query: User input query
        session_id: Session identifier for short-term memory
        user_id: User identifier for long-term memory
        trace_id: Optional trace identifier (generated if not provided)

    Returns:
        Final agent state containing the answer and citations
    """
    # 1. Setup identifiers and services
    current_trace_id = trace_id or str(uuid.uuid4())
    
    # Initialize Redis and MemoryManager
    # In a full FastAPI app, these would be managed via lifespan or dependency overrides
    cache = await create_cache()
    llm = get_llm()
    memory_manager = await create_memory_manager(redis_client=cache._client, llm_client=llm)

    # 2. Pre-execution: Inject memory context
    memory_context = ""
    try:
        memory_context = await memory_manager.inject_memory(session_id, user_id)
    except Exception as e:
        logger.warning(
            "Failed to inject memory context, proceeding with empty memory",
            extra={"session_id": session_id, "user_id": user_id, "error": str(e)},
        )

    # 3. Graph Execution
    initial_state: AgentState = {
        "query": query,
        "trace_id": current_trace_id,
        "session_id": session_id,
        "memory_context": memory_context,
        "retrieved_docs": [],
        "tool_outputs": [],
        "audit_retry_count": 0,
        "metadata": {"user_id": user_id} if user_id else {}
    }

    try:
        graph = build_graph()
        final_state = await graph.ainvoke(initial_state)
        
        # 4. Post-execution: Save interaction to memory
        final_answer = final_state.get("final_answer", "")
        
        if final_answer:
            try:
                # Add user turn
                await memory_manager.add_turn(
                    session_id=session_id,
                    role="user",
                    content=query,
                    trace_id=current_trace_id,
                    user_id=user_id
                )
                
                # Add assistant turn
                await memory_manager.add_turn(
                    session_id=session_id,
                    role="assistant",
                    content=final_answer,
                    trace_id=current_trace_id,
                    user_id=user_id
                )
            except Exception as e:
                logger.warning(
                    "Failed to save turns to memory",
                    extra={"session_id": session_id, "error": str(e)},
                )
        
        return final_state

    except Exception as e:
        logger.error(
            "Workflow execution failed",
            extra={"trace_id": current_trace_id, "error": str(e)},
            exc_info=True
        )
        return {
            "query": query,
            "error": f"Workflow failed: {str(e)}",
            "trace_id": current_trace_id
        }
    finally:
        # Cleanup Redis connection
        await cache.disconnect()
