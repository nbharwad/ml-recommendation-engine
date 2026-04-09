"""Audit Agent - hallucination detection and confidence scoring."""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.prompts import load_prompt
from app.agents.state import AgentState
from app.dependencies import get_llm
from app.observability.logger import get_logger, log_agent_step

logger = get_logger(__name__)


async def audit_agent(state: AgentState) -> dict[str, Any]:
    """Performs final quality check on the response.
    
    Args:
        state: The current agent state.
        
    Returns:
        A dictionary containing audit_passed, final_answer, confidence_score, and citations.
    """
    start_time = time.perf_counter()
    step_name = "audit"
    
    try:
        # Load system prompt
        system_prompt = load_prompt("audit_system")
        
        # Evidence context
        retrieved_docs = state.get("retrieved_docs", [])
        tool_outputs = state.get("tool_outputs", [])
        
        docs_context = "\n".join([
            f"Document [{i+1}]: Source: {doc.get('metadata', {}).get('source', doc.get('metadata', {}).get('publisher', 'Unknown'))}\nContent: {doc.get('content', '')}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        tools_context = "\n".join([
            f"Tool Output [{tool.get('tool_name')}]: {json.dumps(tool.get('output', {}), indent=2)}"
            for tool in tool_outputs
        ])
        
        # Prepare input
        user_input = (
            f"Query: {state['query']}\n\n"
            f"Draft Answer:\n{state.get('reasoning_output', '')}\n\n"
            f"Evidence (Docs):\n{docs_context}\n\n"
            f"Evidence (Tools):\n{tools_context}"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]
        
        # Get LLM (using JSON mode)
        llm = get_llm(json_mode=True)
        
        # Call LLM
        response = await llm.ainvoke(messages)
        
        # Parse output
        audit_data = json.loads(response.content)
        
        audit_passed = audit_data.get("audit_passed", True)
        final_answer = audit_data.get("final_answer", state.get("reasoning_output"))
        confidence_score = audit_data.get("confidence_score", 0.0)
        
        # Extract citations from the draft or final answer (simulated here)
        # In a real system, this would be a more robust parsing
        citations = []
        for i, doc in enumerate(retrieved_docs):
            citations.append({
                "index": i + 1,
                "source": doc.get("metadata", {}).get("source", doc.get("metadata", {}).get("publisher", "Unknown")),
                "text": doc.get("content", "")[:200] + "..."
            })
            
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_agent_step(
            logger,
            step_name,
            f"Audit completed. Passed: {audit_passed}. Confidence: {confidence_score}",
            duration_ms=duration_ms,
            audit_passed=audit_passed,
            confidence_score=confidence_score
        )
        
        return {
            "audit_passed": audit_passed,
            "final_answer": final_answer,
            "confidence_score": confidence_score,
            "citations": citations,
            "audit_retry_count": state.get("audit_retry_count", 0) + (1 if not audit_passed else 0)
        }
        
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Audit error: {str(e)}", exc_info=True)
        return {
            "audit_passed": True,  # Fallback: bypass audit on error
            "final_answer": state.get("reasoning_output"),
            "confidence_score": 0.5,
            "error": f"Audit error: {str(e)}"
        }
