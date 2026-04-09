"""Reasoning Agent - synthesizes docs and tool outputs into a cited response."""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.prompts import load_prompt
from app.agents.state import AgentState
from app.dependencies import get_llm
from app.observability.logger import get_logger, log_agent_step

logger = get_logger(__name__)


async def reasoning_agent(state: AgentState) -> dict[str, Any]:
    """Synthesizes inputs into a cited response.
    
    Args:
        state: The current agent state.
        
    Returns:
        A dictionary containing the reasoning_output.
    """
    start_time = time.perf_counter()
    step_name = "reasoning"
    
    try:
        # Load system prompt
        system_prompt = load_prompt("reasoning_system")
        
        # Prepare evidence context
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
        
        reflection_feedback = ""
        if state.get("reflection_output") and state.get("needs_revision"):
            reflection_output = state["reflection_output"]
            reflection_feedback = f"Reflection Feedback:\nIssues: {reflection_output.get('issues', [])}\nSuggestions: {reflection_output.get('suggestions', [])}"
            
        # Prepare input
        user_input = (
            f"Query: {state['query']}\n\n"
            f"Retrieved Documents:\n{docs_context}\n\n"
            f"Tool Outputs:\n{tools_context}\n\n"
            f"{reflection_feedback}"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]
        
        # Get LLM (streaming tokens could be handled here in Phase 10)
        llm = get_llm(temperature=0.2)
        
        # Call LLM
        response = await llm.ainvoke(messages)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_agent_step(
            logger,
            step_name,
            "Reasoning draft completed",
            duration_ms=duration_ms,
            response_length=len(response.content)
        )
        
        return {
            "reasoning_output": response.content,
            "needs_revision": False  # Reset for next loop if any
        }
        
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Reasoning error: {str(e)}", exc_info=True)
        return {
            "error": f"Reasoning error: {str(e)}"
        }
