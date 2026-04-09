"""Reflection Agent - critiques the reasoning draft against evidence."""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.prompts import load_prompt
from app.agents.state import AgentState
from app.dependencies import get_llm
from app.observability.logger import get_logger, log_agent_step

logger = get_logger(__name__)


async def reflection_agent(state: AgentState) -> dict[str, Any]:
    """Critiques the draft reasoning response.
    
    Args:
        state: The current agent state.
        
    Returns:
        A dictionary containing the reflection_output and needs_revision flag.
    """
    start_time = time.perf_counter()
    step_name = "reflection"
    
    try:
        # Load system prompt
        system_prompt = load_prompt("reflection_system")
        
        # Prepare evidence context for critique
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
            f"Reasoning Draft:\n{state.get('reasoning_output', '')}\n\n"
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
        reflection_data = json.loads(response.content)
        
        needs_revision = reflection_data.get("needs_revision", False)
        issues = reflection_data.get("issues", [])
        suggestions = reflection_data.get("suggestions", [])
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_agent_step(
            logger,
            step_name,
            f"Reflection completed. Needs revision: {needs_revision}",
            duration_ms=duration_ms,
            num_issues=len(issues)
        )
        
        return {
            "reflection_output": reflection_data,
            "needs_revision": needs_revision
        }
        
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Reflection error: {str(e)}", exc_info=True)
        return {
            "needs_revision": False,  # Fallback: bypass reflection on error
            "error": f"Reflection error: {str(e)}"
        }
