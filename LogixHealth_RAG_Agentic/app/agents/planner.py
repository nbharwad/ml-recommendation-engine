"""Planner Agent - classifies intent and determines execution path."""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.prompts import load_prompt
from app.agents.state import AgentState
from app.dependencies import get_llm
from app.observability.logger import get_logger, log_agent_step

logger = get_logger(__name__)


async def planner_agent(state: AgentState) -> dict[str, Any]:
    """Classifies user intent and selects the execution plan.
    
    Args:
        state: The current agent state.
        
    Returns:
        A dictionary containing the updated execution_plan and complexity.
    """
    start_time = time.perf_counter()
    step_name = "planner"
    
    try:
        # Load system prompt
        system_prompt = load_prompt("planner_system")
        
        # Prepare messages
        # Include conversational context if available
        memory_context = state.get("memory_context", "")
        user_input = f"User Query: {state['query']}\n\n{memory_context}"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]
        
        # Get LLM (using JSON mode)
        llm = get_llm(json_mode=True)
        
        # Call LLM
        response = await llm.ainvoke(messages)
        
        # Parse output
        plan_data = json.loads(response.content)
        
        execution_plan = plan_data.get("execution_plan", "retrieval_and_tools")
        complexity = plan_data.get("complexity", "simple")
        reasoning = plan_data.get("reasoning", "")
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_agent_step(
            logger,
            step_name,
            f"Planner selected plan: {execution_plan} (complexity: {complexity})",
            duration_ms=duration_ms,
            execution_plan=execution_plan,
            complexity=complexity,
            reasoning=reasoning
        )
        
        return {
            "execution_plan": execution_plan,
            "complexity": complexity,
            "metadata": {"planner_reasoning": reasoning}
        }
        
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Planner error: {str(e)}", exc_info=True)
        
        # Fallback to safest plan on error
        return {
            "execution_plan": "retrieval_and_tools",
            "complexity": "simple",
            "error": f"Planner error: {str(e)}"
        }
