"""Tool Agent - selects and executes domain tools via function calling."""

import time
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.agents.state import AgentState
from app.dependencies import get_llm
from app.observability.logger import get_logger, log_agent_step
from app.tools import TOOL_REGISTRY

logger = get_logger(__name__)


async def tool_agent(state: AgentState) -> dict[str, Any]:
    """Uses LLM function calling to select and execute tools.
    
    Args:
        state: The current agent state.
        
    Returns:
        A dictionary containing the tool_outputs.
    """
    start_time = time.perf_counter()
    step_name = "tools"
    
    try:
        # Convert our tools to LangChain/OpenAI format
        lc_tools = [tool.to_langgraph_tool() for tool in TOOL_REGISTRY.values()]
        
        # Get LLM and bind tools
        llm = get_llm(temperature=0.0)
        llm_with_tools = llm.bind_tools(lc_tools)
        
        # Call LLM to decide which tools to use
        # In a real scenario, we might want to use rewritten_query if available
        query = state.get("rewritten_query") or state["query"]
        messages = [HumanMessage(content=query)]
        
        response = await llm_with_tools.ainvoke(messages)
        
        tool_outputs = []
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Retrieve actual tool from registry
                # Note: tool_name might need mapping if names differ
                tool = TOOL_REGISTRY.get(tool_name)
                
                if tool:
                    log_agent_step(
                        logger,
                        step_name,
                        f"Executing tool: {tool_name}",
                        tool_args=tool_args
                    )
                    
                    # Execute tool
                    result = tool.safe_execute(**tool_args)
                    
                    tool_outputs.append({
                        "tool_name": tool_name,
                        "input": tool_args,
                        "output": result.output,
                        "status": result.status,
                        "error_message": result.error_message
                    })
                else:
                    tool_outputs.append({
                        "tool_name": tool_name,
                        "status": "error",
                        "error_message": f"Tool '{tool_name}' not found in registry."
                    })
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        log_agent_step(
            logger,
            step_name,
            f"Executed {len(tool_outputs)} tools",
            duration_ms=duration_ms,
            num_tools=len(tool_outputs)
        )
        
        return {
            "tool_outputs": tool_outputs
        }
        
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(f"Tool Agent error: {str(e)}", exc_info=True)
        return {
            "error": f"Tool Agent error: {str(e)}"
        }
