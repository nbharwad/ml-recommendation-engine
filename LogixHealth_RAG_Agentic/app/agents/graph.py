"""LangGraph workflow assembly."""

from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.audit import audit_agent
from app.agents.planner import planner_agent
from app.agents.reasoning import reasoning_agent
from app.agents.reflection import reflection_agent
from app.agents.retrieval_agent import retrieval_agent
from app.agents.state import AgentState
from app.agents.tool_agent import tool_agent
from app.config import settings


def router(state: AgentState) -> list[str] | str:
    """Routes the workflow based on the planner's execution plan."""
    plan = state.get("execution_plan")
    
    if plan == "retrieval_only":
        return "retrieval"
    elif plan == "tool_only":
        return "tools"
    elif plan == "retrieval_and_tools":
        return ["retrieval", "tools"]  # Parallel fan-out
    elif plan == "clarification_needed":
        return END
    else:
        # Fallback to safest path
        return ["retrieval", "tools"]


def reflection_router(state: AgentState) -> str:
    """Routes based on whether reflection thinks revision is needed."""
    if state.get("needs_revision"):
        return "reasoning"
    return "audit"


def audit_router(state: AgentState) -> str:
    """Routes based on whether audit passed and retry limit."""
    if not state.get("audit_passed"):
        if state.get("audit_retry_count", 0) < settings.max_audit_retries:
            return "reasoning"
    return END


def build_graph() -> StateGraph:
    """Assembles the LangGraph workflow.
    
    Returns:
        A compiled StateGraph.
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("planner", planner_agent)
    workflow.add_node("retrieval", retrieval_agent)
    workflow.add_node("tools", tool_agent)
    workflow.add_node("reasoning", reasoning_agent)
    workflow.add_node("reflection", reflection_agent)
    workflow.add_node("audit", audit_agent)

    # 2. Add Edges and Routing
    workflow.add_edge(START, "planner")

    # Planner Conditional Router
    workflow.add_conditional_edges("planner", router)

    # Fan-in to Reasoning
    workflow.add_edge("retrieval", "reasoning")
    workflow.add_edge("tools", "reasoning")

    # Reasoning to Reflection
    workflow.add_edge("reasoning", "reflection")

    # Reflection Conditional Router
    workflow.add_conditional_edges("reflection", reflection_router)

    # Audit Conditional Router
    workflow.add_conditional_edges("audit", audit_router)

    # 3. Compile with Memory Checkpointer for debugging
    return workflow.compile(checkpointer=MemorySaver())
