"""Agent state definition for LangGraph."""

import operator
from typing import Annotated, Any, NotRequired, TypedDict


class AgentState(TypedDict):
    """State managed by the LangGraph workflow."""

    # Input/Context fields
    query: str
    rewritten_query: NotRequired[str | None]
    trace_id: str
    session_id: str
    memory_context: NotRequired[str]
    complexity: NotRequired[str]  # "simple" or "complex"

    # Planner fields
    execution_plan: NotRequired[str | None]  # "retrieval_only", "tool_only", "retrieval_and_tools", "clarification_needed"

    # Retrieval fields (Annotated with operator.add to allow concurrent appends)
    retrieved_docs: Annotated[list[dict[str, Any]], operator.add]

    # Tool fields (Annotated with operator.add)
    tool_outputs: Annotated[list[dict[str, Any]], operator.add]

    # Reasoning fields
    reasoning_output: NotRequired[str | None]
    reflection_output: NotRequired[dict[str, Any] | None]
    needs_revision: NotRequired[bool]

    # Audit fields
    audit_passed: NotRequired[bool]
    audit_retry_count: NotRequired[int]
    final_answer: NotRequired[str | None]
    citations: Annotated[list[dict[str, Any]], operator.add]
    confidence_score: NotRequired[float]

    # Metadata & Error fields
    metadata: NotRequired[dict[str, Any]]
    error: NotRequired[str | None]
