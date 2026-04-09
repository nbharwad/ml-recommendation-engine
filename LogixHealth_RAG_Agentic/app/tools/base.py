"""Base abstractions for domain tools."""

import abc
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Standardized tool output model."""

    status: str = Field(description="Execution status: 'success' or 'error'")
    output: Any = Field(default=None, description="Tool output data")
    error_message: str | None = Field(default=None, description="Error message if failed")


class BaseTool(abc.ABC):
    """Abstract base class for all domain tools."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The unique name of the tool."""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """The description of what the tool does."""
        pass

    @property
    @abc.abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """The Pydantic model for the tool's input."""
        pass

    @property
    @abc.abstractmethod
    def output_schema(self) -> type[BaseModel]:
        """The Pydantic model for the tool's output."""
        pass

    @abc.abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """The core execution logic of the tool."""
        pass

    def safe_execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with error handling and validation."""
        try:
            # 1. Validate input against input_schema
            # This ensures required fields are present and types are correct
            input_data = self.input_schema(**kwargs)

            # 2. Run core execution with validated inputs
            output = self.execute(**input_data.model_dump())

            # 3. Validate output against output_schema
            # Handle cases where execute might return a dict, a model, or other types
            if isinstance(output, self.output_schema):
                validated_output = output
            elif isinstance(output, dict):
                validated_output = self.output_schema(**output)
            else:
                # Fallback for other types if the schema allows them
                validated_output = self.output_schema(output=output)

            return ToolResult(status="success", output=validated_output.model_dump())
        except Exception as e:
            return ToolResult(status="error", error_message=f"{type(e).__name__}: {str(e)}")

    def to_langgraph_tool(self) -> StructuredTool:
        """Convert to a LangChain StructuredTool."""
        return StructuredTool.from_function(
            func=self.execute,
            name=self.name,
            description=self.description,
            args_schema=self.input_schema,
        )


# Global registry for tool discovery
TOOL_REGISTRY: dict[str, BaseTool] = {}


def register_tool(tool_class: type[BaseTool]) -> type[BaseTool]:
    """Decorator to register a tool instance."""
    tool_instance = tool_class()
    TOOL_REGISTRY[tool_instance.name] = tool_instance
    return tool_class
