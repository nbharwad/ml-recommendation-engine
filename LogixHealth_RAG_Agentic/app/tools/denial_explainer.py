"""Denial code explainer tool."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import BaseTool, register_tool


class DenialExplainerInput(BaseModel):
    """Input for denial code explainer."""

    denial_code: str = Field(description="The CARC/RARC denial code (e.g., CO-4, PR-96)")


class DenialExplainerOutput(BaseModel):
    """Output for denial code explainer."""

    code: str = Field(description="The denial code")
    group: str = Field(description="Code group (CO, PR, OA, etc.)")
    category: str = Field(description="High-level category of the denial")
    description: str = Field(description="Full description of the code")
    causes: list[str] = Field(description="Common causes for this denial")
    actions: list[str] = Field(description="Recommended next steps/actions")
    appeal_likelihood: str = Field(
        description="Likelihood of successful appeal (high, medium, low)"
    )


@register_tool
class DenialExplainerTool(BaseTool):
    """Tool for explaining CARC/RARC denial codes."""

    def __init__(self, data_path: str | None = None):
        """Initialize with data from JSON."""
        if data_path is None:
            data_path = str(Path(__file__).parent / "data" / "denial_codes.json")
        self._data_path = data_path
        self._data: dict[str, dict[str, Any]] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load data from JSON file."""
        with open(self._data_path) as f:
            codes_list = json.load(f)
            self._data = {c["code"].upper(): c for c in codes_list}

    @property
    def name(self) -> str:
        return "denial_explainer"

    @property
    def description(self) -> str:
        return "Explains CARC/RARC denial codes, their causes, and recommended actions."

    @property
    def input_schema(self) -> type[BaseModel]:
        return DenialExplainerInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return DenialExplainerOutput

    def execute(self, denial_code: str) -> dict[str, Any]:
        """Look up the denial code in the knowledge base."""
        code_upper = denial_code.upper().strip()

        if code_upper not in self._data:
            raise ValueError(f"Unknown denial code: {denial_code}")

        return self._data[code_upper]
