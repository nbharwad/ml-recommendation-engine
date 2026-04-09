"""CPT code lookup tool."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import BaseTool, register_tool


class CPTLookupInput(BaseModel):
    """Input for CPT code lookup."""

    cpt_code: str = Field(description="The 5-digit CPT procedure code (e.g., 99213, 27447)")


class CPTLookupOutput(BaseModel):
    """Output for CPT code lookup."""

    code: str = Field(description="The CPT code")
    short_desc: str = Field(description="Short description of the procedure")
    long_desc: str = Field(description="Detailed description of the procedure")
    category: str = Field(description="Code category (E/M, Surgery, etc.)")
    rvu_work: float = Field(description="Work Relative Value Unit (RVU) for the procedure")
    rvu_facility: float = Field(description="Facility Practice Expense RVU for the procedure")
    rvu_nonfacility: float = Field(
        description="Non-Facility Practice Expense RVU for the procedure"
    )
    denial_reasons: list[str] = Field(description="Common denial reasons for this CPT code")


@register_tool
class CPTLookupTool(BaseTool):
    """Tool for fetching CPT code metadata."""

    def __init__(self, data_path: str | None = None):
        """Initialize with data from JSON."""
        if data_path is None:
            data_path = str(Path(__file__).parent / "data" / "cpt_codes.json")
        self._data_path = data_path
        self._data: dict[str, dict[str, Any]] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load data from JSON file."""
        with open(self._data_path) as f:
            codes_list = json.load(f)
            self._data = {c["code"]: c for c in codes_list}

    @property
    def name(self) -> str:
        return "cpt_lookup"

    @property
    def description(self) -> str:
        return "Provides descriptions, categories, and RVU data for CPT procedure codes."

    @property
    def input_schema(self) -> type[BaseModel]:
        return CPTLookupInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return CPTLookupOutput

    def execute(self, cpt_code: str) -> dict[str, Any]:
        """Look up the CPT code in the knowledge base."""
        code_str = cpt_code.strip()

        if code_str not in self._data:
            raise ValueError(f"Unknown CPT code: {cpt_code}")

        return self._data[code_str]
