"""Payer policy fetcher tool."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import BaseTool, register_tool


class PolicyFetcherInput(BaseModel):
    """Input for payer policy fetcher."""

    payer_name: str = Field(
        description="The name of the insurance payer (e.g., UnitedHealthcare, Medicare)"
    )
    cpt_code: str | None = Field(default=None, description="Optional CPT code to filter policies")
    denial_code: str | None = Field(
        default=None, description="Optional denial code to filter policies"
    )


class Policy(BaseModel):
    """Single policy model."""

    policy_id: str = Field(description="Unique policy identifier")
    title: str = Field(description="Title of the policy")
    effective_date: str = Field(description="Policy effective date")
    summary: str = Field(description="Brief summary of the policy")
    prior_auth_required: bool = Field(description="Whether prior authorization is required")
    medical_necessity_criteria: str = Field(description="Clinical criteria for medical necessity")
    timely_filing_days: int = Field(description="Deadline for initial claim filing in days")
    appeal_deadline_days: int = Field(description="Deadline for submitting an appeal in days")


class PolicyFetcherOutput(BaseModel):
    """Output for payer policy fetcher."""

    payer: str = Field(description="The payer name")
    policies: list[Policy] = Field(description="List of relevant policies")


@register_tool
class PolicyFetcherTool(BaseTool):
    """Tool for fetching payer-specific policies and rules."""

    def __init__(self, data_path: str | None = None):
        """Initialize with data from JSON."""
        if data_path is None:
            data_path = str(Path(__file__).parent / "data" / "payer_policies.json")
        self._data_path = data_path
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load data from JSON file."""
        with open(self._data_path) as f:
            self._data = json.load(f)

    @property
    def name(self) -> str:
        return "policy_fetcher"

    @property
    def description(self) -> str:
        return "Retrieves payer-specific rules, medical policies, and administrative guidelines."

    @property
    def input_schema(self) -> type[BaseModel]:
        return PolicyFetcherInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return PolicyFetcherOutput

    def execute(
        self, payer_name: str, cpt_code: str | None = None, denial_code: str | None = None
    ) -> dict[str, Any]:
        """Look up the payer policies in the knowledge base."""
        # Fuzzy match payer name
        matched_payer = None
        payer_name_lower = payer_name.lower().strip()

        for p in self._data.keys():
            if payer_name_lower in p.lower() or p.lower() in payer_name_lower:
                matched_payer = p
                break

        if not matched_payer:
            raise ValueError(f"No policies found for payer: {payer_name}")

        policies = self._data[matched_payer]

        # In a real tool, we might filter by CPT or denial code keywords in the title/summary
        if cpt_code:
            policies = [p for p in policies if cpt_code in p["title"] or cpt_code in p["summary"]]

        return {"payer": matched_payer, "policies": policies}
