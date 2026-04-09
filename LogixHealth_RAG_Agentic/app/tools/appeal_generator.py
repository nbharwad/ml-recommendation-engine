"""Appeal letter generator tool."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import BaseTool, register_tool


class ProviderInfo(BaseModel):
    """Provider information model."""

    name: str = Field(description="Name of the provider")
    npi: str = Field(description="National Provider Identifier (NPI)")


class AppealGeneratorInput(BaseModel):
    """Input for appeal letter generator."""

    patient_name: str = Field(description="Name of the patient")
    claim_number: str = Field(description="Original claim reference number")
    date_of_service: str = Field(description="Date the service was provided (YYYY-MM-DD)")
    cpt_code: str = Field(description="CPT code being appealed")
    denial_code: str = Field(description="CARC denial code being appealed")
    payer_name: str = Field(description="Name of the insurance payer")
    clinical_justification: str = Field(
        description="Detailed clinical explanation justifying the service"
    )
    provider_name: str = Field(description="Name of the healthcare provider")
    provider_npi: str = Field(description="NPI of the provider")
    timely_filing_days: int | None = Field(
        default=180, description="The number of days allowed for timely filing by the payer"
    )


class AppealGeneratorOutput(BaseModel):
    """Output for appeal letter generator."""

    letter_text: str = Field(description="The complete generated appeal letter text")
    template_id: str = Field(description="The ID of the template used")
    required_attachments: list[str] = Field(
        description="List of documents to attach with the appeal"
    )
    deadline_days: int = Field(description="Standard number of days allowed for this appeal type")


@register_tool
class AppealGeneratorTool(BaseTool):
    """Tool for generating structured appeal letters from templates."""

    def __init__(self, data_path: str | None = None):
        """Initialize with data from JSON."""
        if data_path is None:
            data_path = str(Path(__file__).parent / "data" / "appeal_templates.json")
        self._data_path = data_path
        self._templates: list[dict[str, Any]] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load data from JSON file."""
        with open(self._data_path) as f:
            self._templates = json.load(f)

    @property
    def name(self) -> str:
        return "appeal_generator"

    @property
    def description(self) -> str:
        return "Generates a structured, professional appeal letter based on templates for specific denial reasons."

    @property
    def input_schema(self) -> type[BaseModel]:
        return AppealGeneratorInput

    @property
    def output_schema(self) -> type[BaseModel]:
        return AppealGeneratorOutput

    def execute(
        self,
        patient_name: str,
        claim_number: str,
        date_of_service: str,
        cpt_code: str,
        denial_code: str,
        payer_name: str,
        clinical_justification: str,
        provider_name: str,
        provider_npi: str,
        timely_filing_days: int = 180,
    ) -> dict[str, Any]:
        """Generate the appeal letter."""
        # Selection logic: match denial code to category
        # For simplicity, we'll try to find a template matching the denial code group or a default
        category = "medical necessity"
        if denial_code.upper().startswith(("CO-4", "CO-16")):
            category = "coding error"
        elif denial_code.upper().startswith("CO-29"):
            category = "timely filing"
        elif denial_code.upper().startswith("CO-197"):
            category = "authorization"

        template = next(
            (t for t in self._templates if t["category"] == category), self._templates[0]
        )

        # Mapping for placeholder replacement
        mapping = {
            "[Date]": datetime.now().strftime("%B %d, %Y"),
            "[Payer Name]": payer_name,
            "[Appeal Department Address]": "[Payer Appeal Address - Found in Policy]",
            "[Patient Name]": patient_name,
            "[Claim Number]": claim_number,
            "[Date of Service]": date_of_service,
            "[CPT Code]": cpt_code,
            "[Denial Code]": denial_code,
            "[Clinical Justification]": clinical_justification,
            "[Provider Name]": provider_name,
            "[Provider NPI]": provider_npi,
            "[Timely Filing Days]": str(timely_filing_days),
            "[Date of Initial Submission]": "[Initial Submission Date]",
        }

        letter_text = template["letter_text"]
        for placeholder, value in mapping.items():
            letter_text = letter_text.replace(placeholder, value)

        return {
            "letter_text": letter_text,
            "template_id": template["template_id"],
            "required_attachments": template["required_attachments"],
            "deadline_days": template["deadline_days"],
        }
