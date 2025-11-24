"""Core LLM-based data generation engine."""

import logging
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, create_model

from makeitup import config

logger = logging.getLogger(__name__)


def _build_prompt(
    columns: dict[str, str],
    target: dict[str, str] | None,
    num_rows: int,
    quality_issues: list[str] | None = None,
) -> str:
    """Build the prompt for data generation.

    Args:
        columns: Dictionary mapping column names to their descriptions
        target: Optional target column with 'name' and 'prompt' keys
        num_rows: Number of rows to generate
        quality_issues: Optional list of quality issues to introduce

    Returns:
        Formatted prompt string
    """
    column_descriptions = "\n".join(
        f"- {name}: {description}" for name, description in columns.items()
    )

    target_section = ""
    if target:
        target_section = f"\n- {target['name']} (target): {target['prompt']}"

    quality_section = ""
    if quality_issues:
        issue_descriptions = []
        if "nulls" in quality_issues:
            issue_descriptions.append(
                "- Include null/None values randomly in some fields (about 5-10% of values)"
            )
        if "outliers" in quality_issues:
            issue_descriptions.append(
                "- Include outlier values that are unusually high or low for numeric fields"
            )
        if "typos" in quality_issues:
            issue_descriptions.append("- Include occasional typos or misspellings in text fields")
        if "duplicates" in quality_issues:
            issue_descriptions.append("- Include some duplicate rows (about 5-10% of rows)")
        quality_section = "\n\nData quality issues to introduce:\n" + "\n".join(issue_descriptions)

    return f"""Generate a dataset with exactly {num_rows} rows containing the following columns:

{column_descriptions}{target_section}{quality_section}

Ensure variety and realism in the generated values."""


def _create_dataset_model(
    columns: dict[str, str],
    target: dict[str, str] | None,
    num_rows: int,
) -> type[BaseModel]:
    """Create a dynamic Pydantic model for the dataset.

    Args:
        columns: Dictionary mapping column names to their descriptions
        target: Optional target column with 'name' and 'prompt' keys
        num_rows: Number of rows to generate

    Returns:
        Pydantic model class for the dataset
    """
    # Build field definitions for a single row
    field_definitions = {}
    for col_name, col_description in columns.items():
        field_definitions[col_name] = (Any, Field(description=col_description))

    if target:
        field_definitions[target["name"]] = (Any, Field(description=target["prompt"]))

    # Create the row model
    RowModel = create_model("DataRow", **field_definitions)

    # Create the dataset model (wrapper for array of rows)
    class Dataset(BaseModel):
        rows: list[RowModel] = Field(
            description=f"Array of exactly {num_rows} data rows with variety and realism"
        )

    return Dataset


def generate_dataset_with_llm(
    columns: dict[str, str],
    num_rows: int,
    target: dict[str, str] | None = None,
    quality_issues: list[str] | None = None,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
) -> list[dict[str, Any]]:
    """Generate synthetic data using LLM.

    Args:
        columns: Dictionary mapping column names to their descriptions.
                 Example: {"age": "Age of employees, 25-55", "salary": "Annual salary in USD"}
        num_rows: Number of rows to generate
        target: Optional target column dict with 'name' and 'prompt' keys.
                Example: {"name": "will_churn", "prompt": "Boolean indicating customer churn"}
        quality_issues: Optional list of quality issues to introduce.
                        Supported: "nulls", "outliers", "typos", "duplicates"
        model: LLM model name. Defaults to config.LLM_MODEL.
        base_url: Base URL for OpenAI-compatible API. Defaults to config.LLM_BASE_URL.
        api_key: API key for the LLM service. Defaults to config.LLM_API_KEY.
        temperature: Sampling temperature. Defaults to config.DATA_GENERATION_TEMPERATURE.

    Returns:
        List of dictionaries containing the generated data

    Raises:
        ValueError: If LLM response cannot be parsed
    """
    logger.info(f"Generating {num_rows} rows with columns: {list(columns.keys())}")

    # Build prompt
    prompt = _build_prompt(columns, target, num_rows, quality_issues)
    logger.debug(f"Prompt: {prompt}")

    # Create dynamic Pydantic model for structured output
    DatasetModel = _create_dataset_model(columns, target, num_rows)

    # Call LLM (use provided params, fall back to config)
    effective_temp = temperature if temperature is not None else config.DATA_GENERATION_TEMPERATURE
    llm_kwargs = {
        "model": model if model is not None else config.LLM_MODEL,
        "temperature": effective_temp,
    }
    effective_base_url = base_url if base_url is not None else config.LLM_BASE_URL
    if effective_base_url:
        llm_kwargs["base_url"] = effective_base_url
    effective_api_key = api_key if api_key is not None else config.LLM_API_KEY
    if effective_api_key:
        llm_kwargs["api_key"] = effective_api_key

    llm = ChatOpenAI(**llm_kwargs)
    structured_llm = llm.with_structured_output(DatasetModel, method="function_calling")

    # Invoke with structured output
    response = structured_llm.invoke(prompt)

    logger.debug(f"Received structured response with {len(response.rows)} rows")

    # Convert Pydantic models to list of dicts
    data = [row.model_dump() for row in response.rows]

    # Validate row count
    if len(data) != num_rows:
        logger.warning(f"Expected {num_rows} rows, got {len(data)}")

    logger.info(f"Generated {len(data)} rows successfully")

    return data
