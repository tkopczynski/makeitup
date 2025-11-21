"""Core LLM-based data generation engine."""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from data_generation import config

logger = logging.getLogger(__name__)


def _build_prompt(
    columns: dict[str, str],
    target: dict[str, str] | None,
    num_rows: int,
) -> str:
    """Build the prompt for data generation.

    Args:
        columns: Dictionary mapping column names to their descriptions
        target: Optional target column with 'name' and 'prompt' keys
        num_rows: Number of rows to generate

    Returns:
        Formatted prompt string
    """
    column_descriptions = "\n".join(
        f"- {name}: {description}" for name, description in columns.items()
    )

    target_section = ""
    if target:
        target_section = f"\n- {target['name']} (target): {target['prompt']}"

    return f"""Generate a dataset with exactly {num_rows} rows containing the following columns:

{column_descriptions}{target_section}

Return ONLY a valid JSON array of objects. No explanation, no markdown, just the JSON array.
Each object must have all the specified columns as keys.
Ensure variety and realism in the generated values."""


def _parse_llm_response(response: str) -> list[dict[str, Any]]:
    """Parse LLM response into list of dictionaries.

    Args:
        response: Raw LLM response string

    Returns:
        List of dictionaries representing rows

    Raises:
        ValueError: If response cannot be parsed as JSON
    """
    # Clean up response - remove markdown code blocks if present
    cleaned = response.strip()
    if cleaned.startswith("```"):
        # Remove first line (```json or ```)
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    return data


def generate_dataset_with_llm(
    columns: dict[str, str],
    num_rows: int,
    target: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Generate synthetic data using LLM.

    Args:
        columns: Dictionary mapping column names to their descriptions.
                 Example: {"age": "Age of employees, 25-55", "salary": "Annual salary in USD"}
        num_rows: Number of rows to generate
        target: Optional target column dict with 'name' and 'prompt' keys.
                Example: {"name": "will_churn", "prompt": "Boolean indicating customer churn"}

    Returns:
        List of dictionaries containing the generated data

    Raises:
        ValueError: If LLM response cannot be parsed
    """
    logger.info(f"Generating {num_rows} rows with columns: {list(columns.keys())}")

    # Build prompt
    prompt = _build_prompt(columns, target, num_rows)
    logger.debug(f"Prompt: {prompt}")

    # Call LLM
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.DATA_GENERATION_TEMPERATURE,
    )

    response = llm.invoke(prompt)
    response_text = response.content

    logger.debug(f"LLM response: {response_text[:500]}...")

    # Parse response
    data = _parse_llm_response(response_text)

    # Validate row count
    if len(data) != num_rows:
        logger.warning(f"Expected {num_rows} rows, got {len(data)}")

    # Validate columns
    expected_columns = set(columns.keys())
    if target:
        expected_columns.add(target["name"])

    for i, row in enumerate(data):
        missing = expected_columns - set(row.keys())
        if missing:
            logger.warning(f"Row {i} missing columns: {missing}")

    logger.info(f"Generated {len(data)} rows successfully")

    return data
