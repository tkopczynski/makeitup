"""LangChain tool for data generation."""

import logging
from pathlib import Path

import pandas as pd
import yaml
from langchain_core.tools import tool

from data_generation.core.generator import generate_data
from data_generation.tools.schema_validation import SchemaValidationError

logger = logging.getLogger(__name__)


@tool
def generate_data_tool(
    schema_yaml: str,
    num_rows: int,
    output_file: str = "generated_data.csv",
    seed: int | None = None,
) -> str:
    """
    Generate synthetic data based on a YAML schema and save to a CSV file.

    Args:
        schema_yaml: YAML string defining the data structure. Format:
            - name: column_name
              type: int|float|date|datetime|category|text|email|phone|name|address|
                    company|product|uuid|bool|currency|percentage|reference
              config:
                min: value (for int/float/currency/percentage)
                max: value (for int/float/currency/percentage)
                precision: digits (for float)
                categories: [list] (for category)
                start_date: "YYYY-MM-DD" (for date/datetime)
                end_date: "YYYY-MM-DD" (for date/datetime)
                text_type: first_name|last_name|full_name|street|city|state|zip|country
                           (for name/address)
                reference_file: path/to/file.csv (for reference - REQUIRED)
                reference_column: column_name (for reference - REQUIRED)
        num_rows: Number of rows to generate
        output_file: Path to save the CSV file (default: generated_data.csv)
        seed: Reproducibility code (optional, auto-generated if not provided)

    Returns:
        Message with file path and reproducibility code
    """
    logger.info(
        f"generate_data_tool called: output_file='{output_file}', num_rows={num_rows}, seed={seed}"
    )

    try:
        # Parse YAML schema
        schema = yaml.safe_load(schema_yaml)
        logger.debug(f"Parsed schema with {len(schema)} columns")

        # Generate data with reproducibility
        logger.info(f"Generating {num_rows} rows of data...")
        data, seed_used = generate_data(schema, num_rows, seed)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Save to CSV
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)

        logger.info(
            f"Successfully generated {num_rows} rows and saved to {output_path.absolute()}"
        )

        return (
            f"Successfully generated {num_rows} rows and saved to {output_path.absolute()}\n"
            f"Reproducibility Code: {seed_used}\n"
            f"(Use --seed {seed_used} to recreate this exact dataset)"
        )

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML schema: {e}") from e
    except SchemaValidationError as e:
        raise ValueError(f"Schema validation error: {e}") from e
    except Exception as e:
        raise ValueError(f"Error generating data: {e}") from e
