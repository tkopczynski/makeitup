"""Schema inference tool using LLM."""

import logging
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import config

logger = logging.getLogger(__name__)


@tool
def infer_schema_tool(description: str) -> str:
    """
    Infer a YAML schema from a natural language description of the dataset.

    Args:
        description: Natural language description of the dataset to generate (e.g., "sales data with store ID, date, and weekly sales amount")

    Returns:
        YAML schema string that can be used with generate_data_tool
    """
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.SCHEMA_INFERENCE_TEMPERATURE)

    prompt = f"""Based on the following description, generate a YAML schema for data generation.

Description: {description}

The schema should be in YAML format with the following structure:
- name: column_name
  type: one of (int, float, date, datetime, category, text, email, phone, name, address, company, product, uuid, bool, currency, percentage)
  config:
    # Optional configuration based on type
    min: value (for int/float/currency/percentage)
    max: value (for int/float/currency/percentage)
    precision: digits (for float)
    categories: [list] (for category)
    start_date: "YYYY-MM-DD" (for date/datetime)
    end_date: "YYYY-MM-DD" (for date/datetime)
    text_type: first_name|last_name|full_name|street|city|state|zip|country (for name/address)

IMPORTANT: Return ONLY the raw YAML schema. Do not include markdown code blocks, backticks, or any formatting. Do not add any explanation before or after the YAML."""

    response = llm.invoke([HumanMessage(content=prompt)])
    response_content = response.content.strip()

    logger.info(f"Inferred schema:\n{response_content}")

    return response_content
