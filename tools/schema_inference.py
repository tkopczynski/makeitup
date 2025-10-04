"""Schema inference tool using LLM."""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


@tool
def infer_schema_tool(description: str) -> str:
    """
    Infer a YAML schema from a natural language description of the dataset.

    Args:
        description: Natural language description of the dataset to generate (e.g., "sales data with store ID, date, and weekly sales amount")

    Returns:
        YAML schema string that can be used with generate_data_tool
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

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

Return ONLY the YAML schema, no additional text or explanation."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()
