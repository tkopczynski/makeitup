"""Agent for autonomous data generation."""

import logging

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from data_generation import config
from data_generation.tools.data_generation import generate_data_tool
from data_generation.tools.schema_inference import infer_schema_tool

logger = logging.getLogger(__name__)


def create_data_generation_agent(seed: int | None = None):
    """
    Create a LangGraph ReAct agent for data generation that can handle multiple files.

    Args:
        seed: Optional reproducibility code to pass to generation tools

    Returns:
        Compiled LangGraph agent configured with data generation tools
    """
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    tools = [infer_schema_tool, generate_data_tool]

    seed_instruction = ""
    if seed is not None:
        seed_instruction = f"""
REPRODUCIBILITY:
- A reproducibility code ({seed}) has been provided
- Pass this code as the 'seed' parameter to generate_data_tool
- This ensures the exact same data is generated every time"""
    else:
        seed_instruction = """
REPRODUCIBILITY:
- No reproducibility code provided, data will be randomly generated
- The generate_data_tool will automatically create a reproducibility code
- Include the reproducibility code in your final response to the user"""

    system_message = f"""You are a data generation assistant. \
You help users generate synthetic datasets.

When the user requests data generation:
1. Identify how many datasets/files they want to generate
2. Determine if datasets have relationships (e.g., foreign keys)
3. For EACH dataset:
   a. Use infer_schema_tool with a description of the data
   b. Use generate_data_tool with the schema, number of rows, output file, and seed
4. Continue until all datasets are generated

IMPORTANT INSTRUCTIONS:
- If num_rows is not specified for a dataset, use 100 as default
- If output file is not specified, use "generated_data.csv" for single file or \
descriptive names for multiple files
- For generate_data_tool, the input MUST be valid JSON with keys: \
schema_yaml, num_rows, output_file, seed
- Use the EXACT schema_yaml from infer_schema_tool output (as a string)
{seed_instruction}

RELATIONSHIPS BETWEEN TABLES:
- When generating related tables (e.g., users and transactions), generate the PARENT \
table FIRST (e.g., users.csv), THEN the child table (e.g., transactions.csv)
- For child tables, use the 'reference' type to link to parent tables
- When describing the schema for a child table, mention the relationship explicitly \
(e.g., "transactions with user_id referencing users.csv")
- The reference type requires reference_file (path to parent CSV) and reference_column \
(column name in parent CSV)"""

    agent = create_react_agent(llm, tools, prompt=system_message)

    return agent


def run_agent(user_request: str, seed: int | None = None) -> str:
    """
    Run the LangGraph ReAct agent with a user request.

    Args:
        user_request: Natural language request for data generation
        seed: Optional reproducibility code for deterministic generation

    Returns:
        Agent's response with reproducibility information
    """
    agent = create_data_generation_agent(seed)

    try:
        if seed is not None:
            logger.info(f"Processing request with reproducibility code {seed}: {user_request}")
        else:
            logger.info(f"Processing request: {user_request}")

        result = agent.invoke({"messages": [("user", user_request)]})

        # Extract the final message from the agent
        messages = result["messages"]
        final_message = messages[-1]

        return final_message.content
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise
