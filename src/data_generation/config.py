"""Configuration for dataset generation."""

# LLM Configuration
LLM_MODEL = "gpt-4o-mini"

# Tool-specific temperatures
SCHEMA_INFERENCE_TEMPERATURE = 0.2  # Low temperature for consistent schema generation

# Agent Configuration
AGENT_VERBOSE = True  # Set to True to see agent reasoning and tool calls

# Reproducibility Configuration
DEFAULT_SEED = None  # None means random generation
SEED_MIN = 100000  # 6-digit codes for memorability
SEED_MAX = 999999
