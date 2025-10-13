# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
A CLI application for generating synthetic datasets using LangChain and OpenAI's GPT models. The application uses a LangGraph ReAct agent to intelligently generate data based on natural language requests.

## Project Structure

The project follows the src-layout pattern for better organization and packaging:

```
data_generation/
├── src/data_generation/     # Main package
│   ├── __init__.py
│   ├── __main__.py          # Allows: python -m data_generation
│   ├── cli.py               # CLI interface (Click)
│   ├── config.py            # Configuration settings
│   ├── core/                # Core business logic
│   │   ├── agent.py         # LangGraph ReAct agent
│   │   └── generator.py     # Data generation engine
│   ├── tools/               # LangChain tools
│   │   ├── schema_inference.py
│   │   └── schema_validation.py
│   └── utils/               # Utilities
│       └── logging.py
├── tests/                   # Tests mirror src structure
├── examples/                # Usage examples
└── pyproject.toml          # Project configuration
```

## Setup and Environment

1. **Virtual environment setup:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Linux/macOS
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -e ".[dev]"
   ```
   Dependencies are managed in `pyproject.toml` - always use this file for dependency management.

3. **Environment configuration:**
   - Copy `.env.example` to `.env`
   - Add OpenAI API key to `.env`: `OPENAI_API_KEY=your-key`

## Running the Application

After installation, you can run the CLI in several ways:

```bash
# Using the installed command
data-generation "Generate 100 users with names and emails"

# Using python module
python -m data_generation "Generate 100 users with names and emails"

# From the old main.py (still works for backward compatibility)
python main.py "Generate 100 users with names and emails"
```

## Development Notes

- Python >= 3.12 required
- The project uses src-layout for better import handling and testing
- Import paths use absolute imports: `from data_generation.core import agent`
- Configuration is centralized in `src/data_generation/config.py`
- Tests can be run with `pytest` from the project root
- The agent supports generating related tables using the `reference` type

## Code Quality

Ruff is used for linting and formatting:

```bash
# Check for issues
ruff check src/

# Auto-fix issues
ruff check --fix src/

# Format code
ruff format src/
```

Configuration is in `pyproject.toml` under `[tool.ruff]`

## Key Features

- **Reference Type**: Generate related tables with foreign key relationships
- **LangGraph Agent**: Autonomous agent that plans and executes data generation
- **Comprehensive Schema**: Support for 17+ data types including references
- **Logging**: Detailed logging of agent decisions and data generation steps
