# Dataset Generation CLI

A CLI application for generating synthetic datasets using LangChain and OpenAI's GPT models. The application uses a LangGraph ReAct agent to intelligently generate data based on natural language requests.

## Setup

1. **Create and activate virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -e ".[dev]"
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-actual-api-key
   ```

## Usage

After installation, you can run the CLI in several ways:

```bash
# Using the installed command
data-generation "Generate 100 users with names and emails"

# Using python module
python -m data_generation "Generate 100 users with names and emails"
```

Get help:
```bash
data-generation --help
# or
python -m data_generation --help
```

## Requirements

- Python >= 3.12
- OpenAI API key

## Key Features

- **LangGraph Agent**: Autonomous agent that plans and executes data generation
- **Reference Type**: Generate related tables with foreign key relationships
- **Comprehensive Schema**: Support for 17+ data types
- **Natural Language Interface**: Describe your data needs in plain English
