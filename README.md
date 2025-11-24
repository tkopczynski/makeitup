# makeitup

[![PyPI version](https://badge.fury.io/py/makeitup.svg)](https://pypi.org/project/makeitup/)

Generate synthetic datasets for ML training using LLM. Describe your columns in plain English and get realistic data back.

```python
from makeitup import make

df = make(
    columns={
        "name": "Person's full name",
        "age": "Age between 25 and 55",
        "email": "Work email address",
    },
    num_rows=100
)
```

## Features

- **Plain English columns** - Describe what you want, get realistic data back
- **ML-ready datasets** - Add target columns for classification or regression
- **Data quality testing** - Inject nulls, outliers, typos, or duplicates to test your pipelines
- **Multiple formats** - Export to CSV, JSON, Parquet, or Excel
- **Local model support** - Works with OpenAI and any OpenAI-compatible API that supports structured output

## Installation

```bash
pip install makeitup
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your-api-key
```

Or create a `.env` file in your project with `OPENAI_API_KEY=your-api-key`.

### Using a Local Model

`makeitup` uses **structured output** to ensure reliable data generation. Local models must support OpenAI-compatible structured output (JSON schema enforcement).

**Supported local setups:**
- **llama.cpp** with function calling enabled (llama-server, LM Studio)
- **vLLM** with `--enable-auto-tool-choice`
- **Ollama** (version 0.3.0+) - newer models like llama3.1, qwen2.5
- Any OpenAI-compatible API that implements structured output

**Example configuration:**

```bash
export LLM_BASE_URL=http://localhost:11434/v1  # Ollama
export LLM_MODEL=llama3.1
export LLM_API_KEY=not-needed  # Required by some local servers
```

**Note:** Not all local models support structured output. If you encounter errors, verify your model and server support JSON schema enforcement.

## Examples

### Basic Data

```python
from makeitup import make

# Customer data
df = make(
    columns={
        "customer_id": "Unique customer identifier",
        "name": "Customer full name",
        "email": "Email address",
        "signup_date": "Date when customer signed up, 2020-2024",
    },
    num_rows=100
)
```

### ML Dataset with Target Column

```python
df = make(
    columns={
        "tenure_months": "Months as customer, 1-60",
        "monthly_spend": "Monthly spending in USD, 10-500",
        "support_tickets": "Number of support tickets, 0-10",
    },
    target={
        "name": "churned",
        "prompt": "Boolean indicating if customer churned"
    },
    num_rows=500
)
```

### Data Quality Degradation

```python
# Generate dataset with intentional quality issues for testing data pipelines
df = make(
    columns={
        "name": "Person's full name",
        "age": "Age between 20 and 60",
        "salary": "Annual salary in USD, 30000-150000",
    },
    num_rows=100,
    quality_issues=["nulls", "outliers"],  # Options: nulls, outliers, typos, duplicates
)
```

### Save to File

```python
# CSV, JSON, Parquet, or Excel - format detected from extension
df = make(
    columns={"name": "Product name", "price": "Price in USD, 10-1000"},
    num_rows=200,
    output_path="products.csv"
)
```

## Output Formats

| Format | Extension |
|--------|-----------|
| CSV | `.csv` |
| JSON | `.json` |
| Parquet | `.parquet` |
| Excel | `.xlsx` |

## Requirements

- Python >= 3.12
- OpenAI API key or a local model that supports structured output (see "Using a Local Model" above)

## Documentation

See [DEVELOPER.md](DEVELOPER.md) for technical details, API reference, and development setup.

## License

See LICENSE file for details.
