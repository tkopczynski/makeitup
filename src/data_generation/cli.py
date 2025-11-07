"""Dataset generation CLI with LangChain and OpenAI."""

import click
from dotenv import load_dotenv

from data_generation.core.agent import run_agent
from data_generation.utils.logging import setup_logging

load_dotenv()
setup_logging()


@click.command()
@click.argument("request", nargs=-1, required=True)
@click.option(
    "--seed",
    "--reproducibility-code",
    type=int,
    default=None,
    help="Reproducibility code (6-digit number) to generate the same data. "
    "Leave blank for random generation.",
)
@click.version_option(version="0.1.0")
def main(request, seed):
    """
    Generate synthetic datasets using natural language requests.

    Examples:
        # Random generation (different every time)
        data-generation "100 users with emails"

        # Reproducible generation (same data every time)
        data-generation "100 users with emails" --seed 123456

        # Using the full parameter name
        data-generation "100 users" --reproducibility-code 987654

    Reproducibility:
        Every generation produces a 6-digit reproducibility code.
        Use this code with --seed to recreate the exact same dataset later.
    """
    user_request = " ".join(request)

    if seed:
        click.echo(f"Processing request (Reproducibility Code: {seed}): {user_request}\n")
    else:
        click.echo(f"Processing request: {user_request}\n")

    try:
        result = run_agent(user_request, seed)
        click.echo(f"\n{result}")
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise click.Abort() from e


if __name__ == "__main__":
    main()
