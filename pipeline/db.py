"""Database connection and schema setup."""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text

SCHEMA_FILE = Path("pipeline/schema.sql")


def get_engine() -> Engine:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Run `python -m pipeline.setup_env` first.")
    return create_engine(url)


def apply_schema(engine: Engine, schema: str | None = None) -> None:
    """Create the tables (idempotent). `schema` puts them in a separate Postgres schema, used by tests."""
    with engine.begin() as conn:
        if schema:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        conn.exec_driver_sql(SCHEMA_FILE.read_text(encoding="utf-8"))
