"""Create .env with the PostgreSQL connection string, then test the connection.

Run:  python -m pipeline.setup_env
The password is typed at a hidden prompt, written only to the git-ignored .env file, and never printed.
"""
from getpass import getpass
from pathlib import Path
from urllib.parse import quote_plus
import sys

from sqlalchemy import create_engine, text

ENV = Path(".env")


def main() -> int:
    if ENV.exists() and input(".env already exists. Overwrite it? [y/N] ").strip().lower() != "y":
        print("Left .env unchanged.")
        return 0

    user = input("PostgreSQL user [postgres]: ").strip() or "postgres"
    password = getpass("PostgreSQL password (hidden): ")
    database = input("Database [pdm]: ").strip() or "pdm"

    # quote_plus makes characters like @ # / : safe inside the URL
    url = f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}@localhost:5432/{database}"

    try:
        with create_engine(url).connect() as conn:
            version = conn.execute(text("SELECT version()")).scalar_one()
    except Exception as exc:  # show the reason without echoing the URL (it contains the password)
        print("Connection failed:", type(exc).__name__, str(exc).splitlines()[0].replace(password, "****"))
        print(".env was NOT written. Check the password and that the database exists, then run this again.")
        return 1

    ENV.write_text(f"DATABASE_URL={url}\n", encoding="utf-8")
    print("Connected:", version.split(",")[0])
    print("Wrote .env (git-ignored). You're ready for Phase 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
