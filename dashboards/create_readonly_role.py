"""Create (or reset) the read-only PostgreSQL login `powerbi_ro` used by Power BI Desktop.

Run:  python -m dashboards.create_readonly_role
Connects with the admin URL in .env, asks for the new login's password at a hidden prompt, and grants SELECT only
(public, ops and the Phase 7b replay schemas). The password is sent to Postgres once and never written or printed.
Power BI keeps it in the Windows credential store.
"""
import sys
from getpass import getpass

from psycopg import sql

from pipeline.db import get_engine

ROLE = "powerbi_ro"


def main() -> int:
    password = getpass(f"New password for {ROLE} (hidden): ")
    if not password or password != getpass("Repeat it (hidden): "):
        print("Passwords empty or different; nothing changed.")
        return 1
    raw = get_engine().raw_connection()
    try:
        cur = raw.cursor()
        cur.execute("SELECT current_database()")
        db = cur.fetchone()[0]
        cur.execute("SELECT nspname FROM pg_namespace WHERE nspname IN ('public', 'ops') OR nspname LIKE 'ops7b%' ORDER BY 1")
        schemas = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (ROLE,))
        verb = "ALTER" if cur.fetchone() else "CREATE"
        cur.execute(sql.SQL(verb + " ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}").format(
            sql.Identifier(ROLE), sql.Literal(password)))
        cur.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(sql.Identifier(ROLE)))
        cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(db), sql.Identifier(ROLE)))
        for s in schemas:
            cur.execute(sql.SQL("GRANT USAGE ON SCHEMA {0} TO {1}; GRANT SELECT ON ALL TABLES IN SCHEMA {0} TO {1}").format(
                sql.Identifier(s), sql.Identifier(ROLE)))
        raw.commit()
    finally:
        raw.close()
    print(f"{verb.title()}d {ROLE}: read-only, SELECT on {len(schemas)} schemas ({', '.join(schemas)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
