
#!/usr/bin/env python3
"""
Bulk‑load a folder of CSV files (all same schema) into a PostgreSQL table.

Features
--------
- Fast COPY FROM STDIN via psycopg2
- Optional auto‑create table (schema inferred from first CSV)
- Optional truncate before load
- Simple logging + dry‑run mode

Requirements
------------
pip install psycopg2-binary pandas python-dotenv
(or use psycopg2 if you have libpq installed)

Examples
--------
python load_csv_to_postgres.py \
    --folder /path/to/csvs \
    --table public.my_table \
    --host localhost --port 5432 --user myuser --password mypass --db mydb \
    --delimiter "," --encoding "utf-8" --truncate --autocreate

You can also use environment variables instead of flags:
  PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE
"""
import argparse
import os
import sys
import csv
from pathlib import Path
from typing import List

import pandas as pd
import psycopg2
from psycopg2 import sql

# --------------- Helpers ---------------

def infer_pg_type(pd_dtype: str) -> str:
    """Map pandas dtype string to a reasonable PostgreSQL type."""
    # You can tweak this mapping as needed
    if pd_dtype.startswith("int"):
        return "BIGINT"
    if pd_dtype.startswith("float"):
        return "DOUBLE PRECISION"
    if pd_dtype == "bool":
        return "BOOLEAN"
    if "datetime" in pd_dtype:
        return "TIMESTAMP"
    if "date" in pd_dtype:
        return "DATE"
    return "TEXT"  # default/fallback

def infer_schema_from_csv(csv_path: Path, delimiter: str, encoding: str) -> List[str]:
    """Return a list of column defs like ['col1 TEXT', 'col2 BIGINT', ...] inferred from the first CSV."""
    df = pd.read_csv(csv_path, nrows=5000, dtype=str, sep=delimiter, encoding=encoding, keep_default_na=True)
    # Try to infer numeric/bool/datetime
    sample = pd.read_csv(csv_path, nrows=5000, sep=delimiter, encoding=encoding, keep_default_na=True)
    coldefs = []
    for col in df.columns:
        dtype = str(sample[col].infer_objects().dtype)
        # pandas sometimes keeps ints as floats, try to refine
        try:
            # If values are integers when parsed as float with no fractional part -> treat as BIGINT
            s = pd.to_numeric(sample[col], errors="coerce")
            if pd.api.types.is_integer_dtype(s.dropna()):
                dtype = "int64"
            elif pd.api.types.is_float_dtype(s.dropna()):
                dtype = "float64"
        except Exception:
            pass
        # Try datetimes
        if dtype == "object":
            try:
                pd.to_datetime(sample[col].dropna().head(100), errors="raise", infer_datetime_format=True)
                dtype = "datetime64[ns]"
            except Exception:
                pass
        coldefs.append(f'{psql_ident(col)} {infer_pg_type(dtype)}')
    return coldefs

def psql_ident(identifier: str) -> str:
    """Safely quote an identifier (lowercase it for consistency)."""
    # We will lowercase identifiers to avoid case-sensitivity headaches.
    # Surround with double quotes if special chars are present.
    safe = identifier.strip().lower().replace('"', '""')
    return f'"{safe}"'

def table_exists(cur, schema: str, table: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
        """,
        (schema, table)
    )
    return cur.fetchone() is not None

def create_table(cur, full_table: str, columns: List[str]):
    ddl = f"CREATE TABLE IF NOT EXISTS {full_table} (\n  {', '.join(columns)}\n);"
    cur.execute(ddl)

def truncate_table(cur, full_table: str):
    cur.execute(f"TRUNCATE TABLE {full_table};")

def copy_csv(cur, full_table: str, csv_path: Path, delimiter: str, encoding: str):
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        # Use COPY with explicit column list (based on header) to ensure correct mapping
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        columns = ", ".join(psql_ident(c) for c in header)
        f.seek(0)
        copy_sql = f"""
            COPY {full_table} ({columns})
            FROM STDIN WITH (FORMAT csv, HEADER true, DELIMITER '{delimiter}');
        """
        cur.copy_expert(copy_sql, f)

# --------------- Main ---------------

def main():
    ap = argparse.ArgumentParser(description="Bulk‑load a folder of CSVs into PostgreSQL.")
    ap.add_argument("--folder", required=True, help="Folder containing CSV files (all same columns/order)")
    ap.add_argument("--table", required=True, help="Target table in the form schema.table (e.g., public.my_table)")
    ap.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    ap.add_argument("--user", default=os.getenv("PGUSER"))
    ap.add_argument("--password", default=os.getenv("PGPASSWORD"))
    ap.add_argument("--db", default=os.getenv("PGDATABASE"))
    ap.add_argument("--delimiter", default=",")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--truncate", action="store_true", help="Truncate the table before loading")
    ap.add_argument("--autocreate", action="store_true", help="Create the table (schema inferred from first CSV) if it does not exist")
    ap.add_argument("--dry_run", action="store_true", help="Show what would happen without writing any data")

    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"[ERROR] Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    if "." not in args.table:
        print("[ERROR] --table must be schema.table (e.g., public.my_table)", file=sys.stderr)
        sys.exit(1)

    schema, table = args.table.split(".", 1)
    full_table = f'{psql_ident(schema)}.{psql_ident(table)}'

    csv_files = sorted([p for p in folder.glob("*.csv") if p.is_file()])
    if not csv_files:
        print(f"[ERROR] No CSV files found in {folder}", file=sys.stderr)
        sys.exit(1)

    # Connect
    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.db,
    )
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            # Create table if requested
            exists = table_exists(cur, schema, table)
            if not exists and args.autocreate:
                print(f"[INFO] Inferring schema from: {csv_files[0].name}")
                coldefs = infer_schema_from_csv(csv_files[0], args.delimiter, args.encoding)
                print(f"[INFO] Creating table {args.table} with columns:")
                for c in coldefs:
                    print(f"       {c}")
                if not args.dry_run:
                    create_table(cur, full_table, coldefs)
                    conn.commit()
                else:
                    print("[DRY‑RUN] Skipping CREATE TABLE")

            elif not exists and not args.autocreate:
                print(f"[ERROR] Target table {args.table} does not exist. Use --autocreate to create it.", file=sys.stderr)
                sys.exit(1)

            # Optionally truncate
            if args.truncate:
                print(f"[INFO] Truncating {args.table}")
                if not args.dry_run:
                    truncate_table(cur, full_table)
                    conn.commit()
                else:
                    print("[DRY‑RUN] Skipping TRUNCATE")

            # Load files
            total_rows = 0
            for csv_path in csv_files:
                print(f"[INFO] Loading {csv_path.name} ...")
                if not args.dry_run:
                    try:
                        copy_csv(cur, full_table, csv_path, args.delimiter, args.encoding)
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        print(f"[ERROR] Failed to load {csv_path.name}: {e}", file=sys.stderr)
                        sys.exit(1)
                # We don't have an exact row count without parsing; optionally estimate:
                try:
                    # Count lines minus header efficiently
                    with open(csv_path, "r", encoding=args.encoding, newline="") as f:
                        row_count = sum(1 for _ in f) - 1
                        total_rows += max(0, row_count)
                except Exception:
                    pass

            print(f"[DONE] Loaded {len(csv_files)} file(s). Approx rows: {total_rows}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
