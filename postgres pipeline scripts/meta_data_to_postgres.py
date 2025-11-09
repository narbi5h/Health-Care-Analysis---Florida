from pathlib import Path
from typing import Dict, List, Optional
import re
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.types import Text

# ---- Canonical columns you want to keep ----
CANONICAL_KEYS = [
    "hospital_name",
    "last_updated_on",
    "version",
    "hospital_location",
    "hospital_address",
    "license_number",
]

INPUT_FOLDER =  Path(__file__).parent  # Current directory

def _clean_token(s: str) -> str:
    """
    Normalize a metadata key for fuzzy matching:
    - lowercase
    - strip BOM/whitespace
    - replace separators with single space
    - drop anything after a '|' (e.g., 'license_number|FL' -> 'license_number')
    - remove non-alphanumeric except spaces/underscores
    """
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\ufeff", "")  # strip BOM if present
    # Drop trailing pipe-suffixes like "|FL"
    s = s.split("|", 1)[0]
    s = s.strip().lower()
    s = re.sub(r"[\-\/]+", " ", s)
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"[^a-z0-9_ ]", "", s)
    return s.strip()

def _canonicalize_key(raw: str) -> Optional[str]:
    """
    Map an incoming raw key to one of the six canonical keys using rules/heuristics.
    Returns None if no reasonable match found.
    """
    tok = _clean_token(raw)

    # direct/obvious aliases
    alias_map = {
        # hospital_name
        "hospital name": "hospital_name",
        "facility name": "hospital_name",
        "provider name": "hospital_name",
        "organization name": "hospital_name",
        "org name": "hospital_name",

        # last_updated_on
        "last updated on": "last_updated_on",
        "last updated": "last_updated_on",
        "last update": "last_updated_on",
        "last_updated": "last_updated_on",
        "update date": "last_updated_on",
        "updated on": "last_updated_on",
        "updated": "last_updated_on",

        # version
        "version": "version",
        "file version": "version",
        "schema version": "version",

        # hospital_location
        "hospital location": "hospital_location",
        "location": "hospital_location",
        "city state": "hospital_location",
        "city, state": "hospital_location",

        # hospital_address
        "hospital address": "hospital_address",
        "address": "hospital_address",
        "street address": "hospital_address",

        # license_number
        "license number": "license_number",
        "licensenumber": "license_number",
        "license": "license_number",
        "license no": "license_number",
        "license id": "license_number",
        "license_num": "license_number",
        "state license": "license_number",
    }
    if tok in alias_map:
        return alias_map[tok]

    # Heuristic contains/startswith checks (after cleaning)
    if tok.startswith("hospital name") or tok.endswith("name"):
        return "hospital_name"
    if "update" in tok or "updated" in tok:
        return "last_updated_on"
    if "version" in tok:
        return "version"
    if "address" in tok or tok.startswith("street"):
        return "hospital_address"
    if "location" in tok or tok in {"city state","citystate"}:
        return "hospital_location"
    if tok.startswith("license") or "license" in tok or tok.endswith("lic"):
        return "license_number"

    # Nothing matched
    return None

def _read_two_rows(path: Path) -> pd.DataFrame:
    """
    Tries UTF-8 (with BOM) then falls back to latin-1 to read only the first 2 rows, no header.
    Returns a 2-row DataFrame of strings (NaNs preserved).
    Removes any fully empty rows.
    """
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                path, header=None, nrows=2, dtype=str, engine="python", encoding=enc
            )
            # Drop rows that are completely empty/NaN
            df = df.dropna(how="all").reset_index(drop=True)
            return df
        except Exception:
            continue
    df = pd.read_csv(path, header=None, nrows=2, dtype=str, engine="python")
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def _rows_to_meta(df2: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Interpret the first two rows as {key -> value} across columns:
      - Row 0 = keys, Row 1 = values (most common pattern when metadata is laid out horizontally)
    Any blank/NaN keys are ignored.
    """
    meta: Dict[str, Optional[str]] = {}
    if df2.shape[0] < 2:
        return meta

    keys = df2.iloc[0].tolist()
    vals = df2.iloc[1].tolist()
    for k, v in zip(keys, vals):
        if pd.isna(k):
            continue
        canon = _canonicalize_key(k)
        if canon:
            meta[canon] = None if (pd.isna(v) or v is None) else str(v).strip()
    return meta

def load_first_two_rows_metadata_to_postgres(
    folder: str,
    pg_conn_str: str,
    table_name: str,
    schema: str = "public",
    include_source_file: bool = True,
) -> pd.DataFrame:
    """
    Extracts first 2 rows as metadata from all CSV files in `folder` and appends them to Postgres.

    Parameters
    ----------
    folder : str
        Path to folder containing CSV files.
    pg_conn_str : str
        SQLAlchemy-style Postgres connection string, e.g.
        "postgresql+psycopg2://user:pass@host:port/dbname"
    table_name : str
        Destination table name (created if not exists).
    schema : str
        Destination schema. Default: "public".
    include_source_file : bool
        If True, adds a 'source_file' column so you can trace the origin.

    Returns
    -------
    pd.DataFrame
        The DataFrame that was appended to Postgres (for inspection/logging).
    """
    folder_path = Path(INPUT_FOLDER)
    csv_paths = sorted([p for p in folder_path.glob("*.csv") if p.is_file()])

    records: List[Dict[str, Optional[str]]] = []

    for p in csv_paths:
        try:
            df2 = _read_two_rows(p)

            # Extra safeguard: drop empty rows again
            df2 = df2.dropna(how="all").reset_index(drop=True)

            meta = _rows_to_meta(df2)

            # Keep only canonical keys; ensure all exist
            row = {k: meta.get(k) for k in CANONICAL_KEYS}
            if include_source_file:
                row["source_file"] = p.name
            records.append(row)
        except Exception as e:
            print(f"[WARN] Skipped {p.name}: {e}")


    if not records:
        print("[INFO] No metadata extracted.")
        return pd.DataFrame(columns=CANONICAL_KEYS + (["source_file"] if include_source_file else []))

    df_out = pd.DataFrame.from_records(records)

    # --- Load to Postgres ---
    engine: Engine = create_engine(pg_conn_str)

    # Create table (if needed) with TEXT columns (simple & safe for messy metadata)
    dtype_map = {col: Text for col in CANONICAL_KEYS}
    if include_source_file:
        dtype_map["source_file"] = Text

    # Ensure schema exists (no-op if it already does)
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))

    df_out.to_sql(
        table_name,
        engine,
        schema=schema,
        if_exists="append",   # append new rows; change to 'replace' if you prefer
        index=False,
        dtype=dtype_map,
        method="multi",
        chunksize=1000,
    )

    return df_out

# # ---------------------------
# # Example usage:
# # ---------------------------
# df_logged = load_first_two_rows_metadata_to_postgres(
#     folder=(INPUT_FOLDER),
#     pg_conn_str="postgresql+psycopg2://postgres:BANA650@localhost:5432/postgres",  #LOCAL POSTGRES 
#     # pg_conn_str="postgresql+psycopg2://postgres:verdansk2020!@iamr007.ddns.net:2345/postgres", #REMOTE POSTGRES
#     table_name="hospital_metadata_test",
#     schema="public",
# )
# print(df_logged.head())
