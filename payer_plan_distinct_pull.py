# -*- coding: utf-8 -*-
# pip install pandas sqlalchemy psycopg2-binary python-dotenv

import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text as sqltext
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError

# =============================== CONFIG ========================================
DB_HOST = os.getenv("DB_HOST", "iamr007.ddns.net")
DB_PORT = int(os.getenv("DB_PORT", "2345"))
DB_NAME = os.getenv("DB_NAME", "hospital_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "verdansk2020!")
SSL_MODE = os.getenv("DB_SSLMODE", "prefer")  # prefer | require | disable

SCHEMA     = os.getenv("DB_SCHEMA", "public")
TABLE_NAME = os.getenv("DB_TABLE",  "hospital_cpt_charges")

PAYER_COL  = os.getenv("PAYER_COL", "payer_name")
PLAN_COL   = os.getenv("PLAN_COL",  "plan_name")

OUT_DIR = Path(os.getenv("OUT_DIR", Path.cwd() / "distinct_exports"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAYER_CSV = OUT_DIR / "distinct_payers.csv"
PLAN_CSV  = OUT_DIR / "distinct_plans.csv"

print(f"[INFO] Output directory: {OUT_DIR.resolve()}")
print(f"[INFO] Connecting -> host={DB_HOST} port={DB_PORT} db={DB_NAME} user={DB_USER} sslmode={SSL_MODE}")

# =========================== ENGINE (SSL fallback) ==============================
def build_engine_with_ssl_fallback():
    url = URL.create(
        "postgresql+psycopg2",
        username=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        query={"sslmode": SSL_MODE},
    )
    try:
        eng = create_engine(url, pool_pre_ping=True)
        with eng.connect() as c:
            c.exec_driver_sql("SELECT 1")
        return eng
    except OperationalError as e:
        msg = str(e).lower()
        if ("server does not support ssl" in msg or "ssl was required" in msg) and SSL_MODE != "disable":
            url_nossl = URL.create(
                "postgresql+psycopg2",
                username=DB_USER,
                password=DB_PASS,
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                query={"sslmode": "disable"},
            )
            eng = create_engine(url_nossl, pool_pre_ping=True)
            with eng.connect() as c:
                c.exec_driver_sql("SELECT 1")
            print("[WARN] Server has no SSL; fell back to sslmode=disable.")
            return eng
        raise

ENGINE = build_engine_with_ssl_fallback()

# ================================ QUERIES ======================================
def col_exists(schema: str, table: str, col: str) -> bool:
    q = sqltext("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table AND column_name = :col
        LIMIT 1
    """)
    with ENGINE.connect() as conn:
        return conn.execute(q, {"schema": schema, "table": table, "col": col}).fetchone() is not None

for col in (PAYER_COL, PLAN_COL):
    if not col_exists(SCHEMA, TABLE_NAME, col):
        raise SystemExit(f"[ERROR] Column '{col}' does not exist on {SCHEMA}.{TABLE_NAME}")

payer_sql = f"SELECT DISTINCT {PAYER_COL} AS payer_name FROM {SCHEMA}.{TABLE_NAME} WHERE {PAYER_COL} IS NOT NULL"
plan_sql  = f"SELECT DISTINCT {PLAN_COL}  AS plan_name  FROM {SCHEMA}.{TABLE_NAME} WHERE {PLAN_COL}  IS NOT NULL"

print("[INFO] Running distinct queries...")
payer_df = pd.read_sql_query(payer_sql, ENGINE)
plan_df  = pd.read_sql_query(plan_sql,  ENGINE)

# Ensure single column and consistent dtype
payer_df = payer_df[["payer_name"]].astype("string")
plan_df  = plan_df[["plan_name"]].astype("string")

# ================================ OUTPUTS ======================================
payer_df.to_csv(PAYER_CSV, index=False, encoding="utf-8-sig")
plan_df.to_csv(PLAN_CSV,  index=False, encoding="utf-8-sig")

print(f"[OK] Wrote {len(payer_df):,} distinct payers -> {PAYER_CSV}")
print(f"[OK] Wrote {len(plan_df):,} distinct plans  -> {PLAN_CSV}")
