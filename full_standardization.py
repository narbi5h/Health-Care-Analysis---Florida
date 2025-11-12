# -*- coding: utf-8 -*-
# pip install pandas sqlalchemy psycopg2-binary python-dotenv
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text as sqltext, MetaData, Table, Column, Text
from sqlalchemy.engine import URL
from sqlalchemy.exc import OperationalError

# =============================== CONFIG ========================================
DB_HOST = os.getenv("DB_HOST", "iamr007.ddns.net")
DB_PORT = int(os.getenv("DB_PORT", "2345"))
DB_NAME = os.getenv("DB_NAME", "hospital_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASSWORD", "verdansk2020!")
SSL_MODE = os.getenv("DB_SSLMODE", "prefer")

SCHEMA     = os.getenv("DB_SCHEMA", "public")
TABLE_NAME = os.getenv("DB_TABLE",  "hospital_cpt_charges")
TABLE      = f"{SCHEMA}.{TABLE_NAME}"

CODE_COL   = os.getenv("CODE_COL",  "code")
CTYPE_COL  = os.getenv("CTYPE_COL", "code_type")
DESCR_COL  = os.getenv("DESCR_COL", "description")

PAYER_COL  = os.getenv("PAYER_COL", "payer_name")
PLAN_COL   = os.getenv("PLAN_COL",  "plan_name")

BUCKET_COL = os.getenv("BUCKET_COL", "bucket")
SPEC_COL   = os.getenv("SPEC_COL",   "specification")

PAYER_CSV = os.getenv("PAYER_CSV", str(Path.cwd() / "distinct_payers.csv"))
PLAN_CSV  = os.getenv("PLAN_CSV",  str(Path.cwd() / "distinct_plans.csv"))

DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"
# ADD_BUCKET_SPEC_COLUMNS = os.getenv("ADD_BUCKET_SPEC_COLUMNS", "true").lower() == "true"

print(f"[INFO] Connecting -> host={DB_HOST} port={DB_PORT} db={DB_NAME} user={DB_USER} sslmode={SSL_MODE}")
print(f"[INFO] Using CSV files:\n       payers={PAYER_CSV}\n       plans={PLAN_CSV}")
# print(f"[INFO] Modes: DRY_RUN={DRY_RUN} ADD_BUCKET_SPEC_COLUMNS={ADD_BUCKET_SPEC_COLUMNS}")

# =========================== ENGINE ============================================
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

# =============================== DB HELPERS ====================================
def ensure_specs_tables(engine, schema: str):
    md = MetaData(schema=schema)
    Table("plan_specs", md,
          Column("name", Text, primary_key=True),
          Column("bucket", Text),
          Column("specification", Text))
    Table("payer_specs", md,
          Column("name", Text, primary_key=True),
          Column("bucket", Text),
          Column("specification", Text))
    md.create_all(bind=engine, checkfirst=True)

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm = {str(c).lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidates:
        k = cand.lower().replace(" ", "").replace("_", "")
        if k in norm:
            return norm[k]
    return None

def read_specs_csv(path: str, name_candidates: list[str]) -> list[tuple[str, str | None, str | None]]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] CSV not found: {path} (skipping)")
        return []
    df = pd.read_csv(p, dtype=str).fillna("")
    name_col = _pick_col(df, name_candidates)
    bucket_col = _pick_col(df, ["bucket"])
    spec_col   = _pick_col(df, ["specification", "spec", "canonical"])
    if not name_col or not spec_col:
        print(f"[WARN] Missing required columns in {p.name}. Found: {list(df.columns)}")
        return []
    rows = []
    for _, r in df.iterrows():
        n = str(r[name_col]).strip()
        b = str(r[bucket_col]).strip() if bucket_col else ""
        s = str(r[spec_col]).strip()
        if n and (b or s):
            rows.append((n, (b or None), (s or None)))
    print(f"[INFO] Loaded {len(rows):,} rows from {p.name}")
    return rows

def upsert_specs(engine, schema: str, table: str, rows):
    if not rows:
        return
    sql = sqltext(f"""
        INSERT INTO {schema}.{table}(name, bucket, specification)
        VALUES (:name, :bucket, :spec)
        ON CONFLICT (name) DO UPDATE
        SET bucket = EXCLUDED.bucket, specification = EXCLUDED.specification
    """)
    with engine.begin() as conn:
        conn.execute(sql, [{"name":n, "bucket":b, "spec":s} for n,b,s in rows])

def ensure_bucket_spec_columns(engine, schema: str, table_fq: str, bucket_col: str, spec_col: str):
    with engine.begin() as conn:
        conn.execute(sqltext(f'ALTER TABLE {table_fq} ADD COLUMN IF NOT EXISTS "{bucket_col}" TEXT'))
        conn.execute(sqltext(f'ALTER TABLE {table_fq} ADD COLUMN IF NOT EXISTS "{spec_col}"   TEXT'))

def backfill_bucket_spec_from_lookups(engine, schema: str, table_fq: str,
                                      payer_col: str, plan_col: str,
                                      bucket_col: str, spec_col: str):
    sql = f"""
    UPDATE {table_fq} h
    SET
      "{bucket_col}" = COALESCE(
          h."{bucket_col}",
          (SELECT p.bucket FROM {schema}.plan_specs p WHERE p.name = h.{plan_col}),
          (SELECT y.bucket FROM {schema}.payer_specs y WHERE y.name = h.{payer_col})
      ),
      "{spec_col}" = COALESCE(
          h."{spec_col}",
          (SELECT p.specification FROM {schema}.plan_specs p WHERE p.name = h.{plan_col}),
          (SELECT y.specification FROM {schema}.payer_specs y WHERE y.name = h.{payer_col})
      )
    WHERE (h."{bucket_col}" IS NULL OR h."{spec_col}" IS NULL);
    """
    with engine.begin() as conn:
        conn.execute(sqltext(sql))

# ========== ZERO-WRITE PREVIEW (BUCKET & SPEC COUNTS USING CSV) ================
def _values_cte(alias: str, rows):
    params, tuples = {}, []
    for i, (n, b, s) in enumerate(rows, start=1):
        tuples.append(f"(:{alias}_n{i}, :{alias}_b{i}, :{alias}_s{i})")
        params[f"{alias}_n{i}"] = n
        params[f"{alias}_b{i}"] = b
        params[f"{alias}_s{i}"] = s
    if not tuples:
        return f"{alias}(name, bucket, specification) AS (SELECT NULL::text, NULL::text, NULL::text WHERE false)", {}
    return f"{alias}(name, bucket, specification) AS (VALUES {', '.join(tuples)})", params

def preview_counts_from_csv(engine, plan_rows, payer_rows):
    cte_plan, p_params = _values_cte("plan_map", plan_rows)
    cte_payer, y_params = _values_cte("payer_map", payer_rows)
    params = {**p_params, **y_params}

    bucket_sql = f"""
    WITH {cte_plan}, {cte_payer}
    SELECT COALESCE(pm.bucket, ym.bucket) AS bucket, COUNT(*) AS count
    FROM {TABLE} h
    LEFT JOIN plan_map pm  ON pm.name = h.{PLAN_COL}
    LEFT JOIN payer_map ym ON ym.name = h.{PAYER_COL}
    WHERE pm.name IS NOT NULL OR ym.name IS NOT NULL
    GROUP BY 1 ORDER BY 2 DESC, 1;
    """

    spec_sql = f"""
    WITH {cte_plan}, {cte_payer}
    SELECT COALESCE(pm.specification, ym.specification) AS specification, COUNT(*) AS count
    FROM {TABLE} h
    LEFT JOIN plan_map pm  ON pm.name = h.{PLAN_COL}
    LEFT JOIN payer_map ym ON ym.name = h.{PAYER_COL}
    WHERE pm.name IS NOT NULL OR ym.name IS NOT NULL
    GROUP BY 1 ORDER BY 2 DESC, 1;
    """

    unmapped_sql = f"""
    WITH {cte_plan}, {cte_payer}
    SELECT COUNT(*) AS unmapped
    FROM {TABLE} h
    LEFT JOIN plan_map pm  ON pm.name = h.{PLAN_COL}
    LEFT JOIN payer_map ym ON ym.name = h.{PAYER_COL}
    WHERE pm.name IS NULL AND ym.name IS NULL;
    """

    with engine.connect() as conn:
        buckets = conn.execute(sqltext(bucket_sql), params).fetchall()
        specs   = conn.execute(sqltext(spec_sql), params).fetchall()
        unmapped = conn.execute(sqltext(unmapped_sql), params).scalar() or 0

    print("\n=== Preview: Bucket counts that WOULD be written (plan-first, then payer) ===")
    for b, c in buckets or []:
        print(f"{(b or 'NULL'):30s} {c:>10,d}")
    if not buckets:
        print("[INFO] No matching buckets.")

    print("\n=== Preview: Specification counts that WOULD be written (plan-first, then payer) ===")
    for s, c in specs or []:
        print(f"{(s or 'NULL'):50s} {c:>10,d}")
    if not specs:
        print("[INFO] No matching specifications.")

    print(f"\n[INFO] Unmapped rows (no plan/payer match in CSVs): {unmapped:,}")

# ================================ MAIN =========================================
def main():
    if DRY_RUN:
        print("[DRY RUN] Skipping table creation.")
    else:
        ensure_specs_tables(ENGINE, SCHEMA)

    payer_rows = read_specs_csv(PAYER_CSV, ["payer_name", "name"])
    plan_rows  = read_specs_csv(PLAN_CSV,  ["plan_name", "name"])

    print(f"[INFO] CSV preview - plans: {len(plan_rows)} rows, payers: {len(payer_rows)} rows")

    # Always safe: preview counts (no writes)
    preview_counts_from_csv(ENGINE, plan_rows, payer_rows)

    if DRY_RUN:
        print("\n[DRY RUN] Would upsert into lookup tables (showing top 3 rows):")
        print("  plans:", plan_rows[:3])
        print("  payers:", payer_rows[:3])
    else:
        if plan_rows:
            upsert_specs(ENGINE, SCHEMA, "plan_specs", plan_rows)
            print(f"[OK] Upserted {len(plan_rows):,} plan specs")
        if payer_rows:
            upsert_specs(ENGINE, SCHEMA, "payer_specs", payer_rows)
            print(f"[OK] Upserted {len(payer_rows):,} payer specs")

    # if ADD_BUCKET_SPEC_COLUMNS:
    #     if DRY_RUN:
    #         print(f"[DRY RUN] Would ensure and backfill {BUCKET_COL}/{SPEC_COL} on {TABLE}")
    #     else:
    #         print(f"[INFO] Ensuring '{BUCKET_COL}', '{SPEC_COL}' and backfilling from CSVs...")
    #         ensure_bucket_spec_columns(ENGINE, SCHEMA, TABLE, BUCKET_COL, SPEC_COL)
    #         backfill_bucket_spec_from_lookups(
    #             ENGINE, SCHEMA, TABLE,
    #             payer_col=PAYER_COL, plan_col=PLAN_COL,
    #             bucket_col=BUCKET_COL, spec_col=SPEC_COL
    #         )
    #         print("[OK] Backfill complete.")
    # else:
    #     print("[INFO] ADD_BUCKET_SPEC_COLUMNS is false — skipping backfill.")

    print("[DONE] CSV-driven bucketing/specification is ready.")

if __name__ == "__main__":
    main()
