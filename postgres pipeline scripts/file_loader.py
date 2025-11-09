#!/usr/bin/env python3
"""
hospital_csv_to_final_streaming_text.py

Stream-process VERY LARGE hospital "standard charges" CSVs safely and
upload all columns as TEXT to Postgres. Also supports writing a streaming CSV.

Features:
- Skips first 2 metadata rows
- Unpivots multiple code/code_type families: supports 'code: N', 'code|N', 'code|N|type'
- Unpivots payer-scoped negotiated/methodology/estimated columns
- Parses payer_name / plan_name from column suffixes after ':'
- Cleans numeric-like fields by removing commas and '%' but leaves all columns as TEXT
- Adds 'source_file' column with the CSV's basename
- Streams in chunks to bound memory usage

Usage examples:
  python hospital_csv_to_final_streaming_text.py --input path/to/file.csv --output out.csv
  python hospital_csv_to_final_streaming_text.py --input path/to/file.csv \
      --to-postgres "postgresql+psycopg2://user:pw@host:5432/db" --table final_table
  python hospital_csv_to_final_streaming_text.py --input huge.csv --to-postgres ... --table ... --chunksize 50000
"""

from __future__ import annotations
import os
import re
import sys
import argparse
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd
from sqlalchemy.types import Text
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

EXPECTED_KEYS = {
    "description", "setting", "modifiers",
    "drug_unit_of_measurement", "drug_type_of_measurement",
    "standard_charge_gross", "standard_charge_discounted_cash",
    "standard_charge_min", "standard_charge_max",
    # measures (payer-scoped)
    "standard_charge_negotiated_dollar",
    "standard_charge_negotiated_percentage",
    "standard_charge_negotiated_algorithm",
    "estimated_amount",
    "standard_charge_methodology",
    # code family roots
    "code", "code_type"
}

def _score_header(cols: list[str]) -> int:
    """Simple score: how many expected tokens or patterns can we see in the header?"""
    if not cols:
        return 0
    cols_norm = [normalize_header(c).lower() for c in cols]
    score = 0
    for c in cols_norm:
        base = c.split(" : ", 1)[0]  # strip payer suffix if present
        if base in EXPECTED_KEYS:
            score += 2
        # code families like 'code : 1', 'code|1', 'code_type : 2', 'code|2|type'
        if re.fullmatch(r"(code|code_type)(\s*:\s*|\|\s*)\d+(\|type)?", c):
            score += 2
    return score

def detect_skiprows(path: str, enc: str) -> int:
    """
    Decide whether to skip 0 or 2 rows.
    Try both headers and pick the one with the higher score.
    """
    try:
        h2 = pd.read_csv(path, skiprows=2, nrows=0, dtype=str, encoding=enc, engine="python", low_memory=True)
        cols2 = list(h2.columns)
        s2 = _score_header(cols2)
    except Exception:
        s2, cols2 = -1, []

    try:
        h0 = pd.read_csv(path, skiprows=0, nrows=0, dtype=str, encoding=enc, engine="python", low_memory=True)
        cols0 = list(h0.columns)
        s0 = _score_header(cols0)
    except Exception:
        s0, cols0 = -1, []

    chosen = 2 if s2 >= s0 else 0
    print(f"🔎 Header detection for {os.path.basename(path)} → skiprows={chosen} (score2={s2}, score0={s0})")
    # Optional: uncomment to inspect
    # print("  hdr(skip=2):", cols2[:5])
    # print("  hdr(skip=0):", cols0[:5])
    return chosen


# ---------------------------
# Final output columns (+ source_file appended later)
# ---------------------------
FINAL_COLUMNS = [
    "description",
    "code",
    "code_type",
    "modifiers",
    "setting",
    "drug_unit_of_measurement",
    "drug_type_of_measurement",
    "standard_charge_gross",
    "standard_charge_discounted_cash",
    "payer_name",
    "plan_name",
    "standard_charge_negotiated_dollar",
    "standard_charge_negotiated_percentage",
    "standard_charge_negotiated_algorithm",
    "estimated_amount",
    "standard_charge_methodology",
    "standard_charge_min",
    "standard_charge_max",
]
FINAL_WITH_SOURCE = FINAL_COLUMNS + ["source_file"]

# Measures that can appear in payer-scoped columns
PAYER_MEASURES = {
    "standard_charge_negotiated_dollar",
    "standard_charge_negotiated_percentage",
    "standard_charge_negotiated_algorithm",
    "estimated_amount",
    "standard_charge_methodology",
}

# Columns considered numeric-like for simple cleaning (we still keep TEXT)
NUMERIC_LIKE = {
    "standard_charge_gross",
    "standard_charge_discounted_cash",
    "standard_charge_negotiated_dollar",
    "standard_charge_negotiated_percentage",
    "estimated_amount",
    "standard_charge_min",
    "standard_charge_max",
}

# ---------------------------
# Encoding probe (header only)
# ---------------------------


def _to_text_cell(x):
    # Normalize any value to a TEXT-safe value (or None)
    if x is None:
        return None
    try:
        # real NaN?
        import numpy as np
        if isinstance(x, float) and np.isnan(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in ("", "nan", "none"):
        return None
    return s


def probe_encoding(path: str) -> str:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            # Verify we can read the header after skipping metadata
            pd.read_csv(
                path, skiprows=2, nrows=0, dtype=str, encoding=enc, engine="python", low_memory=True
            )
            return enc
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read header with common encodings. Last error: {last_err}")

# ---------------------------
# Header normalization (lightweight)
# ---------------------------
def normalize_header(h: str) -> str:
    if h is None:
        return ""
    h = str(h).replace("\ufeff", "").strip()
    # Normalize spaces around colon and compress whitespace
    h = re.sub(r"\s*:\s*", " : ", h)
    h = re.sub(r"\s+", " ", h)
    return h

def maybe_normalize_columns(cols: List[str]) -> List[str]:
    return [normalize_header(c) for c in cols]

# ---------------------------
# Code-family detection & unpivot
# Supports: code, code_type, code: N, code_type: N, code|N, code|N|type
# ---------------------------
CODE_COL_RE = re.compile(r"^(?:code(?:\s*:\s*|\|))(\d+)$", re.IGNORECASE)
CODETYPE_COL_RE = re.compile(r"^(?:code(?:\s*:\s*|\|))(\d+)(?:\s*:\s*type|\|type)$", re.IGNORECASE)

def find_code_families(columns: List[str]) -> Tuple[Optional[str], Optional[str], Dict[str, Tuple[Optional[str], Optional[str]]]]:
    cols_set = set(columns)
    single_code = "code" if "code" in cols_set else None
    single_code_type = "code_type" if "code_type" in cols_set else None

    multi: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    for c in columns:
        cl = c.lower()
        m = CODE_COL_RE.match(cl)
        if m:
            n = m.group(1)
            multi.setdefault(n, [None, None])[0] = c
        m2 = CODETYPE_COL_RE.match(cl)
        if m2:
            n = m2.group(1)
            multi.setdefault(n, [None, None])[1] = c
        # literal "code: N" / "code_type: N"
        m3 = re.fullmatch(r"code\s*:\s*(\d+)", c, flags=re.IGNORECASE)
        if m3:
            n = m3.group(1)
            multi.setdefault(n, [None, None])[0] = c
        m4 = re.fullmatch(r"code_type\s*:\s*(\d+)", c, flags=re.IGNORECASE)
        if m4:
            n = m4.group(1)
            multi.setdefault(n, [None, None])[1] = c
    multi = {k: (v[0], v[1]) for k, v in multi.items()}
    return single_code, single_code_type, multi

def unpivot_code_columns(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    df = df.copy()
    single_code, single_code_type, multi = find_code_families(column_names)

    # Case 1: plain code/code_type only
    if (single_code or single_code_type) and not multi:
        if single_code and single_code != "code":
            df.rename(columns={single_code: "code"}, inplace=True)
        if single_code_type and single_code_type != "code_type":
            df.rename(columns={single_code_type: "code_type"}, inplace=True)
        for c in ("code", "code_type"):
            if c not in df.columns:
                df[c] = np.nan
        return df

    # Case 2: exactly one indexed family
    if len(multi) == 1 and not (single_code or single_code_type):
        n = next(iter(multi))
        code_col, codetype_col = multi[n]
        if code_col and code_col != "code":
            df.rename(columns={code_col: "code"}, inplace=True)
        else:
            if "code" not in df.columns:
                df["code"] = np.nan
        if codetype_col and codetype_col != "code_type":
            df.rename(columns={codetype_col: "code_type"}, inplace=True)
        else:
            if "code_type" not in df.columns:
                df["code_type"] = np.nan
        return df

    # Case 3: multiple families → build rows per family
    family_cols: Set[str] = set()
    for code_col, codetype_col in multi.values():
        if code_col: family_cols.add(code_col)
        if codetype_col: family_cols.add(codetype_col)
    shared_cols = [c for c in df.columns if c not in family_cols]

    out_frames = []
    for n, (code_col, codetype_col) in multi.items():
        temp = df[shared_cols].copy()
        temp["code"] = df[code_col] if code_col in df.columns else np.nan
        temp["code_type"] = df[codetype_col] if codetype_col in df.columns else np.nan
        out_frames.append(temp)

    if out_frames:
        df = pd.concat(out_frames, ignore_index=True)
    if "code" not in df.columns: df["code"] = np.nan
    if "code_type" not in df.columns: df["code_type"] = np.nan
    return df

# ---------------------------
# Payer-scoped columns → melt
# ---------------------------
PAYER_COL_RE = re.compile(
    r"^(standard_charge_negotiated_dollar|standard_charge_negotiated_percentage|standard_charge_negotiated_algorithm|estimated_amount|standard_charge_methodology)\s*:\s*(.+)$",
    re.IGNORECASE,
)

def split_payer_plan(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristic split of '<payer/plan text>' -> (payer_name, plan_name).
    Keeps parentheses with the plan segment naturally via prefix matching.
    """
    if text is None:
        return (None, None)
    s = str(text).strip()
    if not s:
        return (None, None)

    prefixes = [
        "UnitedHealthcare", "United HealthCare", "United Healthcare", "UHC",
        "Blue Cross Blue Shield", "Florida Blue", "BCBS",
        "Aetna Better Health", "Aetna",
        "Cigna Healthcare", "Cigna",
        "Humana",
        "Kaiser Permanente", "Kaiser",
        "Careplus Health Plan", "CarePlus Health Plan", "CarePlus",
        "Ambetter", "Molina Healthcare", "Molina",
        "Oscar Health", "Oscar", "Medica",
        "Tufts Health Plan", "Tufts",
        "Centene", "Anthem", "Tricare",
        "Medicare", "Medicaid",
        "Health Net",
    ]

    s_norm = s
    lower = s.lower()

    for p in sorted(prefixes, key=len, reverse=True):
        if lower.startswith(p.lower()):
            payer = s_norm[:len(p)].strip()
            rest = s_norm[len(p):].strip()
            return (payer if payer else None, rest if rest else None)

    parts = s_norm.split(" ", 1)
    if len(parts) == 1:
        return (parts[0].strip(), None)
    return (parts[0].strip(), parts[1].strip())

def unpivot_payer_columns(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    df = df.copy()
    payer_cols: Dict[str, Dict[str, str]] = {}

    for c in column_names:
        m = PAYER_COL_RE.match(c)
        if m:
            measure = m.group(1).lower()
            payer_raw = m.group(2).strip()
            d = payer_cols.setdefault(payer_raw, {})
            d[measure] = c

    if not payer_cols:
        # Ensure fields exist when payer columns absent
        if "payer_name" not in df.columns: df["payer_name"] = np.nan
        if "plan_name" not in df.columns: df["plan_name"] = np.nan
        for m in PAYER_MEASURES:
            if m not in df.columns:
                df[m] = np.nan
        return df

    # Remove payer-scoped measure columns from the shared set
    drop_set = set()
    for mp in payer_cols.values():
        drop_set.update(mp.values())
    shared_cols = [c for c in df.columns if c not in drop_set]

    out_frames = []
    for payer_raw, measures_map in payer_cols.items():
        payer_name, plan_name = split_payer_plan(payer_raw)
        temp = df[shared_cols].copy()
        temp["payer_name"] = payer_name
        temp["plan_name"] = plan_name
        for m in PAYER_MEASURES:
            temp[m] = df[measures_map[m]] if m in measures_map else np.nan
        out_frames.append(temp)

    return pd.concat(out_frames, ignore_index=True) if out_frames else df

# ---------------------------
# Cleaning (keep TEXT)
# ---------------------------
def clean_numeric_like_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _clean_cell(x):
        if x is None:
            return None
        if isinstance(x, float):
            import numpy as np
            if np.isnan(x):
                return None
        s = str(x).strip()
        if s.lower() in ("", "nan", "none"):
            return None
        return s.replace(",", "").replace("%", "")

    for col in NUMERIC_LIKE:
        if col in df.columns:
            df[col] = df[col].map(_clean_cell)
    return df




def ensure_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df

# ---------------------------
# Chunk transformer
# ---------------------------
def transform_chunk(df_chunk: pd.DataFrame, normalized_cols: List[str]) -> pd.DataFrame:
    # Enforce normalized headers
    df_chunk = df_chunk.copy()
    df_chunk.columns = normalized_cols

    # ⚠️ FORCE EVERYTHING TO TEXT EARLY (no .str accessor used)
    for c in df_chunk.columns:
        df_chunk[c] = df_chunk[c].map(_to_text_cell)

    # A) code families
    df1 = unpivot_code_columns(df_chunk, normalized_cols)

    # B) payer-scoped columns
    colnames_after_code = list(df1.columns)
    df2 = unpivot_payer_columns(df1, colnames_after_code)

    # C) ensure final columns
    df2 = ensure_final_columns(df2)

    # D) clean numeric-like strings WITHOUT .str
    df2 = clean_numeric_like_strings(df2)
    # E) order to final schema (source_file will be added later)
    df2 = df2[FINAL_COLUMNS]

    # F) last defense: map again to TEXT-safe values for upload/CSV
    for c in df2.columns:
        df2[c] = df2[c].map(_to_text_cell)

    # 🚨 STEP 1: Drop rows where BOTH code and code_type are empty
    if "code" in df2.columns and "code_type" in df2.columns:
        before = len(df2)

        def _nonempty(s):
            return s.notna() & (s != "")

        keep_mask_codes = (
            _nonempty(df2["code"])
            | _nonempty(df2["code_type"])
        )

        df2 = df2[keep_mask_codes]
        dropped = before - len(df2)
        if dropped:
            print(f"⚠️  Dropped {dropped:,} rows where BOTH code and code_type were empty")

    # ORIGINAL STEP 2 OUTDATED BELOW KEPT FOR REFERENCE
    # # 🚨 STEP 2 (modified): KEEP only rows where BOTH negotiated dollar AND percentage are empty
    # if "standard_charge_negotiated_dollar" in df2.columns and "standard_charge_negotiated_percentage" in df2.columns:
    #     before = len(df2)
    #     empty_both = (
    #         (~_nonempty(df2["standard_charge_negotiated_dollar"])) &
    #         (~_nonempty(df2["standard_charge_negotiated_percentage"]))
    #     )
    #     df2 = df2[empty_both]
    #     dropped = before - len(df2)
    #     if dropped:
    #         print(f"⚠️  Dropped {dropped:,} rows where at least one negotiated dollar/percentage value was present; kept {len(df2):,} rows with BOTH empty")
    ### END OF OUTDATED STEP 2 ###

    # 🚨 STEP 2 (updated): Keep a row iff ANY 'standard_charge*' column is populated; drop only if ALL are empty
    sc_cols = [c for c in df2.columns if "standard_charge" in c.lower()]

    if sc_cols:
        before = len(df2)

        # treat NaN / whitespace as empty; "0" counts as populated
        nonempty = lambda s: s.notna() & (s.astype(str).str.strip() != "")

        has_any_sc = pd.DataFrame({c: nonempty(df2[c]) for c in sc_cols}).any(axis=1)

        include_mask = has_any_sc & (df2["code_type"] == "CPT")
        df2 = df2[include_mask].copy()
        dropped = before - include_mask.sum()
        if dropped:
            print(f"⚠️  Dropped {dropped:,} rows where ALL standard_charge* columns were empty "
              f"and code != 'CPT' ({len(sc_cols)} columns checked).")
        dropped = before - has_any_sc.sum()
        # if dropped:
        #     print(f"⚠️  Dropped {dropped:,} rows where ALL standard_charge* columns were empty "
        #         f"({len(sc_cols)} columns checked).")
    else:
        print("ℹ️  No columns containing 'standard_charge' found; skipping this filter.")        
    
    return df2


# ---------------------------
# Upload helpers
# ---------------------------
def upload_to_postgres(df: pd.DataFrame, conn_str: str, table: str, chunksize: int = 10_000):
    from sqlalchemy import create_engine
    engine = create_engine(conn_str)
    dtype_map = {col: Text for col in df.columns}
    df.to_sql(
        table,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=chunksize,
        dtype=dtype_map,
    )

# ---------------------------
# Main streaming pipeline
# ---------------------------
def process_file_streaming(
    path: str,
    to_postgres: Optional[str],
    table: Optional[str],
    # out_csv: Optional[str],
    chunksize: int,
    normalize_headers: bool,
):
    enc = probe_encoding(path)
    skip = detect_skiprows(path, enc)

    # Read just the columns with chosen skiprows
    header_df = pd.read_csv(
        path, skiprows=skip, nrows=0, dtype=str, encoding=enc, engine="python", low_memory=True
    )
    raw_cols = list(header_df.columns)
    normalized_cols = maybe_normalize_columns(raw_cols) if normalize_headers else raw_cols

    # Iterate chunks using the same skiprows
    reader = pd.read_csv(
        path,
        skiprows=skip,
        dtype=str,
        encoding=enc,
        engine="python",
        chunksize=chunksize,
        low_memory=True,
    )


    basename = os.path.basename(path)
    total_rows_out = 0
    for chunk in reader:
        df_final = transform_chunk(chunk, normalized_cols)

        # Add source_file
        df_final["source_file"] = basename

        # Output destinations
        if to_postgres and table:
            upload_to_postgres(df_final, to_postgres, table)
        # if out_handle:
        #     df_final.to_csv(out_handle, header=False, index=False)

        total_rows_out += len(df_final)

    # if out_handle:
    #     out_handle.close()

    msg = f"✅ Processed {basename} → {total_rows_out:,} output rows"
    if to_postgres and table:
        msg += f" (appended to {table})"
    print(msg)

    # After successful upload, move file to uploaded_to_postgres folder
    if to_postgres and table:
        dest_dir = os.path.join(os.path.dirname(path), "uploaded_to_postgres_multi_thread")
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, basename)
        try:
            os.rename(path, dest_path)
            print(f"📦 Moved {basename} → {dest_dir}")
        except Exception as e:
            print(f"⚠️ Could not move {basename}: {e}")


# --- helper: expand folder to file list ---
def _gather_input_files(input_path: str, pattern: str) -> list[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(str(x) for x in p.glob(pattern) if x.is_file())
        if not files:
            print(f"⚠️ No files matched pattern '{pattern}' in {p}")
        return files
    raise FileNotFoundError(f"Input path not found: {input_path}")


# --- worker wrapper so a failure in one file doesn't crash the whole pool ---
def _process_one_file(args_tuple) -> tuple[str, bool, str]:
    (path, to_postgres, table, chunksize, normalize_headers) = args_tuple
    try:
        process_file_streaming(
            path=path,
            to_postgres=to_postgres,
            table=table,
            chunksize=chunksize,
            normalize_headers=normalize_headers,
        )
        return (os.path.basename(path), True, "ok")
    except Exception as e:
        return (os.path.basename(path), False, str(e))
    

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV or a folder")
    ap.add_argument("--pattern", default="*.csv", help="Glob when --input is a folder (default: *.csv)")
    ap.add_argument("--output", help="(Unused in parallel mode) Optional path to write transformed CSV")
    ap.add_argument("--to-postgres", dest="pg", help="SQLAlchemy URL, e.g., postgresql+psycopg2://user:pw@host:5432/db")
    ap.add_argument("--table", help="Target Postgres table (required if --to-postgres set)")
    ap.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk (default 100000)")
    ap.add_argument("--no-header-normalize", action="store_true", help="Disable header normalization")
    ap.add_argument("--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="Number of worker processes (default: CPU cores - 1)")
    args = ap.parse_args()

    if args.pg and not args.table:
        print("❌ --table is required when using --to-postgres", file=sys.stderr)
        sys.exit(2)

    files = _gather_input_files(args.input, args.pattern)

    # Single file: run in-process (keeps behavior identical)
    if len(files) == 1:
        process_file_streaming(
            path=files[0],
            to_postgres=args.pg,
            table=args.table,
            chunksize=args.chunksize,
            normalize_headers=not args.no_header_normalize,
        )
        return

    # Multiple files: run in parallel (one process per file)
    print(f"🚀 Parallel mode: {len(files)} files | workers={args.jobs}")
    tasks = [
        (f, args.pg, args.table, args.chunksize, not args.no_header_normalize)
        for f in files
    ]

    # IMPORTANT for Windows: protect entry point
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(_process_one_file, t) for t in tasks]
        ok = 0
        for fut in as_completed(futs):
            fname, success, msg = fut.result()
            if success:
                ok += 1
                print(f"✅ {fname} | {msg}")
            else:
                print(f"❌ {fname} | {msg}")

    print(f"Done. {ok}/{len(files)} succeeded."
    )

if __name__ == "__main__":
    main()
