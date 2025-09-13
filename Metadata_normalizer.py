print("Running...")
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import csv
import re
import os
import datetime as dt

TARGET_COLUMNS = [
    "hospital_name",
    "last_updated_on",
    "version",
    "hospital_location",
    "hospital_address",
    "license_number",
    "description",
    "code",
    "code_type",
    "setting",
    "drug_unit_of_measurement",
    "drug_type_of_measurement",
    "standard_charge_gross",
    "standard_charge_discounted_cash",
    "payer_name",
    "plan_name",
    "modifiers",
    "standard_charge_negotiated_dollar",
    "standard_charge_negotiated_percentage",
    "standard_charge_negotiated_algorithm",
    "estimated_amount",
    "standard_charge_min",
    "standard_charge_max",
    "standard_charge_methodology",
    "additional_generic_notes",
]

def _norm_key(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _colstrip(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def to_float(x) -> float:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if not isinstance(x, str) or x.strip() == "":
        return np.nan
    xx = x.strip().replace("$", "").replace(",", "")
    try:
        return float(xx)
    except Exception:
        return np.nan

def read_csv_try_encodings(path: Path, header: int) -> Tuple[pd.DataFrame, str]:
    """
    Try fast/default engine first; final fallback uses utf-8/ignore, and last resort
    the python engine (without low_memory) with on_bad_lines='skip'.
    """
    encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encs:
        try:
            df = pd.read_csv(path, header=header, dtype=str, low_memory=False, encoding=enc)
            return df, enc
        except Exception:
            continue
    # Tolerant fallback
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        try:
            df = pd.read_csv(f, header=header, dtype=str, low_memory=False)
            return df, "utf-8/ignore"
        except Exception:
            f.seek(0)
            df = pd.read_csv(f, header=header, dtype=str, engine="python", on_bad_lines="skip")
            return df, "python:utf-8/ignore"

def read_meta_rows_if_present(path: Path) -> Tuple[Dict[str, str], bool]:
    """Look for first two metadata rows with encoding fallback."""
    encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encs:
        try:
            with open(path, "r", encoding=enc, errors="strict", newline="") as f:
                r = csv.reader(f)
                row1 = next(r, None)
                row2 = next(r, None)
            if row1 and row2 and row1 != row2:
                keys = [str(c).strip() for c in row1]
                vals = [str(c).strip() for c in row2]
                if len(vals) < len(keys):
                    vals += [""] * (len(keys) - len(vals))
                return dict(zip(keys, vals)), True
            break
        except Exception:
            continue
    # fallback ignore
    try:
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.reader(f)
            row1 = next(r, None)
            row2 = next(r, None)
        if row1 and row2 and row1 != row2:
            keys = [str(c).strip() for c in row1]
            vals = [str(c).strip() for c in row2]
            if len(vals) < len(keys):
                vals += [""] * (len(keys) - len(vals))
            return dict(zip(keys, vals)), True
    except Exception:
        pass
    return {}, False

def read_table_allow_meta(path: Path) -> Tuple[pd.DataFrame, Dict[str, str], str]:
    """Read as long; if first two rows are metadata, header=2; else header=0. Handles encodings and strips cols/values."""
    meta, has_meta = read_meta_rows_if_present(path)
    header_idx = 2 if has_meta else 0
    df, enc = read_csv_try_encodings(path, header=header_idx)
    df = _colstrip(df)
    return df, meta, enc

def parse_path_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """If 'standard_charge_path' exists but parsed fields do not, derive charge_scope/payer/plan/charge_type."""
    need_parse = "standard_charge_path" in df.columns and "charge_type" not in df.columns
    if not need_parse:
        return df

    def parse_path(p: str):
        parts = [s.strip() for s in p.split("|")] if isinstance(p, str) else []
        if not parts or parts[0] != "standard_charge":
            return {"charge_scope": None, "payer": None, "plan": None, "charge_type": None}
        parts = parts[1:]
        if not parts:
            return {"charge_scope": None, "payer": None, "plan": None, "charge_type": None}
        if parts[0] in {"gross", "discounted_cash", "min", "max"}:
            return {"charge_scope": "global", "payer": None, "plan": None, "charge_type": parts[0]}
        payer = parts[0] if len(parts) >= 1 else None
        plan  = parts[1] if len(parts) >= 2 else None
        ctype = parts[2] if len(parts) >= 3 else None
        return {"charge_scope": "negotiated", "payer": payer, "plan": plan, "charge_type": ctype}

    parsed = df["standard_charge_path"].map(parse_path).apply(pd.Series)
    return pd.concat([df, parsed], axis=1)

def bridge_semi_wide_to_long(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    If DF looks like 'semi-wide' (payer_name/plan_name present + 'standard_charge|...' columns),
    reshape to long with columns: charge_type, amount, amount_raw.
    """
    sc_cols = [c for c in df.columns if c.startswith("standard_charge|")]
    if not sc_cols:
        return None
    if ("payer_name" not in df.columns) and ("plan_name" not in df.columns):
        return None

    id_cols = [c for c in df.columns if c not in sc_cols]
    long = df.melt(id_vars=id_cols, value_vars=sc_cols, var_name="charge_col", value_name="value_raw")
    long["charge_type"] = long["charge_col"].str.split("|").str[-1].str.strip()
    is_meth = long["charge_type"].str.lower().eq("methodology")
    long["amount_raw"] = long["value_raw"]
    long["amount"] = np.where(is_meth, np.nan, long["value_raw"].map(to_float))
    long = long.drop(columns=["charge_col", "value_raw"])
    long["charge_scope"] = np.where(long["charge_type"].isin(["gross","discounted_cash","min","max"]), "global", "negotiated")
    return _colstrip(long)

def pick_meta_value(meta: Dict[str, str], df: pd.DataFrame, keys: List[str], df_meta_prefix="meta__") -> str:
    """Choose hospital-level value from metadata rows; fallback to meta__ columns in df; otherwise ""."""
    if meta:
        norm_map = {_norm_key(k): k for k in meta.keys()}
        for cand in keys:
            n = _norm_key(cand)
            if n in norm_map:
                return meta.get(norm_map[n], "")
        for cand in keys:
            n = _norm_key(cand)
            for nk, ok in norm_map.items():
                if nk.startswith(n):
                    return meta.get(ok, "")
    for cand in keys:
        col = f"{df_meta_prefix}{_norm_key(cand)}"
        if col in df.columns:
            v = df[col].dropna().astype(str).str.strip()
            if len(v) > 0:
                return v.iloc[0]
    return ""

def get_series(df: pd.DataFrame, name: str, fill_value=None) -> pd.Series:
    """Safe accessor: always returns a Series aligned to df.index."""
    if name in df.columns:
        s = df[name]
    else:
        s = pd.Series([fill_value] * len(df), index=df.index)
    return s

def normalize_long_df(df_in: pd.DataFrame, meta: Dict[str, str]) -> pd.DataFrame:
    """Normalize a long (or bridged semi-wide) dataframe to the target schema."""
    bridged = bridge_semi_wide_to_long(df_in)
    df = bridged.copy() if bridged is not None else df_in.copy()

    if "amount" in df.columns:
        df["amount"] = df["amount"].apply(lambda x: to_float(x) if isinstance(x, str) else (x if pd.notna(x) else np.nan))
    else:
        series = df["amount_raw"] if "amount_raw" in df.columns else pd.Series(np.nan, index=df.index)
        df["amount"] = series.map(to_float)

    df = parse_path_if_needed(df)

    if "payer_name" not in df.columns:
        df["payer_name"] = df.get("payer", df.get("Payer", np.nan))
    if "plan_name" not in df.columns:
        df["plan_name"] = df.get("plan", df.get("market", df.get("Plan", np.nan)))
    if "code" not in df.columns:
        df["code"] = df.get("code|1", df.get("primary_code", df.get("cpt_code", df.get("hcpcs", df.get("hcpcs_code", "")))))
    if "code_type" not in df.columns:
        df["code_type"] = df.get("code|1|type", df.get("primary_code_type", df.get("cpt_hcpcs", "")))
    if "additional_generic_notes" not in df.columns and "additional_payer_notes" in df.columns:
        df = df.rename(columns={"additional_payer_notes": "additional_generic_notes"})

    svc_keys = [k for k in ["description","code","code_type","setting","drug_unit_of_measurement","drug_type_of_measurement","modifiers","estimated_amount","additional_generic_notes"] if k in df.columns]

    scope = get_series(df, "charge_scope", "").fillna("").astype(str).str.lower()

    globals_df = df[scope.eq("global")]
    if globals_df.empty and "charge_type" in df.columns:
        globals_df = df[df["charge_type"].isin(["gross","discounted_cash","min","max"])]
    if not globals_df.empty:
        g_pivot = globals_df.pivot_table(index=svc_keys, columns="charge_type", values="amount", aggfunc="first").reset_index().rename(columns={
            "gross": "standard_charge_gross",
            "discounted_cash": "standard_charge_discounted_cash",
            "min": "standard_charge_min",
            "max": "standard_charge_max",
        })
    else:
        g_pivot = pd.DataFrame(columns=svc_keys + ["standard_charge_gross","standard_charge_discounted_cash","standard_charge_min","standard_charge_max"])

    nego_df = df[scope.eq("negotiated")]
    if nego_df.empty and "charge_type" in df.columns:
        nego_df = df[df["charge_type"].isin(["negotiated_dollar","negotiated_percentage","negotiated_algorithm","methodology"])]
    if not nego_df.empty:
        meth = nego_df[nego_df["charge_type"] == "methodology"]
        money = nego_df[nego_df["charge_type"] != "methodology"]

        meth_pivot = meth.pivot_table(index=svc_keys + ["payer_name","plan_name"], values="amount_raw", aggfunc="first").reset_index().rename(columns={"amount_raw":"standard_charge_methodology"})
        money_pivot = money.pivot_table(index=svc_keys + ["payer_name","plan_name"], columns="charge_type", values="amount", aggfunc="first").reset_index().rename(columns={
            "negotiated_dollar": "standard_charge_negotiated_dollar",
            "negotiated_percentage": "standard_charge_negotiated_percentage",
            "negotiated_algorithm": "standard_charge_negotiated_algorithm",
        })
        n_pivot = pd.merge(money_pivot, meth_pivot, on=svc_keys + ["payer_name","plan_name"], how="left")
    else:
        base = df[svc_keys].drop_duplicates() if svc_keys else pd.DataFrame({"__dummy__":[1]})
        n_pivot = base.copy()
        if "__dummy__" in n_pivot.columns:
            n_pivot = n_pivot.drop(columns=["__dummy__"])
        for col in ["payer_name","plan_name","standard_charge_negotiated_dollar","standard_charge_negotiated_percentage","standard_charge_negotiated_algorithm","standard_charge_methodology"]:
            if col not in n_pivot.columns:
                n_pivot[col] = np.nan

    if not g_pivot.empty and not n_pivot.empty:
        merged = pd.merge(n_pivot, g_pivot, on=svc_keys, how="left")
    elif not n_pivot.empty:
        merged = n_pivot.copy()
        for col in ["standard_charge_gross","standard_charge_discounted_cash","standard_charge_min","standard_charge_max"]:
            if col not in merged.columns:
                merged[col] = np.nan
    else:
        merged = g_pivot.copy()
        for col in ["payer_name","plan_name","standard_charge_negotiated_dollar","standard_charge_negotiated_percentage","standard_charge_negotiated_algorithm","standard_charge_methodology"]:
            if col not in merged.columns:
                merged[col] = np.nan

    hospital_name        = pick_meta_value(meta, df, ["hospital_name","facility_name","provider_name"])
    last_updated_on      = pick_meta_value(meta, df, ["last_updated_on","last_updated","file_last_updated","last_updated_date","last_update"])
    version              = pick_meta_value(meta, df, ["version"])
    hospital_location    = pick_meta_value(meta, df, ["hospital_location","facility_location","city_state"])
    hospital_address_raw = pick_meta_value(meta, df, ["hospital_address","facility_address","address","street_address","address_line_1"])
    hospital_address     = hospital_address_raw.replace(",", "") if isinstance(hospital_address_raw, str) else hospital_address_raw
    license_number       = pick_meta_value(meta, df, ["license_number","license_no","license","license_#","licensenumber"])

    merged["hospital_name"] = hospital_name
    merged["last_updated_on"] = last_updated_on
    merged["version"] = version
    merged["hospital_location"] = hospital_location
    merged["hospital_address"] = hospital_address
    merged["license_number"] = license_number

    for col in TARGET_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[TARGET_COLUMNS]
    merged = _colstrip(merged)
    return merged

def process_folder_normalize_long(input_dir: Path, output_dir: Path, combined_out: Optional[Path] = None, pattern: str = "*.csv") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    outs: List[pd.DataFrame] = []
    report_rows = []
    files = sorted(input_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {input_dir}")
    for p in files:
        try:
            df_long, meta, enc = read_table_allow_meta(p)
            df_norm = normalize_long_df(df_long, meta)
            outp = output_dir / f"{p.stem}__normalized.csv"
            df_norm.to_csv(outp, index=False)
            outs.append(df_norm)
            msg = f"OK ({len(df_norm)} rows; encoding={enc})"
            print(f"[OK] {p.name}: {msg}")
            report_rows.append({"filename": p.name, "status": "OK", "rows_out": len(df_norm), "encoding": enc, "message": ""})
        except Exception as e:
            emsg = str(e)
            print(f"[SKIPPED] {p.name}: {emsg}")
            report_rows.append({"filename": p.name, "status": "SKIPPED", "rows_out": 0, "encoding": "", "message": emsg})
    if outs:
        if combined_out is None:
            combined_out = output_dir / "combined_normalized.csv"
        combined = pd.concat(outs, ignore_index=True, sort=False)
        combined.to_csv(combined_out, index=False)
        print(f"[COMBINED] {combined_out} ({len(combined)} rows)")
    report_df = pd.DataFrame(report_rows, columns=["filename","status","rows_out","encoding","message"])
    report_path = output_dir / f"normalization_report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    report_df.to_csv(report_path, index=False)
    print(f"[REPORT] {report_path}")
    return output_dir

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Normalize LONG/semi-wide hospital standard charges CSVs (address commas removed).")
    ap.add_argument("--input_dir", type=str, default="", help="Folder containing CSVs (default: script folder)")
    ap.add_argument("--output_dir", type=str, default="", help="Folder to write normalized CSVs (default: <script folder>/normalized_long)")
    ap.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for input files (default: *.csv)")
    ap.add_argument("--combined_out", type=str, default="", help="Path to a single combined normalized CSV (optional; default: <output_dir>/combined_normalized.csv)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_dir = Path(os.path.expandvars(args.input_dir)).expanduser().resolve() if args.input_dir else script_dir
    output_dir = Path(os.path.expandvars(args.output_dir)).expanduser().resolve() if args.output_dir else (script_dir / "normalized_long")
    combined_out = Path(os.path.expandvars(args.combined_out)).expanduser().resolve() if args.combined_out else None

    process_folder_normalize_long(input_dir, output_dir, combined_out=combined_out, pattern=args.pattern)
