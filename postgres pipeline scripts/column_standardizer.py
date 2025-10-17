# column_standardizer.py

import pandas as pd
import re
import os
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


# ---------- SETTINGS ----------
INPUT_FOLDER = Path(r"C:\Users\gio12\Desktop\New folder\meta_data_rows_removed")                       # folder with raw wide CSVs
OUTPUT_FOLDER = Path(r"C:\Users\gio12\Desktop\New folder\column_standardized")   # cleaned outputs live here
FAILED_FOLDER = Path(r"C:\Users\gio12\Desktop\New folder\column_standardized_failed")      # raw files that failed processing

OUTPUT_FOLDER.mkdir(exist_ok=True)
FAILED_FOLDER.mkdir(exist_ok=True)

# ---------- HEADER NORMALIZER ----------

def normalize_header(col: str) -> str | None:
    """
    Normalize a single column header according to the specified rules.
    Return None to DROP the column.
    """
    col = str(col).replace("\ufeff", "").strip()  # strip BOM if present
    lower = col.lower()

    # 1) Drop any column containing "notes" (case-insensitive)
    if "notes" in lower:
        return None

    # 2) code|N or code_N  -> "code: N"
    m = re.fullmatch(r"code[|_](\d+)", lower)
    if m:
        return f"code: {m.group(1)}"

    # 3) code|N|type or code_N_type  -> "code_type: N"
    m = re.fullmatch(r"code[|_](\d+)[|_]type", lower)
    if m:
        return f"code_type: {m.group(1)}"

    # 4) fixed standard_charge columns (case-insensitive)
    fixed_map = {
        "standard_charge|gross": "standard_charge_gross",
        "standard_charge|discounted_cash": "standard_charge_discounted_cash",
        "standard_charge|min": "standard_charge_min",
        "standard_charge|max": "standard_charge_max",
    }
    fixed_map_lower = {k.lower(): v for k, v in fixed_map.items()}
    if lower in fixed_map_lower:
        return fixed_map_lower[lower]

    # 5) dynamic payer/plan fields
    #    - standard_charge|<PAYER>|<PLAN>|(negotiated_dollar|negotiated_percentage|negotiated_algorithm|methodology)
    #    - estimated_amount|<PAYER>|<PLAN>
    parts = col.split("|")
    if len(parts) >= 3:
        if parts[0].strip().lower() == "standard_charge":
            field = parts[-1].strip().lower()
            payer_plan = " ".join(p.strip() for p in parts[1:-1] if p.strip())
            field_map = {
                "negotiated_dollar": "standard_charge_negotiated_dollar",
                "negotiated_percentage": "standard_charge_negotiated_percentage",
                "negotiated_algorithm": "standard_charge_negotiated_algorithm",
                "methodology": "standard_charge_methodology",
            }
            if field in field_map and payer_plan:
                return f"{field_map[field]} : {payer_plan}"

        if parts[0].strip().lower() == "estimated_amount":
            payer_plan = " ".join(p.strip() for p in parts[1:] if p.strip())
            if payer_plan:
                return f"estimated_amount: {payer_plan}"

    # 6) leave untouched if no rule applies
    return col


def rename_and_drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_names = []
    keep_mask = []

    for col in df.columns:
        new_name = normalize_header(col)
        if new_name is None:
            keep_mask.append(False)
            new_names.append(None)
        else:
            keep_mask.append(True)
            new_names.append(new_name)

    # keep only columns not flagged for drop
    df = df.loc[:, keep_mask]
    df.columns = [n for n in new_names if n is not None]
    return df


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Strip leading/trailing spaces in string cells
    - Remove '%' and ',' from all values (string context)
    """
    # strip whitespace
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col].dtype):
            df[col] = df[col].astype(str).str.strip()

    # remove % and , globally (works on object dtype)
    df = df.replace({r"[%,]": ""}, regex=True)
    return df


def process_file(path: Path) -> bool:
    """
    Read -> rename/drop columns -> clean values -> write cleaned file to OUTPUT_FOLDER.
    On success: delete the original.
    On failure: move the original to FAILED_FOLDER.
    """
    try:
        # robust encoding fallbacks
        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
        last_err = None
        for enc in encodings:
            try:
                df = pd.read_csv(path, dtype=str, encoding=enc, low_memory=False)
                break
            except UnicodeDecodeError as e:
                last_err = e
                df = None
        if df is None:
            raise last_err or UnicodeDecodeError("unknown", b"", 0, 1, "encoding error")

        # transform
        df = rename_and_drop_columns(df)
        # df = clean_values(df)

        df.columns = [col.replace('|', '_') for col in df.columns]
        df.columns = [col.lower() for col in df.columns]

        # write cleaned output
        out_path = OUTPUT_FOLDER / path.name
        df.to_csv(out_path, index=False)

        # delete original (so only cleaned copy remains in converted/)
        try:
            path.unlink()
        except Exception:
            # If deletion fails (e.g., locked), attempt to move to converted as _raw
            raw_shadow = OUTPUT_FOLDER / f"{path.stem}__raw{path.suffix}"
            shutil.move(str(path), raw_shadow)

        print(f"✅ Processed {path.name}")
        return True

    except Exception as e:
        print(f"❌ Failed {path.name} | Error: {e}")
        try:
            shutil.move(str(path), FAILED_FOLDER / path.name)
        except Exception:
            # if move fails, leave it in place but report
            print(f"⚠️ Could not move {path.name} to failed folder (possibly locked).")
        return False
    
def main():
    """
    Process all CSV files in the input folder once. Uses processes for parallelism.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", "-j", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                    help="Worker processes (default: cores-1)")
    ap.add_argument("--pattern", default="*.csv",
                    help="Glob pattern to match files (default: *.csv)")
    args = ap.parse_args()

    files = sorted(INPUT_FOLDER.glob(args.pattern))
    if not files:
        print(f"No files matched {args.pattern} in {INPUT_FOLDER}")
        return

    print(f"🚀 Parallel run: {len(files)} files | workers={args.jobs}")
    ok = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(process_file, f): f.name for f in files}
        for fut in as_completed(futs):
            fname = futs[fut]
            try:
                success = fut.result()
                ok += 1 if success else 0
                print(("✅" if success else "❌"), fname)
            except Exception as e:
                print(f"❌ {fname} | {e}")

    print(f"Done. {ok}/{len(files)} succeeded.")



if __name__ == "__main__":
    main()


