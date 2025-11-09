import os
import shutil
import pandas as pd
from pathlib import Path
import csv
from pandas.errors import ParserError

# ------------ CONFIG ------------
INPUT_FOLDER = Path(__file__).parent / "ready_for_pipeline"
PROCESSED_FOLDER = Path(__file__).parent / "meta_data_rows_removed"
FAILED_FOLDER = Path(__file__).parent / "meta_data_rows_removed_failed"
SKIPROWS = 2  # number of metadata rows to skip at the top of each CSV #NOT SURE WHY THIS WAS MISSING


# Ensure output folders exist
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(FAILED_FOLDER, exist_ok=True)

# Delimiter sniffing helpers (paste these below the imports)
DELIMS = [",", ";", "\t", "|"]

def _sniff_delim(path: Path, enc: str) -> str:
    sample = path.read_text(encoding=enc, errors="replace")[:16384]
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMS)).delimiter
    except Exception:
        if "\t" in sample: return "\t"
        if "|"  in sample: return "|"
        if ";"  in sample: return ";"
        return ","

def _pick_encoding(path: Path) -> str:
    for e in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=e, errors="strict") as f:
                f.read(256)
            return e
        except UnicodeDecodeError:
            continue
    return "latin-1"

def _coerce_to_header_len(path: Path, enc: str, sep: str, skiprows: int = 2) -> tuple[Path, int]:
    """
    Build a cleaned temp CSV using csv.reader/writer.
    - Skips the first `skiprows` metadata rows
    - Writes header + data
    - If a row has too many fields, merges the extras into the last column
    - If too few, pads with blanks
    Returns (temp_path, num_fixed_rows)
    """
    tmp = path.with_suffix(path.suffix + ".tmp_clean")
    fixed = 0
    with open(path, "r", encoding=enc, errors="replace", newline="") as r, \
         open(tmp,  "w", encoding="utf-8", newline="") as w:
        reader = csv.reader(r, delimiter=sep, quotechar='"', doublequote=True)
        writer = csv.writer(w, delimiter=sep, quotechar='"', doublequote=True, lineterminator="\n")

        # Skip metadata rows
        for _ in range(skiprows):
            next(reader, None)

        header = next(reader, None)
        if header is None:
            raise ValueError("File has no header after skiprows=2")

        header_len = len(header)
        writer.writerow(header)

        for row in reader:
            if len(row) > header_len:
                row = row[:header_len-1] + [sep.join(row[header_len-1:])]
                fixed += 1
            elif len(row) < header_len:
                row = row + [""] * (header_len - len(row))
                fixed += 1
            writer.writerow(row)

    return tmp, fixed
    
def process_csv(file_path):
    file_path = Path(file_path)
    try:
        enc = _pick_encoding(file_path)
        sep = _sniff_delim(file_path, enc)

        try:
            # Python engine, NO low_memory arg
            df = pd.read_csv(
                file_path,
                skiprows=2,
                dtype=str,
                sep=sep,
                engine="python",
                quotechar='"',
                doublequote=True,
                escapechar="\\",
                on_bad_lines="error",
                encoding=enc,
            )
        except ParserError:
            # Coerce bad rows to header width, then read cleaned temp (still python engine, no low_memory)
            tmp, fixed = _coerce_to_header_len(file_path, enc, sep, skiprows=2)
            try:
                df = pd.read_csv(
                    tmp,
                    dtype=str,
                    sep=sep,
                    engine="python",
                    quotechar='"',
                    doublequote=True,
                    escapechar="\\",
                    on_bad_lines="error",
                    encoding="utf-8",
                )
            finally:
                tmp.replace(file_path)
            print(f"⚠️  Fixed {fixed} row(s) with mismatched columns in {file_path.name}")

        # Clean up empty rows
        df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all").reset_index(drop=True)

        # Save back (preserve detected delimiter; change to comma by removing sep=sep if you want)
        df.to_csv(file_path, index=False, encoding="utf-8", sep=sep)
        print(f"✅ Success: {file_path.name} (sep='{sep}', enc='{enc}')")
        return True

    except Exception as e:
        print(f"❌ Failed: {file_path.name} | Error: {e}")
        return False


def main():
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith(".csv"):
            file_path = os.path.join(INPUT_FOLDER, file)

            if process_csv(file_path):
                shutil.move(file_path, os.path.join(PROCESSED_FOLDER, file))
            else:
                shutil.move(file_path, os.path.join(FAILED_FOLDER, file))

if __name__ == "__main__":
    main()
# ------------ END CONFIG ------------