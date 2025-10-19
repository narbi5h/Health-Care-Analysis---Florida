#!/usr/bin/env python3
"""
Auto verify & stage:
- Confirms CSVs have 2 metadata rows followed by a tabular header row
- Supports "metadata table" layout: row0 = metadata headers, row1 = metadata values, row2 = tabular header
- Sniffs delimiter/quote and encoding (UTF-8 -> CP1252 -> Latin-1)
- Ensures ≥ 3 lines total (2 metadata + ≥1 tabular row)
- Moves valid files -> ready_for_processing/
- Moves invalid files -> failed_initial_file_integrity/

Edit the CONFIG section below and run this file. No CLI args needed.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import csv
import io
import shutil
import time

# =========================
# CONFIG
# =========================
INPUT_DIR  =  Path(__file__).parent
READY_DIR  = INPUT_DIR / "ready_for_pipeline"
FAILED_DIR = INPUT_DIR / "failed_initial_file_integrity"
PATTERN    = "*.csv"


# If True, keep polling INPUT_DIR every POLL_INTERVAL_SEC seconds
WATCH_MODE = False
POLL_INTERVAL_SEC = 30
# =========================

PREFERRED_ENCODINGS = ("utf-8", "cp1252", "latin-1")
DELIMS = [",", ";", "\t", "|"]

def remove_empty_rows_inplace(path: Path) -> None:
    """
    Remove rows where all fields are empty/whitespace, in place.
    Keeps detected delimiter/quote style, writes UTF-8.
    """
    # Sniff encoding & dialect using your existing helpers
    sample_bytes = read_sample_bytes(path)
    sample_text, _enc = try_decode(sample_bytes)
    dialect = sniff_dialect(sample_text)

    tmp = path.with_suffix(path.suffix + ".tmp_clean")

    # Read with best-effort decoding, write UTF-8
    # newline='' is important for csv module on Windows
    with path.open("r", encoding="utf-8", errors="replace", newline="") as r, \
         tmp.open("w", encoding="utf-8", newline="") as w:
        reader = csv.reader(r, dialect)
        writer = csv.writer(
            w,
            delimiter=getattr(dialect, "delimiter", ","),
            quotechar=getattr(dialect, "quotechar", '"'),
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
            doublequote=True
        )
        for row in reader:
            # Consider it empty if every cell is empty or whitespace
            if not row or all((c is None) or (str(c).strip() == "") for c in row):
                continue
            writer.writerow(row)

    # Replace original with cleaned
    tmp.replace(path)



def read_sample_bytes(path: Path, size: int = 8192) -> bytes:
    with path.open("rb") as f:
        return f.read(size)

def try_decode(data: bytes) -> Tuple[str, str]:
    for enc in PREFERRED_ENCODINGS:
        try:
            return (data.decode(enc), enc)
        except UnicodeDecodeError:
            continue
    # Last resort: latin-1 with replacement never fails
    return (data.decode("latin-1", errors="replace"), "latin-1")

def sniff_dialect(sample_text: str) -> csv.Dialect:
    sniffer = csv.Sniffer()
    try:
        return sniffer.sniff(sample_text, delimiters="".join(DELIMS))
    except Exception:
        class Fallback(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL
        return Fallback()

def count_total_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)

def is_header_like(fields: List[str]) -> bool:
    """Heuristic: 2+ columns, many alpha tokens, not mostly numeric, no giant blobs."""
    if len(fields) < 2:
        return False
    clean = [f.strip() for f in fields]
    if any(len(f) > 200 for f in clean):
        return False
    has_alpha = sum(1 for f in clean if any(ch.isalpha() or ch == "_" for ch in f))
    numeric_only = sum(1 for f in clean if f and all(ch.isdigit() or ch in ".-+" for ch in f))
    return has_alpha >= max(1, len(clean) // 2) and numeric_only < len(clean) * 0.5

def looks_like_metadata_line(fields: List[str], header_cols: int) -> bool:
    """Original heuristic kept: metadata often 1 col, different width, or long blob."""
    if len(fields) <= 1:
        return True
    if header_cols >= 2 and abs(len(fields) - header_cols) >= max(1, header_cols // 2):
        return True
    if any(len((f or "").strip()) > 200 for f in fields):
        return True
    return False

def parse_first_n_rows(text: str, dialect: csv.Dialect, n: int = 5) -> List[List[str]]:
    buf = io.StringIO(text)
    reader = csv.reader(buf, dialect)
    rows = []
    for i, row in enumerate(reader):
        rows.append(row)
        if i + 1 >= n:
            break
    return rows

def validate_file(path: Path) -> Tuple[bool, str, Optional[str], Optional[csv.Dialect]]:
    """
    Returns (is_valid, reason, encoding, dialect).
    Accepts both classic layout and metadata-table layout (row0 headers, row1 values, row2 tabular header).
    """
    try:
        total_lines = count_total_lines(path)
    except Exception as e:
        return (False, f"I/O error reading file: {e}", None, None)

    if total_lines < 3:
        return (False, f"Insufficient lines ({total_lines}); need at least 3", None, None)

    # Sniff encoding & dialect
    sample_bytes = read_sample_bytes(path)
    sample_text, enc = try_decode(sample_bytes)
    dialect = sniff_dialect(sample_text)

    # Read a bit of the file with strict decode; fallback to replacement
    try:
        with path.open("r", encoding=enc, errors="strict", newline="") as f:
            preview_text = f.read(16384)
    except UnicodeDecodeError:
        with path.open("r", encoding=enc, errors="replace", newline="") as f:
            preview_text = f.read(16384)

    rows = parse_first_n_rows(preview_text, dialect, n=5)
    if len(rows) < 3:
        return (False, f"Could not parse 3 rows with dialect ({getattr(dialect, 'delimiter', ',')})", enc, dialect)

    r0, r1, r2 = rows[0], rows[1], rows[2]

    # --- SPECIAL CASE: metadata-table short-circuit (your file pattern)
    def _tokset(row):
        return {c.strip().lower() for c in row if isinstance(c, str) and c.strip()}

    metadata_keys = {
        "hospital_name","last_updated_on","version",
        "hospital_location","hospital_address","license_number","license_number|fl"
    }
    tabular_hints = {
        "description","code","code|1","standard_charge|gross",
        "payer_name","plan_name","setting"
    }

    has_meta_header = len(metadata_keys & _tokset(r0)) >= 3
    has_tabular_header = len(tabular_hints & _tokset(r2)) >= 2
    if has_meta_header and has_tabular_header:
        # Accept: row0 = metadata headers, row1 = metadata values, row2 = tabular header
        return (True, "OK (metadata-table layout detected)", enc, dialect)

    # --- Original heuristic path
    header_candidate = r2
    if not is_header_like(header_candidate):
        return (False, "Third row does not look like a tabular header", enc, dialect)

    header_cols = len(header_candidate)
    meta1_ok = looks_like_metadata_line(r0, header_cols)
    meta2_ok = looks_like_metadata_line(r1, header_cols)

    # Relaxation: if row0 looks like metadata AND row2 is header-like, allow row1 to be metadata too
    if meta1_ok and not meta2_ok and is_header_like(r2):
        meta2_ok = True

    if not (meta1_ok and meta2_ok):
        reason = []
        if not meta1_ok: reason.append("row 1 not metadata-like")
        if not meta2_ok: reason.append("row 2 not metadata-like")
        return (False, "; ".join(reason), enc, dialect)

    return (True, "OK", enc, dialect)

def move_file(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        stem, suffix = src.stem, src.suffix
        i = 1
        while True:
            candidate = dest_dir / f"{stem}__{i}{suffix}"
            if not candidate.exists():
                dest = candidate
                break
            i += 1
    shutil.move(str(src), str(dest))
    return dest

def process_once() -> None:
    READY_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"[ERROR] Input folder does not exist: {INPUT_DIR}")
        return

    files = sorted(INPUT_DIR.glob(PATTERN))
    if not files:
        print("[INFO] No files matched the pattern.")
        return

    ok_count = 0
    fail_count = 0

    for f in files:
        if not f.is_file():
            continue

        # 1) Clean in place: drop fully empty rows first
        try:
            remove_empty_rows_inplace(f)
        except Exception as e:
            # If we can't even clean, send to failed and continue
            moved_to = move_file(f, FAILED_DIR)
            print(f"❌ FAILED (cleaning error): {f.name} -> {moved_to} | reason: {e}")
            fail_count += 1
            continue

        # 2) Now validate the cleaned file
        valid, reason, enc, dialect = validate_file(f)
        if valid:
            moved_to = move_file(f, READY_DIR)
            delims = getattr(dialect, "delimiter", ",")
            quote = getattr(dialect, "quotechar", '"')
            print(f"✅ READY: {f.name} -> {moved_to} | enc={enc} delim='{delims}' quote='{quote}' | {reason}")
            ok_count += 1
        else:
            moved_to = move_file(f, FAILED_DIR)
            print(f"❌ FAILED: {f.name} -> {moved_to} | reason: {reason}")
            fail_count += 1


    print(f"[SUMMARY] {ok_count} ready, {fail_count} failed.")

def main():
    if not WATCH_MODE:
        process_once()
        return
    print(f"[WATCH] Scanning {INPUT_DIR} every {POLL_INTERVAL_SEC}s for {PATTERN}")
    while True:
        process_once()
        time.sleep(POLL_INTERVAL_SEC)

if __name__ == "__main__":
    main()
