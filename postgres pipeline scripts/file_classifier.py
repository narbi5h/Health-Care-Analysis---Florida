#!/usr/bin/env python3
"""
route_files_by_columns.py

Moves files into folders based on column characteristics.

- Skips the first 2 rows as metadata (ignored for header)
- Standardizes headers:
    * lowercase
    * spaces -> _
    * '|' -> _
    * normalizes variants like code#1 / code 1 / code|1 / code_1 -> code_1
- Fuzzy/alias mapping for minor header discrepancies.
- Routes into:
    • single_payer_multi_code
    • long
    • wide  (many dynamic payer columns like standard_charge|<payer>|<plan>|..., and code_3+)

Usage
-----
# Dry-run (no moves), verbose
python route_files_by_columns.py --in ./incoming --out ./routed --dry-run --verbose

# Actually move files, scanning subfolders too
python route_files_by_columns.py --in ./incoming --out ./routed --recurse

# Include Excel inspection as well
python route_files_by_columns.py --in ./incoming --out ./routed --xlsx
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set
from difflib import SequenceMatcher

try:
    import pandas as pd  # only needed if --xlsx is used
except Exception:
    pd = None


# -------------------------
# Canonical columns per type
# -------------------------

FIRST_TYPE_FOLDER = "single_payer_multi_code"
FIRST_TYPE_CORE: Set[str] = {
    "description",
    "code_1",
    "code_1_type",
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
    "additional_generic_notes",
}
FIRST_TYPE_OPTIONAL: Set[str] = {"code_2", "code_2_type"}  # not required but allowed

LONG_TYPE_FOLDER = "long"
LONG_TYPE_CORE: Set[str] = {
    "description",
    "code_1",
    "code_1_type",
    # "setting",
    # "drug_unit_of_measurement",
    # "drug_type_of_measurement",
    # "standard_charge_gross",
    # "standard_charge_discounted_cash",
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
    # "additional_generic_notes",
}

WIDE_TYPE_FOLDER = "wide"
# Wide also frequently includes these base fields (plus many dynamic payer columns)
WIDE_BASE_RECOGNIZED: Set[str] = {
    "description",
    "billing_class",
    "setting",
    "drug_unit_of_measurement",
    "drug_type_of_measurement",
    "modifiers",
    "standard_charge_gross",
    "standard_charge_discounted_cash",
    "standard_charge_min",
    "standard_charge_max",
    "additional_generic_notes",
    # codes 1..n (we detect dynamically)
}

# Universe of canonical names we try to match against (for fuzzy mapping)
CANON_UNIVERSE: Set[str] = (
    FIRST_TYPE_CORE
    | FIRST_TYPE_OPTIONAL
    | LONG_TYPE_CORE
    | WIDE_BASE_RECOGNIZED
)


# -------------------------
# Header normalization & mapping
# -------------------------

# Regex alias patterns -> canonical name (handles common variants)
ALIAS_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'^code[#_\-\s]?(\d+)$'), lambda m: f"code_{m.group(1)}"),
    (re.compile(r'^code[_\s\-]*([0-9]+)[_\s\-]*type$'), lambda m: f"code_{m.group(1)}_type"),
    (re.compile(r'^(std|standard)[_\s\-]*chg[_\s\-]*gross$'), lambda m: "standard_charge_gross"),
    (re.compile(r'^(std|standard)[_\s\-]*chg[_\s\-]*discounted[_\s\-]*cash$'), lambda m: "standard_charge_discounted_cash"),
    (re.compile(r'^(std|standard)[_\s\-]*chg[_\s\-]*negotiated[_\s\-]*dollar$'), lambda m: "standard_charge_negotiated_dollar"),
    (re.compile(r'^(std|standard)[_\s\-]*chg[_\s\-]*negotiated[_\s\-]*percentage$'), lambda m: "standard_charge_negotiated_percentage"),
    (re.compile(r'^(std|standard)[_\s\-]*chg[_\s\-]*negotiated[_\s\-]*algorithm$'), lambda m: "standard_charge_negotiated_algorithm"),
    (re.compile(r'^(std|standard)[_\s\-]*chg[_\s\-]*methodology$'), lambda m: "standard_charge_methodology"),
    (re.compile(r'^(std|standard)[_\s\-]*chg[_\s\-]*min$'), lambda m: "standard_charge_min"),
    (re.compile(r'^(std|standard)[_\s\-]*chg[_\s\-]*max$'), lambda m: "standard_charge_max"),
    (re.compile(r'^drug[_\s\-]*unit[_\s\-]*of[_\s\-]*measurement$'), lambda m: "drug_unit_of_measurement"),
    (re.compile(r'^drug[_\s\-]*type[_\s\-]*of[_\s\-]*measurement$'), lambda m: "drug_type_of_measurement"),
    (re.compile(r'^additional[_\s\-]*generic[_\s\-]*notes$'), lambda m: "additional_generic_notes"),
    (re.compile(r'^billing[_\s\-]*class$'), lambda m: "billing_class"),
]

def _alias_map(token: str) -> Optional[str]:
    for pat, repl in ALIAS_PATTERNS:
        m = pat.match(token)
        if m:
            return repl(m) if callable(repl) else repl
    return None

def normalize_header(name: str) -> str:
    """
    Standardization required:
      - lowercase
      - spaces -> _
      - '|' -> _
      - remove non [a-z0-9_#-] -> '_'
      - collapse multiple underscores
      - trim underscores
      - normalize code-number variants (code#1 / code 1 / code|1 / code_1 -> code_1)
    """
    s = str(name).strip().lower()
    s = s.replace('|', '_').replace(' ', '_')
    s = re.sub(r'[^0-9a-z_#\-]+', '_', s)
    s = re.sub(r'__+', '_', s).strip('_')
    # normalize "code" + number variants
    s = re.sub(r'^code[#_\-\s]?(\d+)$', r'code_\1', s)
    return s

def normalize_headers(headers: List[str]) -> List[str]:
    out = [normalize_header(h) for h in headers]
    # Apply alias patterns
    mapped = [(_alias_map(h) or h) for h in out]
    # Dedup (rare collisions)
    seen: Dict[str, int] = {}
    deduped: List[str] = []
    for h in mapped:
        if h not in seen:
            seen[h] = 1
            deduped.append(h)
        else:
            seen[h] += 1
            deduped.append(f"{h}__{seen[h]}")
    return deduped

def fuzzy_map_to_canonical(headers: List[str], universe: Set[str], threshold: float = 0.86) -> List[str]:
    """
    For each header, if it's not already canonical, map to the closest canonical
    name with similarity >= threshold. Otherwise keep as-is.
    """
    canon_list = list(universe)
    result = []
    for h in headers:
        if h in universe:
            result.append(h)
            continue
        # try alias mapping again just in case
        alias = _alias_map(h)
        if alias and alias in universe:
            result.append(alias)
            continue
        # fuzzy match
        best = None
        best_score = 0.0
        for c in canon_list:
            s = SequenceMatcher(None, h, c).ratio()
            if s > best_score:
                best, best_score = c, s
        result.append(best if best and best_score >= threshold else h)
    return result


# -------------------------
# Header reading (skip 2 meta rows)
# -------------------------

def sniff_delimiter(sample: str) -> str:
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=[',', '\t', ';', '|'])
        return dialect.delimiter
    except Exception:
        counts = {d: sample.count(d) for d in [',', '\t', ';', '|']}
        return max(counts, key=counts.get) or ','

def read_header_from_csv(path: Path, meta_rows: int = 2, encoding_candidates: Tuple[str, ...] = ('utf-8-sig', 'utf-8', 'latin-1')) -> Optional[List[str]]:
    for enc in encoding_candidates:
        try:
            with path.open('r', encoding=enc, errors='strict', newline='') as f:
                peek = f.read(4096)
                if not peek:
                    return None
                delim = sniff_delimiter(peek)
                f.seek(0)
                for _ in range(meta_rows):
                    f.readline()
                reader = csv.reader(f, delimiter=delim)
                header = next(reader, None)
                if header is None:
                    return None
                return [h.strip() for h in header]
        except Exception:
            continue
    return None

def read_header_from_xlsx(path: Path, meta_rows: int = 2) -> Optional[List[str]]:
    if pd is None:
        return None
    try:
        df = pd.read_excel(path, header=meta_rows, nrows=0)
        return list(df.columns.astype(str))
    except Exception:
        return None


# -------------------------
# Type detection
# -------------------------

def contains_all(required: Set[str], cols: Set[str]) -> bool:
    return required.issubset(cols)

def _max_code_index(cols: Set[str]) -> int:
    max_idx = 0
    for c in cols:
        m = re.match(r'^code_(\d+)(?:$|__\d+)', c)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx

def _collect_dynamic_payer_cols(cols: Set[str]) -> Dict[str, Set[str]]:
    """
    Detect dynamic 'wide' columns. After normalization, a typical name looks like:
      standard_charge_<payer>_<plan>_<field>
      estimated_amount_<payer>_<plan>
      additional_payer_notes_<payer>_<plan>
    We try to capture the payer_key '<payer>_<plan>' and the field.
    Returns a mapping payer_key -> set(fields_seen).
    """
    payer_map: Dict[str, Set[str]] = {}

    # fields of interest for dynamic blocks
    dyn_suffixes = {
        "negotiated_dollar",
        "negotiated_percentage",
        "negotiated_algorithm",
        "methodology",
        # estimated_amount handled below without suffix list
    }

    for h in cols:
        parts = h.split('_')
        if len(parts) < 3:
            continue

        if parts[0] in {"standard", "estimated", "additional"}:
            # Some files might normalize 'standard_charge' as 'standard_charge' already,
            # but if someone typed 'standard|charge', normalization becomes 'standard_charge'.
            pass

        # standard_charge_* pattern
        if h.startswith("standard_charge_"):
            # Expect >= 4 parts: standard_charge, payer..., plan..., field
            # We'll find the field as one of dyn_suffixes at the end, and everything between forms payer_key
            for suf in dyn_suffixes:
                if h.endswith(f"_{suf}") or h.endswith(f"__{suf}"):
                    stem = h[:-(len(suf)+1)]  # remove trailing _<suf>
                    payer_key = stem.replace("standard_charge_", "")
                    # collapse repeated underscores (rare)
                    payer_key = re.sub(r'__+', '_', payer_key).strip('_')
                    payer_map.setdefault(payer_key, set()).add(suf)
                    break

        # estimated_amount_* pattern (no trailing negotiated_* field)
        elif h.startswith("estimated_amount_"):
            payer_key = h.replace("estimated_amount_", "")
            payer_key = re.sub(r'__+', '_', payer_key).strip('_')
            payer_map.setdefault(payer_key, set()).add("estimated_amount")

        # additional_payer_notes_* pattern
        elif h.startswith("additional_payer_notes_"):
            payer_key = h.replace("additional_payer_notes_", "")
            payer_key = re.sub(r'__+', '_', payer_key).strip('_')
            payer_map.setdefault(payer_key, set()).add("additional_payer_notes")

    return payer_map

def is_first_type(norm_or_fuzzy: List[str]) -> bool:
    cols = set(norm_or_fuzzy)
    if not contains_all(FIRST_TYPE_CORE, cols):
        return False
    # Ensure at least one code_* exists (should be true via core)
    if not any(re.match(r'^code_\d+($|__\d+)', c) for c in cols):
        return False
    return True

def is_long_type(norm_or_fuzzy: List[str]) -> bool:
    cols = set(norm_or_fuzzy)
    if not contains_all(LONG_TYPE_CORE, cols):
        return False
    if "code_1" not in cols:
        return False
    return True

def is_wide_type(norm_or_fuzzy: List[str]) -> bool:
    """
    Heuristics for 'wide':
      - Many dynamic payer columns starting with standard_charge_ / estimated_amount_ / additional_payer_notes_
      - And EITHER:
          * max code index >= 3 (code_3, code_4, ...)
        OR
          * there are "enough" dynamic payer groups (e.g., >= 5)
      - Base wide columns often include billing_class (optional), standard_charge_min/max, etc.
    """
    cols = set(norm_or_fuzzy)
    payer_map = _collect_dynamic_payer_cols(cols)
    num_groups = len(payer_map)
    total_dyn_fields = sum(len(v) for v in payer_map.values())
    max_code = _max_code_index(cols)

    # Strong signal: lots of dynamic payer fields
    many_dynamic = (total_dyn_fields >= 15) or (num_groups >= 5)

    # Also consider moderate dynamic with multiple codes
    moderate_dynamic = (total_dyn_fields >= 6 and max_code >= 2)

    # If it looks like a wide table (payer blocks) and has >2 codes OR many payer blocks, call it wide
    if many_dynamic or moderate_dynamic:
        return True

    # Secondary signal: presence of billing_class + at least some dynamic fields + code_3+
    if "billing_class" in cols and total_dyn_fields >= 3 and max_code >= 2:
        return True

    return False


# -------------------------
# Routing
# -------------------------

TEXT_EXT = {'.csv', '.tsv', '.txt'}
EXCEL_EXT = {'.xlsx', '.xls'}

def classify_file(path: Path, use_xlsx: bool, meta_rows: int = 2, verbose: bool = False) -> Optional[str]:
    ext = path.suffix.lower()
    headers = None

    if ext in TEXT_EXT:
        headers = read_header_from_csv(path, meta_rows=meta_rows)
    elif use_xlsx and ext in EXCEL_EXT:
        headers = read_header_from_xlsx(path, meta_rows=meta_rows)

    if headers is None:
        if verbose:
            print(f"[skip] Could not read header for {path.name}")
        return None

    normalized = normalize_headers(headers)
    # fuzzy-map into known canonical base names where appropriate
    fuzzy = fuzzy_map_to_canonical(normalized, CANON_UNIVERSE)

    if verbose:
        print(f"[hdr ] {path.name}:")
        print(f"       raw   -> {headers}")
        print(f"       norm  -> {normalized}")
        print(f"       fuzzy -> {fuzzy}")

    # Order matters: wide check first (since wide also contains some of the 'long' base names)
    if is_wide_type(fuzzy):
        return WIDE_TYPE_FOLDER
    if is_first_type(fuzzy):
        return FIRST_TYPE_FOLDER
    if is_long_type(fuzzy):
        return LONG_TYPE_FOLDER

    return None

def move_file(file_path: Path, out_root: Path, subfolder: str, dry_run: bool, verbose: bool) -> None:
    target_dir = out_root / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / file_path.name

    counter = 1
    while dest.exists():
        dest = target_dir / f"{file_path.stem}__{counter}{file_path.suffix}"
        counter += 1

    if dry_run:
        print(f"[dry] {file_path} -> {dest}")
    else:
        shutil.move(str(file_path), str(dest))
        if verbose:
            print(f"[move] {file_path} -> {dest}")

def walk_files(in_root: Path, recurse: bool, include_xlsx: bool) -> List[Path]:
    exts = set(TEXT_EXT)
    if include_xlsx:
        exts |= EXCEL_EXT
    files: List[Path] = []
    if recurse:
        for p in in_root.rglob('*'):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    else:
        for p in in_root.iterdir():
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    return files

def main():
    ap = argparse.ArgumentParser(description="Move files into folders based on column characteristics.")
    ap.add_argument('--in', dest='in_dir', required=True, help="Input folder to scan")
    ap.add_argument('--out', dest='out_dir', required=True, help="Output root where subfolders will be created")
    ap.add_argument('--recurse', action='store_true', help="Scan subdirectories recursively")
    ap.add_argument('--xlsx', action='store_true', help="Also inspect Excel files (.xlsx/.xls). Requires pandas.")
    ap.add_argument('--meta-rows', type=int, default=2, help="Number of metadata rows before header (default: 2)")
    ap.add_argument('--dry-run', action='store_true', help="Show what would move, but do not move files")
    ap.add_argument('--verbose', action='store_true', help="Verbose logging")
    args = ap.parse_args()

    in_root = Path(args.in_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.xlsx and pd is None:
        print("[warn] --xlsx specified but pandas is not available; Excel files will be skipped.", flush=True)

    files = walk_files(in_root, recurse=args.recurse, include_xlsx=args.xlsx)

    if args.verbose:
        print(f"[info] Scanning {len(files)} file(s) in {in_root}")

    moved = 0
    for f in files:
        category = classify_file(f, use_xlsx=args.xlsx, meta_rows=args.meta_rows, verbose=args.verbose)
        if category:
            move_file(f, out_root, category, dry_run=args.dry_run, verbose=args.verbose)
            moved += 1
        elif args.verbose:
            print(f"[hold] {f.name}: no matching rule (left in place)")

    print(f"[done] Routed {moved} file(s).")

if __name__ == "__main__":
    main()
