# ================================
# Hospital Files Classifier & Renamer
# LONG / WIDE / JSON / NONE + pincode
# Google Colab + Google Drive
# ================================



import re
import os
import json
import shutil
import pandas as pd
from pathlib import Path

# -------- CONFIGURE THIS --------
# BASE_DIR = Path("D:/BANA 650/HOSPITAL FILES/") # <-- put your folder here
BASE_DIR = Path("//narbfileserver/BANA650GROUP/HOSPITAL FILES")
DRY_RUN = False   # True = preview only, False = actually move/rename
# --------------------------------

# Create output folders
for sub in ["long", "wide", "json", "none"]:
    (BASE_DIR / sub).mkdir(parents=True, exist_ok=True)

INDEX_CSV = BASE_DIR / "master_file_index.csv"

# ---------- Helpers ----------
def clean_name(s: str, fallback="UnknownHospital"):
    """Make a safe, short hospital name for filenames."""
    if not isinstance(s, str) or not s.strip():
        return fallback
    s = re.sub(r"[^A-Za-z0-9]+", "_", s.strip())
    return (s or fallback)[:60]

def extract_pincode(address: str) -> str:
    """Return 5-digit ZIP from hospital_address (or 00000)."""
    if not isinstance(address, str):
        return "00000"
    m = re.search(r"\b\d{5}\b", address)
    return m.group(0) if m else "00000"

def read_first_rows(path: Path, n=6):
    """Quickly read first N lines as lists of cells (comma-split, tolerant)."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break
            rows.append([c.strip() for c in line.rstrip("\r\n").split(",")])
    return rows

def locate_header_row(rows):
    """
    Heuristics to find the real header row (often row 3 in CMS CSVs):
    - contains 'description' or 'setting'
    - or 'payer_name' / 'plan_name'
    - or any column startswith 'standard_charge'
    """
    best = 0
    for i, r in enumerate(rows[:5]):
        lower = [c.lower() for c in r]
        if "description" in lower or "setting" in lower:
            return i
        if "payer_name" in lower or "plan_name" in lower:
            return i
        if any(str(c).lower().startswith("standard_charge") for c in r):
            best = i
    return best

def classify_headers(headers):
    """
    Decide LONG vs WIDE vs NONE using header content:
    - WIDE: any header like standard_charge|<payer>|<plan>|<field> (>=3 pipes)
    - LONG: has payer_name or plan_name (payer/plan in rows)
            (also allow LONG if code|i present + description, and no WIDE pattern)
    - NONE: otherwise
    """
    H = [h.strip().lower() for h in headers]

    # WIDE: payer/plan embedded in header with >=3 pipes
    for h in H:
        if h.startswith("standard_charge|") and h.count("|") >= 3:
            return "WIDE"

    # LONG: explicit payer/plan columns
    if "payer_name" in H or "plan_name" in H:
        return "LONG"

    # Additional hint for LONG when code|i exists with description (no WIDE)
    if any(re.match(r"^code\|\d+$", h) for h in H) and "description" in H:
        return "LONG"

    return "NONE"

def extract_metadata_from_rows(rows):
    """
    Try to get hospital_name, state (from license_number|XX), pincode (from hospital_address)
    using the top two rows as key/value pairs when present.
    """
    hospital = "UnknownHospital"
    state = "XX"
    pincode = "00000"

    if not rows:
        return hospital, state, pincode

    row0_lower = [c.lower() for c in rows[0]]
    row1 = rows[1] if len(rows) > 1 else []

    # key/value mapping if row0 holds keys
    def val_for(key):
        if key in row0_lower:
            idx = row0_lower.index(key)
            return row1[idx] if idx < len(row1) else ""
        return ""

    # hospital_name
    hname = val_for("hospital_name")
    hospital = clean_name(hname) if hname else hospital

    # hospital_address -> ZIP
    haddr = val_for("hospital_address")
    pincode = extract_pincode(haddr) if haddr else pincode

    # license_number|XX in keys -> state
    for k in row0_lower:
        if k.startswith("license_number|") and "|" in k and len(k.split("|")[1]) == 2:
            state = k.split("|")[1].upper()
            break

    return hospital, state, pincode

def json_get_state(data):
    """
    Attempt to extract state from JSON:
    - license_information.state (string)
    - or license_information is a dict/list with .state inside
    """
    try:
        lic = data.get("license_information", None)
        if isinstance(lic, dict):
            st = lic.get("state")
            if isinstance(st, str) and len(st) == 2:
                return st.upper()
        elif isinstance(lic, list):
            # pick first entry that has a 2-letter state
            for item in lic:
                if isinstance(item, dict):
                    st = item.get("state")
                    if isinstance(st, str) and len(st) == 2:
                        return st.upper()
    except Exception:
        pass
    return "XX"

def detect_type_and_metadata(path: Path):
    """
    Classify file and extract metadata for naming.
    Returns: (ftype, hospital, state, pincode)
    """
    ext = path.suffix.lower()

    # JSON
    if ext == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            ftype = "JSON"
            hospital = clean_name(data.get("hospital_name", "UnknownHospital"))
            # Prefer JSON license_information.state if present
            state = json_get_state(data)
            pincode = extract_pincode(data.get("hospital_address", ""))
            return ftype, hospital, state, pincode
        except Exception:
            return "NONE", "UnknownHospital", "XX", "00000"

    # CSV
    if ext == ".csv":
        rows = read_first_rows(path, n=6)
        if not rows:
            return "NONE", "UnknownHospital", "XX", "00000"
        header_idx = locate_header_row(rows)
        headers = rows[header_idx] if header_idx < len(rows) else []
        ftype = classify_headers(headers)
        hospital, state, pincode = extract_metadata_from_rows(rows)
        return ftype, hospital, state, pincode

    # Other → NONE
    return "NONE", "UnknownHospital", "XX", "00000"

def build_new_name(hospital, state, pincode, ftype, ext):
    return f"{hospital}_{state}_{pincode}_{ftype}{ext}"

# ---------- Main ----------
records = []
type_counts = {"LONG":0, "WIDE":0, "JSON":0, "NONE":0}

for item in BASE_DIR.iterdir():
    # skip subfolders and our index file
    if item.is_dir() or item.name == INDEX_CSV.name:
        continue

    ftype, hospital, state, pincode = detect_type_and_metadata(item)
    type_counts[ftype] = type_counts.get(ftype, 0) + 1

    new_name = build_new_name(hospital, state, pincode, ftype, item.suffix.lower())
    dest_dir = BASE_DIR / ftype.lower()
    dest_path = dest_dir / new_name

    # ensure uniqueness
    counter = 1
    while dest_path.exists():
        dest_path = dest_dir / f"{dest_path.stem}__{counter}{dest_path.suffix}"
        counter += 1

    status = "ok"
    if DRY_RUN:
        # Preview only
        print(f"[DRY RUN] {item.name}  -->  {dest_path}")
    else:
        try:
            shutil.move(str(item), str(dest_path))
        except Exception as e:
            status = f"error: {e}"
            dest_path = item

    records.append({
        "original": item.name,
        "new_name": dest_path.name if hasattr(dest_path, "name") else "",
        "type": ftype,
        "hospital": hospital,
        "state": state,
        "pincode": pincode,
        "status": status
    })

# write audit log
df = pd.DataFrame(records)
df.to_csv(INDEX_CSV, index=False)
print("\nLog written to:", INDEX_CSV)

# quick summary
print("\nSummary counts by type:")
for k in ["LONG","WIDE","JSON","NONE"]:
    print(f"  {k}: {type_counts.get(k,0)}")

print("\nSample of planned/resulting names:")
print(df.head(12))


#TEST
