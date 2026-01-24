import sys, os

################# LOGGING EVERYTHING TO A FILE SO I DON'T HAVE TO READ THE OUTPUT IN THE LITTLE CONSOLE WINDOW BC THAT'S ANNOYING###################################
from pathlib import Path
from datetime import datetime  # not strictly needed for fixed name, but harmless

repo_root = Path(__file__).resolve().parent.parent
out_dir = repo_root / "outputs"
out_dir.mkdir(exist_ok=True)

log_path = out_dir / "healthcare_run.log"  # overwrite each run

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"[INFO] Logging to {log_path.resolve()}")
################################################################
print(">>> healthcare_utilities.py STARTING")
print(">>> __file__:", __file__)
print(">>> cwd:", os.getcwd())
print(">>> python exe:", sys.executable)


from dotenv import load_dotenv, find_dotenv
from dotenv import dotenv_values
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import scipy as sp
from scipy import stats
from sklearn.ensemble import IsolationForest

print(">>> healthcare_utilities.py STARTING")

##########################################********************* DATA LOADING & CLEANING SECTION
#### Database Connection Setup
load_dotenv()

# force-load .env from the project root (parent of notebook/)
env_path = Path(__file__).resolve().parent.parent / ".env"

raw = env_path.read_bytes()
print("[DEBUG] .env size bytes:", len(raw))
print("[DEBUG] first 80 bytes:", raw[:80])
print("[DEBUG] contains NUL? ", b"\x00" in raw)

text = env_path.read_text(encoding="utf-8", errors="replace")
print("[DEBUG] first 5 lines as python sees them:")
print("\n".join(text.splitlines()[:5]))

load_dotenv(env_path, override=True)
print("[DEBUG] dotenv_values keys:", dotenv_values(env_path).keys())
print("[DEBUG] dotenv_values full:", dotenv_values(env_path))

print("[INFO] Loaded .env from:", env_path)

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "iamr007.ddns.net")
DB_NAME = os.getenv("DB_NAME", "hospital_db")

if not DB_USER or not DB_PASS:
    raise ValueError("Missing DB_USER or DB_PASS in .env / environment")

_raw_port = os.getenv("DB_PORT")
DB_PORT = "2345" if not _raw_port or _raw_port.strip().lower() == "none" else _raw_port.strip()

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
safe_url = f"postgresql+psycopg2://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print("[DEBUG] DB_URL =", safe_url)

engine = create_engine(DB_URL, pool_pre_ping=True)
print("[INFO] Database engine created.")
print(">>> about to run pd.read_sql()")

sql = """
SELECT
    cpt_code,
    hospital_name,
    "Rate_using_min",
    "Rate_using_max",
    bucket,
    specification
FROM public.standardized_cpt_columns

"""

chunks = pd.read_sql(sql, engine, chunksize=100_000)

dfs = []
for i, chunk in enumerate(chunks, start=1):
    print(f"Loaded chunk {i}, shape={chunk.shape}")
    dfs.append(chunk)

df = pd.concat(dfs, ignore_index=True)
print(">>> SQL query finished, df shape =", df.shape)


# TEMP: work on a sample so you don't melt your RAM

# TARGET_N = 500_000

# if len(df) > TARGET_N:
#     df = df.sample(n=TARGET_N, random_state=42)
#     print(">>> After sampling, df shape =", df.shape)
# else:
#     print(">>> Skipping sampling; df has only", len(df), "rows")


# ---- normalize rate columns from DB ----
# Map DB columns Rate_using_min / Rate_using_max → internal rate_min / rate_max.
for src, dst in {
    "Rate_using_min": "rate_min",
    "Rate_using_max": "rate_max",
}.items():
    if src in df.columns and dst not in df.columns:
        df[dst] = pd.to_numeric(df[src], errors="coerce")

# If the table doesn't have standard_charge_gross, alias it to rate_min so
# the rest of the script (which expects standard_charge_gross) doesn't break.
if "standard_charge_gross" not in df.columns and "rate_min" in df.columns:
    df["standard_charge_gross"] = df["rate_min"]



# ===== SANITY PING =====
print("\n=== SANITY PING: df ===")
print("Shape:", df.shape)
print("First 8 cols:", list(df.columns[:8]))

# Normalize common column-name variants (do this FIRST)
_rename = {
    "facility_id":"source_file", "hospital_id":"source_file", "filename":"source_file",
    "cpt_code":"code", "hcpcs_code":"code",
    "charge":"standard_charge_gross", "gross_charge":"standard_charge_gross", "standard_charge":"standard_charge_gross",
    "min_charge":"standard_charge_min", "max_charge":"standard_charge_max",
    "cash_price":"standard_charge_discounted_cash", "self_pay_price":"standard_charge_discounted_cash",
    "payer":"payer_name", "payer_display_name":"payer_name",
    "plan":"plan_name",
}
df.rename(columns={k:v for k,v in _rename.items() if k in df.columns}, inplace=True)

# Ensure required columns exist (AFTER rename)
# We only *hard-require* code + standard_charge_gross.
required_core = {"code", "standard_charge_gross"}
missing = required_core - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in df: {sorted(missing)}")

# Derive a hospital identifier for source_file if it doesn't exist.
# Many of your tables use hospital_name instead.
if "source_file" not in df.columns:
    if "hospital_name" in df.columns:
        df["source_file"] = df["hospital_name"]
        print(">>> Created source_file from hospital_name")
    else:
        # absolute fallback so later groupbys don't die
        df["source_file"] = "UNKNOWN"
        print(">>> WARNING: no hospital identifier, using 'UNKNOWN' as source_file")
# ===== /SANITY PING =====


# 3 Coerce obvious numeric columns 
numeric_cols = [
    "standard_charge_gross",
    "standard_charge_discounted_cash",
    "standard_charge_negotiated_dollar",
    "standard_charge_negotiated_percentage",
    "estimated_amount",
    "standard_charge_min",
    "standard_charge_max",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
##########$$$$$$$$$$$$$$$$$  SUPER IMPORTANT SECTION - CHOOSE RATE SOURCE FOR ANALYSIS ###########################
##################################################################################################################
#############$$$$$$$$$$$$$$$%%%%%%%%%%##############$$$$$$$$$$#########@!!!!!#@@@@@@@@@###########################
# ---- choose which rate drives ALL downstream analysis ----
def _use_rate(col_name: str):
    """
    Overwrite standard_charge_gross with the selected rate column.
    All later code keeps using standard_charge_gross, but it now
    represents whatever rate column you choose (min or max).
    """
    assert col_name in df.columns, f"Missing column: {col_name}"
    df["standard_charge_gross"] = pd.to_numeric(df[col_name], errors="coerce")
    print(f"\n[PRICE SOURCE] standard_charge_gross ← {col_name}")

# Default to MIN per teammate’s guidance:
# _use_rate("rate_min")
# If you want to rerun everything off max later, change that to:
_use_rate("rate_max")

###^^^^^^^^^^^^^^^^^^^^^ THIS ALLOWS FOR US TO SWITCH BETWEEN MIN/MAX RATES FOR THE ENTIRE ANALYSIS  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##########################################********************* QUALITATIVE COLUMNS ALREADY ADDED. THE NUMERIC_COLS IS JUST A SUBSET   


# # 4 (Optional later) Load specs in separate DFs for future joins  (SOFT OPTIONAL)
# try:
#     payer_specs = pd.read_sql(
#         """
#         SELECT
#             name,
#             bucket,
#             specification
#         FROM public.payer_specs
#         """,
#         engine,
#     )
#     print("Loaded payer_specs:", payer_specs.shape)
#     if not {"name","bucket"}.issubset(payer_specs.columns):
#         raise ValueError("payer_specs must have columns: 'name' and 'bucket'")
# except Exception as e:
#     payer_specs = None
#     print("payer_specs not available:", repr(e))
# # ---- single source of truth about payer_specs presence ----
# HAS_PAYER_SPECS = payer_specs is not None
# print("HAS_PAYER_SPECS =", HAS_PAYER_SPECS)

# # Only if/when plan_specs exists
# try:
#     plan_specs = pd.read_sql(
#         """
#         SELECT
#             name,
#             bucket,
#             specification
#         FROM public.plan_specs
#         """,
#         engine,
#     )
# except Exception:
#     plan_specs = None

# At this point:
# - df          = clean working dataset for ANY analysis
# - payer_specs = mapping table you can merge when you decide how
# - plan_specs  = optional mapping table (or None)

################************* SUMMARY STATS SECTION  



# ---------- 1. BASIC STRUCTURE ----------
# Row/column counts
n_rows, n_cols = df.shape
print("Rows:", n_rows, "Columns:", n_cols)

# Column data types
print(df.dtypes)

# Missing values per column
nulls = df.isna().sum().sort_values(ascending=False)
print(nulls)


# ---------- 2. NUMERIC SUMMARY (KEY PRICE COLS) ----------

numeric_cols = [
    "standard_charge_gross",
    "standard_charge_discounted_cash",
    "standard_charge_negotiated_dollar",
    "standard_charge_negotiated_percentage",
    "estimated_amount",
    "standard_charge_min",
    "standard_charge_max",
]

# keep only those that actually exist
numeric_cols = [c for c in numeric_cols if c in df.columns]

if numeric_cols:
    numeric_summary = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    print(numeric_summary)
else:
    print("No numeric columns found for summary.")


# ---------- 3. TOP PAYERS / PLANS (COUNTS ONLY FOR NOW) ----------

if "specification" in df.columns:
    spec_counts = (
        df["specification"]
        .value_counts(dropna=False)
        .head(25)
    )
    print("\nTop 25 specifications by row count:")
    print(spec_counts)


# ---------- 4. SIMPLE CHARGE STATS BY PAYER (NO BUCKETS YET) ----------

if "specification" in df.columns and "standard_charge_gross" in df.columns:
    spec_charge_summary = (
        df.groupby("specification", dropna=False)["standard_charge_gross"]
          .agg(
              rows="count",
              mean="mean",
              median="median",
              p25=lambda x: x.quantile(0.25),
              p75=lambda x: x.quantile(0.75),
              min="min",
              max="max",
          )
          .sort_values("rows", ascending=False)
          .head(25)
    )
    print("\nSpecification-level summary (top 25 by rows):")
    print(spec_charge_summary)


# ---------- 5. BUCKET-LEVEL SUMMARY (using bucket already in standardized_cpt_columns) ----------

if {"bucket", "standard_charge_gross"}.issubset(df.columns):
    tmp = df.copy()

    bucket_summary = (
        tmp.groupby("bucket", dropna=False)["standard_charge_gross"]
           .agg(
               rows="count",
               mean="mean",
               median="median",
               p25=lambda x: x.quantile(0.25),
               p75=lambda x: x.quantile(0.75),
               min="min",
               max="max",
           )
           .sort_values("rows", ascending=False)
    )
    print("\nBucket-level summary (using df.bucket):")
    print(bucket_summary)
        
        
# # =========================
# # 3) PRICE-LEVEL KPIs (GLOBAL)
# # =========================
# # Safe numeric copy
# #######$$$$$$$$%%%%%%%%%%  NO OUTLIERS for this one
# # [3] Global price KPIs — ROBUST (DROPS outliers)

# PRICE_COLS = ["standard_charge_gross",
#               "standard_charge_discounted_cash",
#               "standard_charge_min",
#               "standard_charge_max"]

# present = [c for c in PRICE_COLS if c in df.columns]
# if not present:
#     print("\n[3] Skipping KPIs: no price columns present.")
# else:
#     # --- NO-OUTLIER (trimmed) view ---
#     _num = df.copy()
#     for c in present:
#         _num[c] = pd.to_numeric(_num[c], errors="coerce")

#     # clip 2%–98% per column
#     for c in present:
#         lo, hi = _num[c].quantile([0.02, 0.98])
#         _num[c] = _num[c].clip(lo, hi)

#     # outlier flag per (hospital, code) on gross
#     if {"source_file", "code", "standard_charge_gross"}.issubset(_num.columns):
#         def _tukey_flags(s):
#             q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
#             return (s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)
#         _num["is_outlier"] = (
#             _num.groupby(["source_file","code"])["standard_charge_gross"]
#                 .transform(lambda s: _tukey_flags(s).astype(bool))
#         )
#         _num = _num.loc[~_num["is_outlier"]].copy()

#     # ratios (safe div) basically no divide by zero issues and stuff like that
#     if {"standard_charge_discounted_cash","standard_charge_gross"}.issubset(_num.columns):
#         _num["cash_over_gross"] = _num["standard_charge_discounted_cash"] / _num["standard_charge_gross"].replace(0, np.nan)
#     if {"standard_charge_max","standard_charge_min"}.issubset(_num.columns):
#         _num["max_over_min"] = _num["standard_charge_max"] / _num["standard_charge_min"].replace(0, np.nan)
#     if {"standard_charge_gross","standard_charge_min"}.issubset(_num.columns):
#         _num["gross_minus_min"] = _num["standard_charge_gross"] - _num["standard_charge_min"]
#     if {"standard_charge_max","standard_charge_gross"}.issubset(_num.columns):
#         _num["max_minus_gross"] = _num["standard_charge_max"] - _num["standard_charge_gross"]

#     kpi_cols_no = [c for c in ["cash_over_gross","max_over_min","gross_minus_min","max_minus_gross"] if c in _num.columns]
#     if kpi_cols_no:
#         global_kpis_no = _num[kpi_cols_no].describe(percentiles=[0.25,0.5,0.75]).T
#         print("\n[3] Global price KPIs (NO outliers):")
#         print(global_kpis_no)
#     else:
#         print("\n[3] No ratio KPIs computed in NO-outlier view.")

#     # --- INCLUDING OUTLIERS Raw view ---
#     _raw = df.copy()
#     for c in present:
#         _raw[c] = pd.to_numeric(_raw[c], errors="coerce")

#     if {"standard_charge_discounted_cash","standard_charge_gross"}.issubset(_raw.columns):
#         _raw["cash_over_gross"] = _raw["standard_charge_discounted_cash"] / _raw["standard_charge_gross"].replace(0, np.nan)
#     if {"standard_charge_max","standard_charge_min"}.issubset(_raw.columns):
#         _raw["max_over_min"] = _raw["standard_charge_max"] / _raw["standard_charge_min"].replace(0, np.nan)
#     if {"standard_charge_gross","standard_charge_min"}.issubset(_raw.columns):
#         _raw["gross_minus_min"] = _raw["standard_charge_gross"] - _raw["standard_charge_min"]
#     if {"standard_charge_max","standard_charge_gross"}.issubset(_raw.columns):
#         _raw["max_minus_gross"] = _raw["standard_charge_max"] - _raw["standard_charge_gross"]

#     kpi_cols_incl = [c for c in ["cash_over_gross","max_over_min","gross_minus_min","max_minus_gross"] if c in _raw.columns]
#     if kpi_cols_incl:
#         global_kpis_incl = _raw[kpi_cols_incl].describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).T
#         print("\n[3] Global price KPIs (INCLUDING outliers):")
#         print(global_kpis_incl)
#     else:
#         print("\n[3] No ratio KPIs computed in RAW view.")



# =========================
# 5) VARIATION BY CPT CODE (dispersion)
# =========================
TRIM_LO, TRIM_HI = 0.01, 0.99    # adjust if you want tighter/looser tails
RUN_RAW_COMPARISON_5 = False     # set True to also compute/print RAW (untrimmed)

if {"code","standard_charge_gross"}.issubset(df.columns):
    _v = df.copy()
    _v["standard_charge_gross"] = pd.to_numeric(_v["standard_charge_gross"], errors="coerce")

    # ----------------------------------------------------------------------
    # >>> GLOBAL TRIM (comment out if you prefer PER-CODE TRIM below)
    #lo, hi = _v["standard_charge_gross"].quantile([TRIM_LO, TRIM_HI])
    #_v["gross_trim"] = _v["standard_charge_gross"].clip(lo, hi)
    # ----------------------------------------------------------------------

    # ======================================================================
    # >>> PER-CODE TRIM <<<
    _v["gross_trim"] = (
        _v.groupby("code")["standard_charge_gross"]
          .transform(lambda s: s.clip(*s.quantile([TRIM_LO, TRIM_HI])))
    )
    # ======================================================================

    def q25(x): return x.quantile(0.25)
    def q75(x): return x.quantile(0.75)

    # ---- TRIMMED dispersion by code ----
    code_stats = (_v.groupby("code")["gross_trim"]
                    .agg(rows="count",
                         vmin="min",
                         q25=q25,
                         median="median",
                         q75=q75,
                         vmax="max",
                         mean="mean",
                         std="std")
                    .reset_index())

    # optional counts
    if "specification" in _v.columns:
        code_stats["spec_count"] = _v.groupby("code")["specification"].nunique().values
    if "source_file" in _v.columns:
        code_stats["hospital_count"] = _v.groupby("code")["source_file"].nunique().values

    # derived metrics (trimmed)
    code_stats["cv"] = code_stats["std"] / code_stats["mean"]
    code_stats["iqr_over_median"] = (code_stats["q75"] - code_stats["q25"]) / code_stats["median"]
    # tail width on trimmed values (still informative)
    code_stats["range_ratio"] = code_stats["vmax"] / code_stats["vmin"]

    code_stats = code_stats.sort_values("iqr_over_median", ascending=False)
    print("\n[5] Code-level dispersion — TRIMMED (head):")
    print(code_stats.head(20))

    # ---- Optional RAW (untrimmed) comparison ----
    if RUN_RAW_COMPARISON_5:
        _r = df.copy()
        _r["standard_charge_gross"] = pd.to_numeric(_r["standard_charge_gross"], errors="coerce")

        code_stats_raw = (_r.groupby("code")["standard_charge_gross"]
                            .agg(rows="count",
                                 vmin="min",
                                 q05=lambda x: x.quantile(0.05),
                                 q25=q25,
                                 median="median",
                                 q75=q75,
                                 q95=lambda x: x.quantile(0.95),
                                 vmax="max",
                                 mean="mean",
                                 std="std")
                            .reset_index())

        if "specification" in _r.columns:
            code_stats_raw["spec_count"] = _r.groupby("code")["specification"].nunique().values
        if "source_file" in _r.columns:
            code_stats_raw["hospital_count"] = _r.groupby("code")["source_file"].nunique().values

        code_stats_raw["cv"] = code_stats_raw["std"] / code_stats_raw["mean"]
        code_stats_raw["iqr_over_median"] = (code_stats_raw["q75"] - code_stats_raw["q25"]) / code_stats_raw["median"]
        code_stats_raw["p95_over_p05"] = code_stats_raw["q95"] / code_stats_raw["q05"]
        code_stats_raw["range_ratio"] = code_stats_raw["vmax"] / code_stats_raw["vmin"]

        code_stats_raw = code_stats_raw.sort_values("p95_over_p05", ascending=False)

        print("\n[5] Code-level dispersion — RAW (head):")
        print(code_stats_raw.head(20))


# =========================
# 6) CPT-LEVEL VARIATION DECOMPOSITION
#    (within-facility vs across-facility, but reported ONLY at CPT level)
# =========================

_TRIM_LO_6, _TRIM_HI_6 = 0.01, 0.99

_required = {"source_file", "code", "standard_charge_gross"}
if _required.issubset(df.columns):
    _w = df.copy()
    _w["standard_charge_gross"] = pd.to_numeric(_w["standard_charge_gross"], errors="coerce")
    _w = _w.dropna(subset=["source_file", "code", "standard_charge_gross"])

    # Per-(hospital, CPT) trim to suppress tails while keeping local shape
    _w["gross_trim"] = (
        _w.groupby(["source_file", "code"])["standard_charge_gross"]
          .transform(lambda s: s.clip(*s.quantile([_TRIM_LO_6, _TRIM_HI_6])) if s.notna().any() else s)
    )

    # 1) Within-hospital dispersion for each (hospital, CPT)
    within = (
        _w.groupby(["source_file", "code"], dropna=False)["gross_trim"]
          .agg(
              within_median="median",
              within_iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
              rows="count",
          )
          .reset_index()
    )

    # 2) Cross-hospital dispersion of those within-medians (per CPT)
    cross = (
        within.groupby("code", dropna=False)["within_median"]
              .agg(
                  cross_std="std",
                  cross_iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
                  cross_p10=lambda x: x.quantile(0.10),
                  cross_p90=lambda x: x.quantile(0.90),
              )
              .reset_index()
    )

    # 3) Summarize “within” at CPT level (no hospital output)
    within_by_code = (
        within.groupby("code", dropna=False)
              .agg(
                  hospitals_reporting=("source_file", "nunique"),
                  hospital_code_pairs=("source_file", "count"),
                  median_within_iqr=("within_iqr", "median"),
                  median_within_median=("within_median", "median"),
                  median_rows_per_hosp=("rows", "median"),
              )
              .reset_index()
    )

    # 4) Merge into one CPT-level “decomposition” table
    code_var_decomp = within_by_code.merge(cross, on="code", how="left")

    # A simple “where is variation coming from?” indicator
    code_var_decomp["cross_over_within_iqr"] = (
        code_var_decomp["cross_iqr"] /
        code_var_decomp["median_within_iqr"].replace(0, np.nan)
    )

    print("\n[6] CPT-level variation decomposition (TRIMMED):")
    print(
        code_var_decomp.sort_values("cross_over_within_iqr", ascending=False)
                      .head(25)
    )

    print("\n[6] CPTs with highest across-facility spread (cross_iqr) (TRIMMED):")
    print(
        code_var_decomp.sort_values("cross_iqr", ascending=False)
                      .head(25)
    )

else:
    print("\n[6] Skipped: need columns", _required)

#### some details as to why this section is the way ti is: 
#Per-slice trim (by hospital+code) kills absurd tails without flattening real hospital differences.
# Two dispersion layers: “within” shows internal consistency; “cross” shows how hospitals disagree
# Finder table surfaces high cross variance + low internal variance codes → prime targets for audit.
# Safe merges: it only merges with code_stats if it actually exists.
# Guarded math: numeric coercion + NA handling avoids crashing on junk rows.


# =========================
# 7) PAYER EFFECTS (WITHIN HOSPITAL)
# =========================
# ==== [7] Payer effects within hospital — TRIMMED BY DEFAULT ====

TRIM_LO, TRIM_HI = 0.01, 0.99
RUN_RAW_COMPARISON_7 = False  # set True to also compute/print RAW (untrimmed)

# Now we assume bucket is already present in standardized_cpt_columns.
required_7 = {"source_file", "code", "standard_charge_gross", "bucket"}

if required_7.issubset(df.columns):
    _pe = df.copy()

    # Coerce to numeric + clean bucket
    _pe["standard_charge_gross"] = pd.to_numeric(_pe["standard_charge_gross"], errors="coerce")
    _pe["bucket"] = _pe["bucket"].fillna("UNKNOWN")

    # ----------------------------------------------------------------------
    # TRIM CHOICE A  PER-(HOSPITAL, CODE) TRIM
    # -> same cutoffs across buckets inside that hospital+code slice
    _pe["gross_trim"] = (
        _pe.groupby(["source_file", "code"])["standard_charge_gross"]
           .transform(lambda s: s.clip(*s.quantile([TRIM_LO, TRIM_HI])) if s.notna().any() else s)
    )
    # ----------------------------------------------------------------------

    # ======================================================================
    # TRIM CHOICE B : PER-(HOSPITAL, CODE, BUCKET) TRIM
    # -> each bucket gets its own cutoffs (commented out by default)
    # _pe["gross_trim"] = (
    #     _pe.groupby(["source_file", "code", "bucket"])["standard_charge_gross"]
    #        .transform(lambda s: s.clip(*s.quantile([TRIM_LO, TRIM_HI])) if s.notna().any() else s)
    # )
    # ======================================================================

    # ---- Median by (hospital, code, payer bucket), trimmed ----
    payer_bucket_medians = (
        _pe.groupby(["source_file", "code", "bucket"], dropna=False)["gross_trim"]
           .median()
           .reset_index(name="median_gross_trim")
    )
    print("\n[7] Payer-bucket medians (TRIMMED) — sample:")
    print(payer_bucket_medians.head(20))

    # # ---- Optional: per-payer (name) medians too (trimmed) ----
    # payer_name_medians = (
    #     _pe.groupby(["source_file", "code", "payer_name"], dropna=False)["gross_trim"]
    #        .median()
    #        .reset_index(name="median_gross_trim")
    # )

    # ---- Deltas vs benchmarks inside each (hospital, code) ----
    def _bench_delta(g, ref, col="median_gross_trim"):
        ref_val = g.loc[g["bucket"] == ref, col]
        ref_val = ref_val.iloc[0] if len(ref_val) else np.nan
        g[f"delta_vs_{ref}"] = g[col] - ref_val
        return g

    for ref in ["Medicare", "Self-Pay"]:
        if ref in payer_bucket_medians["bucket"].astype(str).unique():
            payer_bucket_medians = (
                payer_bucket_medians
                .groupby(["source_file", "code"], as_index=False)
                .apply(lambda g: _bench_delta(g, ref, col="median_gross_trim"))
                .reset_index(drop=True)
            )

    print("\n[7] Payer-bucket medians with deltas (TRIMMED) — sample:")
    print(payer_bucket_medians.head(20))

    # ---------------- Optional RAW (untrimmed) comparison ----------------
    if RUN_RAW_COMPARISON_7:
        _raw = _pe.copy()
        # use the original untrimmed column
        payer_bucket_medians_raw = (
            _raw.groupby(["source_file", "code", "bucket"], dropna=False)["standard_charge_gross"]
                .median()
                .reset_index(name="median_gross_raw")
        )

        def _bench_delta_raw(g, ref, col="median_gross_raw"):
            ref_val = g.loc[g["bucket"] == ref, col]
            ref_val = ref_val.iloc[0] if len(ref_val) else np.nan
            g[f"delta_vs_{ref}_raw"] = g[col] - ref_val
            return g

        for ref in ["Medicare", "Self-Pay"]:
            if ref in payer_bucket_medians_raw["bucket"].astype(str).unique():
                payer_bucket_medians_raw = (
                    payer_bucket_medians_raw
                    .groupby(["source_file", "code"], as_index=False)
                    .apply(lambda g: _bench_delta_raw(g, ref, col="median_gross_raw"))
                    .reset_index(drop=True)
                )

        print("\n[7] Payer-bucket medians with deltas — RAW (untrimmed) — sample:")
        print(payer_bucket_medians_raw.head(20))

else:
    print("\n[7] Skipped: need columns", required_7)
        
        
##################################
#    OTHER TYPES OF ANALYSES    #
###################################

# Detect weird pricing patterns per hospital+code using features that actually mean something
# (not just single values).

# ==========================================
# 8) ANOMALY DETECTION — CPT-LEVEL (NO HOSPITAL OUTPUT)
# ==========================================

TRIM_LO, TRIM_HI = 0.01, 0.99
CONTAM = 0.02  # expected anomaly fraction (2%)

req = {"source_file", "code", "standard_charge_gross"}
if not req.issubset(df.columns):
    print("\n[8] Skipped: need columns", req)
else:
    Z = df.copy()
    Z["standard_charge_gross"] = pd.to_numeric(Z["standard_charge_gross"], errors="coerce")
    Z = Z.dropna(subset=["source_file", "code", "standard_charge_gross"])

    # bucket/specification are optional; normalize if present
    if "bucket" in Z.columns:
        Z["bucket"] = Z["bucket"].fillna("UNKNOWN")
    else:
        Z["bucket"] = "UNKNOWN"

    if "specification" not in Z.columns:
        Z["specification"] = "UNKNOWN"

    # --- per-(hospital, code) winsorize to avoid dumb tails dominating ---
    Z["gross_trim"] = (
        Z.groupby(["source_file", "code"])["standard_charge_gross"]
         .transform(lambda s: s.clip(*s.quantile([TRIM_LO, TRIM_HI])) if s.notna().any() else s)
    )

    def q(p): return lambda x: x.quantile(p)

    # ---- Step 1: build (hospital, code) slice features ----
    slice_agg = (
        Z.groupby(["source_file", "code"])
         .agg(
             rows=("gross_trim", "count"),
             med=("gross_trim", "median"),
             q25=("gross_trim", q(0.25)),
             q75=("gross_trim", q(0.75)),
             p05=("gross_trim", q(0.05)),
             p95=("gross_trim", q(0.95)),
             vmin=("gross_trim", "min"),
             vmax=("gross_trim", "max"),
             mean=("gross_trim", "mean"),
             std=("gross_trim", "std"),
             spec_count=("specification", "nunique"),
             bucket_count=("bucket", "nunique"),
         )
         .reset_index()
    )

    slice_agg["iqr"] = slice_agg["q75"] - slice_agg["q25"]
    slice_agg["iqr_over_med"] = slice_agg["iqr"] / slice_agg["med"].replace(0, np.nan)
    slice_agg["p95_over_p05"] = slice_agg["p95"] / slice_agg["p05"].replace(0, np.nan)
    slice_agg["range_ratio"] = slice_agg["vmax"] / slice_agg["vmin"].replace(0, np.nan)
    slice_agg["cv"] = slice_agg["std"] / slice_agg["mean"].replace(0, np.nan)

    # Bucket mix proportions (optional, improves signal)
    mix = Z.pivot_table(
        index=["source_file", "code"],
        columns="bucket",
        values="gross_trim",
        aggfunc="count",
        fill_value=0,
    )
    mix = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0).add_prefix("mix_")
    slice_agg = slice_agg.merge(mix, on=["source_file", "code"], how="left")

    # ---- Step 2: fit anomaly model at slice-level ----
    feature_cols = [
        "rows", "med", "iqr", "iqr_over_med", "p95_over_p05",
        "range_ratio", "cv", "spec_count", "bucket_count",
    ] + [c for c in slice_agg.columns if c.startswith("mix_")]

    X = slice_agg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    X = StandardScaler().fit_transform(X)

    iso = IsolationForest(n_estimators=300, contamination=CONTAM, random_state=42)
    slice_agg["slice_anomaly_flag"] = (iso.fit_predict(X) == -1)
    slice_agg["slice_anomaly_score"] = -iso.decision_function(X)  # higher = more anomalous

    # ---- Step 3: roll slice anomalies up to CPT-level (NO hospital names) ----
    code_anoms = (
        slice_agg.groupby("code", dropna=False)
                 .agg(
                     hospitals_reporting=("source_file", "nunique"),
                     hospital_code_pairs=("source_file", "count"),
                     frac_slices_flagged=("slice_anomaly_flag", "mean"),
                     mean_slice_score=("slice_anomaly_score", "mean"),
                     p90_slice_score=("slice_anomaly_score", lambda s: s.quantile(0.90)),
                     median_rows=("rows", "median"),
                     median_iqr_over_med=("iqr_over_med", "median"),
                 )
                 .reset_index()
                 .sort_values(["p90_slice_score", "frac_slices_flagged"], ascending=False)
    )

    print("\n[8] CPT-level anomaly summary (derived from hospital+CPT slices; NO hospital output):")
    print(code_anoms.head(50))

 # ==========================================
    # 8A) GOV vs COMM: same CPT anomaly rollup, filtered
    # ==========================================

    GOV_BUCKETS  = {"Medicare", "Medicaid"}
    COMM_BUCKETS = {"Commercial"}
    MIN_SLICES = 10  # ignoring cpt's with too few hospital code slices in either of the above buckets

    def bucket_group(x):
        x = "UNKNOWN" if pd.isna(x) else str(x)
        if x in GOV_BUCKETS:
            return "GOV"
        if x in COMM_BUCKETS:
            return "COMM"
        return "OTHER"

    def cpt_anoms_for_group(df_in, group_name):
        # Filter rows to the group
        Zg = df_in.copy()
        Zg["standard_charge_gross"] = pd.to_numeric(Zg["standard_charge_gross"], errors="coerce")
        Zg = Zg.dropna(subset=["source_file", "code", "standard_charge_gross"])

        Zg["bucket"] = Zg["bucket"].fillna("UNKNOWN")
        Zg["bucket_group"] = Zg["bucket"].map(bucket_group)
        Zg = Zg[Zg["bucket_group"] == group_name]

        if Zg.empty:
            return None

        # Per-(hospital, code) winsorizing within this group
        Zg["gross_trim"] = (
            Zg.groupby(["source_file", "code"])["standard_charge_gross"]
              .transform(lambda s: s.clip(*s.quantile([TRIM_LO, TRIM_HI])) if s.notna().any() else s)
        )

        def q(p): return lambda x: x.quantile(p)

        # Slice features (hospital, code)
        slice_agg_g = (
            Zg.groupby(["source_file", "code"])
              .agg(
                  rows=("gross_trim", "count"),
                  med=("gross_trim", "median"),
                  q25=("gross_trim", q(0.25)),
                  q75=("gross_trim", q(0.75)),
                  p05=("gross_trim", q(0.05)),
                  p95=("gross_trim", q(0.95)),
                  vmin=("gross_trim", "min"),
                  vmax=("gross_trim", "max"),
                  mean=("gross_trim", "mean"),
                  std=("gross_trim", "std"),
              )
              .reset_index()
        )

        slice_agg_g["iqr"] = slice_agg_g["q75"] - slice_agg_g["q25"]
        slice_agg_g["iqr_over_med"] = slice_agg_g["iqr"] / slice_agg_g["med"].replace(0, np.nan)
        slice_agg_g["p95_over_p05"] = slice_agg_g["p95"] / slice_agg_g["p05"].replace(0, np.nan)
        slice_agg_g["range_ratio"] = slice_agg_g["vmax"] / slice_agg_g["vmin"].replace(0, np.nan)
        slice_agg_g["cv"] = slice_agg_g["std"] / slice_agg_g["mean"].replace(0, np.nan)

        feature_cols_g = [
            "rows", "med", "iqr", "iqr_over_med", "p95_over_p05",
            "range_ratio", "cv",
        ]

        Xg = slice_agg_g[feature_cols_g].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        Xg = StandardScaler().fit_transform(Xg)

        iso_g = IsolationForest(n_estimators=300, contamination=CONTAM, random_state=42)
        slice_agg_g["flag"] = (iso_g.fit_predict(Xg) == -1)
        slice_agg_g["score"] = -iso_g.decision_function(Xg)

        # Roll up to CPT-level (no hospital names)
        out = (
            slice_agg_g.groupby("code", dropna=False)
                       .agg(
                           slices=("source_file", "count"),
                           hospitals=("source_file", "nunique"),
                           flag_frac=("flag", "mean"),
                           score_p90=("score", lambda s: s.quantile(0.90)),
                           score_mean=("score", "mean"),
                       )
                       .reset_index()
        )

        suffix = group_name.lower()
        out = out.rename(columns={
            "slices": f"slices_{suffix}",
            "hospitals": f"hospitals_{suffix}",
            "flag_frac": f"flag_frac_{suffix}",
            "score_p90": f"score_p90_{suffix}",
            "score_mean": f"score_mean_{suffix}",
        })
        return out

    gov = cpt_anoms_for_group(df, "GOV")
    comm = cpt_anoms_for_group(df, "COMM")

    if gov is None or comm is None:
        print("\n[8A] Skipped: GOV or COMM had zero rows. Check your bucket labels vs GOV_BUCKETS/COMM_BUCKETS.")
    else:
        comp = gov.merge(comm, on="code", how="inner")

        # keep CPTs with enough evidence in BOTH groups
        comp = comp[(comp["slices_gov"] >= MIN_SLICES) & (comp["slices_comm"] >= MIN_SLICES)].copy()

        comp["delta_flag_frac_gov_minus_comm"] = comp["flag_frac_gov"] - comp["flag_frac_comm"]
        comp["delta_score_p90_gov_minus_comm"] = comp["score_p90_gov"] - comp["score_p90_comm"]

        comp = comp.sort_values(["delta_score_p90_gov_minus_comm", "delta_flag_frac_gov_minus_comm"], ascending=False)

        print("\n[8A] GOV vs COMM CPT anomaly comparison (top 50; positive delta => GOV looks more anomalous):")
        print(comp.head(50))
    
# =========================
# VISUAL: BELL-CURVE DISTRIBUTION ACROSS CPT CODES
# =========================

def plot_bell_curve_for_rate(df, rate_col, label):
    """Build and save bell-curve style histogram for a given rate column."""
    if {"code", rate_col}.issubset(df.columns) is False:
        print(f"[8] SKIPPED {rate_col}: missing one of ['code', '{rate_col}']")
        return

    vis = df.copy()
    vis[rate_col] = pd.to_numeric(vis[rate_col], errors="coerce")
    vis = vis.dropna(subset=["code", rate_col])

    # One number per CPT code: median price for that rate column
    code_price = (
        vis.groupby("code")[rate_col]
           .median()
           .reset_index(name="median_gross")
    )

    # Keep positive medians only
    code_price = code_price[code_price["median_gross"] > 0]
    if code_price.empty:
        print(f"[8] SKIPPED {rate_col}: no positive medians.")
        return

    code_price["log_median_gross"] = np.log10(code_price["median_gross"])

    print(f"\n[8] Code-level median gross summary for {rate_col}:")
    print(code_price["median_gross"].describe())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw scale
    sns.histplot(
        code_price["median_gross"],
        bins=50,
        kde=True,
        ax=axes[0],
    )
    axes[0].set_title(f"Distribution of median price across CPT codes ({label})")
    axes[0].set_xlabel("Median standard charge (USD)")
    axes[0].set_ylabel("Number of CPT codes")

    # Right: log10 scale
    sns.histplot(
        code_price["log_median_gross"],
        bins=50,
        kde=True,
        ax=axes[1],
    )
    axes[1].set_title(f"log10 distribution of median price across CPT codes ({label})")
    axes[1].set_xlabel("log10(median standard charge)")
    axes[1].set_ylabel("Number of CPT codes")

    plt.tight_layout()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"cpt_price_bell_curve_{label}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[8] Saved bell-curve visualization ({label}) to: {out_path}")
    plt.close(fig)


# Run for both min and max in one go
plot_bell_curve_for_rate(df, "rate_min", "rate_min")
plot_bell_curve_for_rate(df, "rate_max", "rate_max")