import sys, os

print(">>> healthcare_utilities.py STARTING")
print(">>> __file__:", __file__)
print(">>> cwd:", os.getcwd())
print(">>> python exe:", sys.executable)


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

# 1 Connect to DB  (edit credentials as needed)
engine = create_engine(
    "postgresql+psycopg2://postgres:verdansk2020!@iamr007.ddns.net:2345/hospital_db",
    future=True,
    pool_pre_ping=True,
)

print(">>> about to run pd.read_sql()")

# 2 Load base charges table into a single working DataFrame
df = pd.read_sql(
    """
    SELECT *
    FROM public.standardized_cpt_columns

    """,
    engine,
)

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
_use_rate("rate_min")
# If you want to rerun everything off max later, change that to:
# _use_rate("rate_max")

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



# # ---------- 1. BASIC STRUCTURE ----------
# # Row/column counts
# n_rows, n_cols = df.shape
# print("Rows:", n_rows, "Columns:", n_cols)

# # Column data types
# print(df.dtypes)

# # Missing values per column
# nulls = df.isna().sum().sort_values(ascending=False)
# print(nulls)


# # ---------- 2. NUMERIC SUMMARY (KEY PRICE COLS) ----------

# numeric_cols = [
#     "standard_charge_gross",
#     "standard_charge_discounted_cash",
#     "standard_charge_negotiated_dollar",
#     "standard_charge_negotiated_percentage",
#     "estimated_amount",
#     "standard_charge_min",
#     "standard_charge_max",
# ]

# # keep only those that actually exist
# numeric_cols = [c for c in numeric_cols if c in df.columns]

# if numeric_cols:
#     numeric_summary = df[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
#     print(numeric_summary)
# else:
#     print("No numeric columns found for summary.")


# # ---------- 3. TOP PAYERS / PLANS (COUNTS ONLY FOR NOW) ----------

# if "payer_name" in df.columns:
#     payer_counts = (
#         df["payer_name"]
#         .value_counts(dropna=False)
#         .head(25)
#     )
#     print("\nTop 25 payers by row count:")
#     print(payer_counts)

# if "plan_name" in df.columns:
#     plan_counts = (
#         df["plan_name"]
#         .value_counts(dropna=False)
#         .head(25)
#     )
#     print("\nTop 25 plans by row count:")
#     print(plan_counts)


# # ---------- 4. SIMPLE CHARGE STATS BY PAYER (NO BUCKETS YET) ----------

# if "payer_name" in df.columns and "standard_charge_gross" in df.columns:
#     payer_charge_summary = (
#         df.groupby("payer_name", dropna=False)["standard_charge_gross"]
#           .agg(
#               rows="count",
#               mean="mean",
#               median="median",
#               p25=lambda x: x.quantile(0.25),
#               p75=lambda x: x.quantile(0.75),
#               min="min",
#               max="max",
#           )
#           .sort_values("rows", ascending=False)
#           .head(25)
#     )
#     print("\nPayer-level summary (top 25 by rows):")
#     print(payer_charge_summary)


# # ---------- 5. BUCKET-LEVEL SUMMARY (using bucket already in standardized_cpt_columns) ----------

# if {"bucket", "standard_charge_gross"}.issubset(df.columns):
#     tmp = df.copy()

#     bucket_summary = (
#         tmp.groupby("bucket", dropna=False)["standard_charge_gross"]
#            .agg(
#                rows="count",
#                mean="mean",
#                median="median",
#                p25=lambda x: x.quantile(0.25),
#                p75=lambda x: x.quantile(0.75),
#                min="min",
#                max="max",
#            )
#            .sort_values("rows", ascending=False)
#     )
#     print("\nBucket-level summary (using df.bucket):")
#     print(bucket_summary)
        
        
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

#     # clip 1%–99% per column
#     for c in present:
#         lo, hi = _num[c].quantile([0.01, 0.99])
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

# ## =========================
# # 4) PAYER MIX & CONTRIBUTION (with value shares)
# # =========================

# TRIM_LO, TRIM_HI = 0.01, 0.99

# print("\n=== [4] Payer mix & contribution ===")

# # -----------------------------
# # 4A) Bucket mix — TRIMMED
# # -----------------------------
# # Uses ONLY standardized_cpt_columns: expects df has `bucket` and `standard_charge_gross`.

# if {"bucket", "standard_charge_gross"}.issubset(df.columns):
#     _mix = df.copy()

#     # ensure numeric + clean bucket
#     _mix["standard_charge_gross"] = pd.to_numeric(
#         _mix["standard_charge_gross"], errors="coerce"
#     )
#     _mix["bucket"] = _mix["bucket"].fillna("UNKNOWN")

#     # per-bucket trimming
#     _mix["gross_trim"] = (
#         _mix.groupby("bucket")["standard_charge_gross"]
#             .transform(lambda s: s.clip(*s.quantile([TRIM_LO, TRIM_HI])))
#     )

#     bucket_mix_df = (
#         _mix.groupby("bucket", dropna=False)["gross_trim"]
#             .agg(rows="size", val_sum="sum", median="median")
#             .reset_index()
#     )
#     bucket_mix_df["row_share"] = bucket_mix_df["rows"] / bucket_mix_df["rows"].sum()
#     bucket_mix_df["val_share"] = bucket_mix_df["val_sum"] / bucket_mix_df["val_sum"].sum()
#     bucket_mix_df = bucket_mix_df.sort_values("val_sum", ascending=False)

#     print("\n[4A] Bucket mix (TRIMMED rows/value share):")
#     print(bucket_mix_df)
# else:
#     print("\n[4A] Skipped: df is missing 'bucket' or 'standard_charge_gross'.")


# # -----------------------------
# # 4B) Top payers — TRIMMED
# # -----------------------------
# if {"payer_name", "standard_charge_gross"}.issubset(df.columns):
#     _tp = df.copy()
#     _tp["standard_charge_gross"] = pd.to_numeric(
#         _tp["standard_charge_gross"], errors="coerce"
#     )

#     # global trim for payer-level totals
#     lo, hi = _tp["standard_charge_gross"].quantile([TRIM_LO, TRIM_HI])
#     _tp["gross_trim"] = _tp["standard_charge_gross"].clip(lo, hi)

#     top_payers_rows = _tp["payer_name"].value_counts().head(25)
#     top_payers_value = (
#         _tp.groupby("payer_name", dropna=False)["gross_trim"]
#            .sum()
#            .sort_values(ascending=False)
#            .head(25)
#     )

#     print("\n[4B] Top 25 payers by rows (TRIMMED):")
#     print(top_payers_rows)
#     print("\n[4B] Top 25 payers by sum(gross) — TRIMMED:")
#     print(top_payers_value)
# else:
#     print("\n[4B] Skipped: df is missing 'payer_name' or 'standard_charge_gross'.")


# # -----------------------------
# # 4C) Bucket mix — RAW (no trim)
# # -----------------------------
# if {"bucket", "standard_charge_gross"}.issubset(df.columns):
#     _mix_raw = df.copy()
#     _mix_raw["standard_charge_gross"] = pd.to_numeric(
#         _mix_raw["standard_charge_gross"], errors="coerce"
#     )
#     _mix_raw["bucket"] = _mix_raw["bucket"].fillna("UNKNOWN")

#     bucket_mix_raw = (
#         _mix_raw.groupby("bucket", dropna=False)["standard_charge_gross"]
#                 .agg(rows="size", val_sum="sum", median="median")
#                 .reset_index()
#     )
#     bucket_mix_raw["row_share"] = bucket_mix_raw["rows"] / bucket_mix_raw["rows"].sum()
#     bucket_mix_raw["val_share"] = bucket_mix_raw["val_sum"] / bucket_mix_raw["val_sum"].sum()
#     bucket_mix_raw = bucket_mix_raw.sort_values("val_sum", ascending=False)

#     print("\n[4C] Bucket mix (RAW, untrimmed):")
#     print(bucket_mix_raw)
# else:
#     print("\n[4C] Skipped RAW bucket mix: df is missing 'bucket' or 'standard_charge_gross'.")


# # -----------------------------
# # 4D) Top payers — RAW (no trim)
# # -----------------------------
# if {"payer_name", "standard_charge_gross"}.issubset(df.columns):
#     _tp_raw = df.copy()
#     _tp_raw["standard_charge_gross"] = pd.to_numeric(
#         _tp_raw["standard_charge_gross"], errors="coerce"
#     )

#     top_payers_rows_raw = _tp_raw["payer_name"].value_counts().head(25)
#     top_payers_value_raw = (
#         _tp_raw.groupby("payer_name", dropna=False)["standard_charge_gross"]
#                .sum()
#                .sort_values(ascending=False)
#                .head(25)
#     )

#     print("\n[4D] Top 25 payers by rows (RAW):")
#     print(top_payers_rows_raw)
#     print("\n[4D] Top 25 payers by sum(gross) — RAW:")
#     print(top_payers_value_raw)
# else:
#     print("\n[4D] Skipped RAW payer lists: df is missing 'payer_name' or 'standard_charge_gross'.")


# # =========================
# # 5) VARIATION BY CPT CODE (dispersion)
# # =========================
# TRIM_LO, TRIM_HI = 0.01, 0.99    # adjust if you want tighter/looser tails
# RUN_RAW_COMPARISON_5 = False     # set True to also compute/print RAW (untrimmed)

# if {"code","standard_charge_gross"}.issubset(df.columns):
#     _v = df.copy()
#     _v["standard_charge_gross"] = pd.to_numeric(_v["standard_charge_gross"], errors="coerce")

#     # ----------------------------------------------------------------------
#     # >>> GLOBAL TRIM (comment out if you prefer PER-CODE TRIM below)
#     #lo, hi = _v["standard_charge_gross"].quantile([TRIM_LO, TRIM_HI])
#     #_v["gross_trim"] = _v["standard_charge_gross"].clip(lo, hi)
#     # ----------------------------------------------------------------------

#     # ======================================================================
#     # >>> PER-CODE TRIM <<<
#     _v["gross_trim"] = (
#         _v.groupby("code")["standard_charge_gross"]
#           .transform(lambda s: s.clip(*s.quantile([TRIM_LO, TRIM_HI])))
#     )
#     # ======================================================================

#     def q25(x): return x.quantile(0.25)
#     def q75(x): return x.quantile(0.75)

#     # ---- TRIMMED dispersion by code ----
#     code_stats = (_v.groupby("code")["gross_trim"]
#                     .agg(rows="count",
#                          vmin="min",
#                          q25=q25,
#                          median="median",
#                          q75=q75,
#                          vmax="max",
#                          mean="mean",
#                          std="std")
#                     .reset_index())

#     # optional counts
#     if "payer_name" in _v.columns:
#         code_stats["payer_count"] = _v.groupby("code")["payer_name"].nunique().values
#     if "source_file" in _v.columns:
#         code_stats["hospital_count"] = _v.groupby("code")["source_file"].nunique().values

#     # derived metrics (trimmed)
#     code_stats["cv"] = code_stats["std"] / code_stats["mean"]
#     code_stats["iqr_over_median"] = (code_stats["q75"] - code_stats["q25"]) / code_stats["median"]
#     # tail width on trimmed values (still informative)
#     code_stats["range_ratio"] = code_stats["vmax"] / code_stats["vmin"]

#     code_stats = code_stats.sort_values("iqr_over_median", ascending=False)
#     print("\n[5] Code-level dispersion — TRIMMED (head):")
#     print(code_stats.head(20))

#     # ---- Optional RAW (untrimmed) comparison ----
#     if RUN_RAW_COMPARISON_5:
#         _r = df.copy()
#         _r["standard_charge_gross"] = pd.to_numeric(_r["standard_charge_gross"], errors="coerce")

#         code_stats_raw = (_r.groupby("code")["standard_charge_gross"]
#                             .agg(rows="count",
#                                  vmin="min",
#                                  q05=lambda x: x.quantile(0.05),
#                                  q25=q25,
#                                  median="median",
#                                  q75=q75,
#                                  q95=lambda x: x.quantile(0.95),
#                                  vmax="max",
#                                  mean="mean",
#                                  std="std")
#                             .reset_index())

#         if "payer_name" in _r.columns:
#             code_stats_raw["payer_count"] = _r.groupby("code")["payer_name"].nunique().values
#         if "source_file" in _r.columns:
#             code_stats_raw["hospital_count"] = _r.groupby("code")["source_file"].nunique().values

#         code_stats_raw["cv"] = code_stats_raw["std"] / code_stats_raw["mean"]
#         code_stats_raw["iqr_over_median"] = (code_stats_raw["q75"] - code_stats_raw["q25"]) / code_stats_raw["median"]
#         code_stats_raw["p95_over_p05"] = code_stats_raw["q95"] / code_stats_raw["q05"]
#         code_stats_raw["range_ratio"] = code_stats_raw["vmax"] / code_stats_raw["vmin"]

#         code_stats_raw = code_stats_raw.sort_values("p95_over_p05", ascending=False)

#         print("\n[5] Code-level dispersion — RAW (head):")
#         print(code_stats_raw.head(20))


# # =========================
# # 6) WITHIN HOSPITAL vs CROSSHOSPITAL DISPERSION
# # =========================
# _TRIM_LO_6, _TRIM_HI_6 = 0.01, 0.99       # Winsorize tails per (hospital, code)
# _RUN_RAW_COMPARISON_6 = False             # Flip True if you want raw comparison too

# _required = {"source_file", "code", "standard_charge_gross"}
# if _required.issubset(df.columns):
#     _w = df.copy()
#     _w["standard_charge_gross"] = pd.to_numeric(_w["standard_charge_gross"], errors="coerce")

#     # Per-(hospital, code) trim to suppress crazy tails while keeping local shape
#     _w["gross_trim"] = (
#         _w.groupby(["source_file", "code"])["standard_charge_gross"]
#           .transform(lambda s: s.clip(*s.quantile([_TRIM_LO_6, _TRIM_HI_6])) if s.notna().any() else s)
#     )

#     # WITHIN-hospital dispersion for each (hospital, code)
#     within = (
#         _w.groupby(["source_file", "code"], dropna=False)["gross_trim"]
#           .agg(median="median",
#                iqr=lambda x: x.quantile(0.75) - x.quantile(0.25))
#           .reset_index()
#     )

#     # CROSS-hospital dispersion of those within medians (how hospitals differ for same code)
#     cross = (
#         within.groupby("code", dropna=False)["median"]
#               .agg(cross_std="std",
#                    cross_iqr=lambda x: x.quantile(0.75) - x.quantile(0.25))
#               .reset_index()
#     )

#     # If code_stats from Section 5 exists, show everything side-by-side
#     _have_code_stats = "code_stats" in globals()
#     within_cross = code_stats.merge(cross, on="code", how="left") if _have_code_stats else cross

#     print("\n[6] Within vs cross hospital — TRIMMED (head):")
#     print(within_cross.head(20))

#     # Finder: codes that look stable inside hospitals but vary a lot across hospitals
#     hospital_iqr = (
#         within.groupby("code", dropna=False)["iqr"]
#               .median()
#               .rename("median_within_iqr")
#               .reset_index()
#     )
#     finder = (hospital_iqr.merge(cross, on="code", how="left")
#                         .sort_values(["cross_std", "median_within_iqr"], ascending=[False, True]))

#     print("\n[6] Codes with low within-hospital IQR but high cross-hospital variance — TRIMMED (top 20):")
#     print(finder.head(20))

#     # -------- Optional RAW (untrimmed) comparison --------
#     if _RUN_RAW_COMPARISON_6:
#         _wr = df.copy()
#         _wr["standard_charge_gross"] = pd.to_numeric(_wr["standard_charge_gross"], errors="coerce")

#         within_raw = (
#             _wr.groupby(["source_file", "code"], dropna=False)["standard_charge_gross"]
#                .agg(median="median",
#                     iqr=lambda x: x.quantile(0.75) - x.quantile(0.25))
#                .reset_index()
#         )

#         cross_raw = (
#             within_raw.groupby("code", dropna=False)["median"]
#                       .agg(cross_std_raw="std",
#                            cross_iqr_raw=lambda x: x.quantile(0.75) - x.quantile(0.25))
#                       .reset_index()
#         )

#         try:
#             within_cross_raw = code_stats.merge(cross_raw, on="code", how="left")
#         except NameError:
#             within_cross_raw = cross_raw

#         print("\n[6] Within vs cross hospital — RAW (head):")
#         print(within_cross_raw.head(20))

#         finder_raw = (
#             within_raw.groupby("code", dropna=False)["iqr"]
#                       .median()
#                       .rename("median_within_iqr_raw")
#                       .reset_index()
#                       .merge(cross_raw, on="code", how="left")
#                       .sort_values(["cross_std_raw", "median_within_iqr_raw"], ascending=[False, True])
#         )
#         print("\n[6] Codes with low within-hospital IQR but high cross-hospital variance — RAW (top 20):")
#         print(finder_raw.head(20))
# else:
#     print("\n[6] Skipped: need columns", _required)

# #### some details as to why this section is the way ti is: 
# #Per-slice trim (by hospital+code) kills absurd tails without flattening real hospital differences.
# # Two dispersion layers: “within” shows internal consistency; “cross” shows how hospitals disagree
# # Finder table surfaces high cross variance + low internal variance codes → prime targets for audit.
# # Safe merges: it only merges with code_stats if it actually exists.
# # Guarded math: numeric coercion + NA handling avoids crashing on junk rows.


# # =========================
# # 7) PAYER EFFECTS (WITHIN HOSPITAL)
# # =========================
# # ==== [7] Payer effects within hospital — TRIMMED BY DEFAULT ====

# TRIM_LO, TRIM_HI = 0.01, 0.99
# RUN_RAW_COMPARISON_7 = False  # set True to also compute/print RAW (untrimmed)

# # Now we assume bucket is already present in standardized_cpt_columns.
# required_7 = {"source_file", "code", "payer_name", "standard_charge_gross", "bucket"}

# if required_7.issubset(df.columns):
#     _pe = df.copy()

#     # Coerce to numeric + clean bucket
#     _pe["standard_charge_gross"] = pd.to_numeric(_pe["standard_charge_gross"], errors="coerce")
#     _pe["bucket"] = _pe["bucket"].fillna("UNKNOWN")

#     # ----------------------------------------------------------------------
#     # TRIM CHOICE A (Recommended): PER-(HOSPITAL, CODE) TRIM
#     # -> same cutoffs across buckets inside that hospital+code slice
#     _pe["gross_trim"] = (
#         _pe.groupby(["source_file", "code"])["standard_charge_gross"]
#            .transform(lambda s: s.clip(*s.quantile([TRIM_LO, TRIM_HI])) if s.notna().any() else s)
#     )
#     # ----------------------------------------------------------------------

#     # ======================================================================
#     # TRIM CHOICE B (Alternative): PER-(HOSPITAL, CODE, BUCKET) TRIM
#     # -> each bucket gets its own cutoffs (commented out by default)
#     # _pe["gross_trim"] = (
#     #     _pe.groupby(["source_file", "code", "bucket"])["standard_charge_gross"]
#     #        .transform(lambda s: s.clip(*s.quantile([TRIM_LO, TRIM_HI])) if s.notna().any() else s)
#     # )
#     # ======================================================================

#     # ---- Median by (hospital, code, payer bucket), trimmed ----
#     payer_bucket_medians = (
#         _pe.groupby(["source_file", "code", "bucket"], dropna=False)["gross_trim"]
#            .median()
#            .reset_index(name="median_gross_trim")
#     )
#     print("\n[7] Payer-bucket medians (TRIMMED) — sample:")
#     print(payer_bucket_medians.head(20))

#     # ---- Optional: per-payer (name) medians too (trimmed) ----
#     payer_name_medians = (
#         _pe.groupby(["source_file", "code", "payer_name"], dropna=False)["gross_trim"]
#            .median()
#            .reset_index(name="median_gross_trim")
#     )

#     # ---- Deltas vs benchmarks inside each (hospital, code) ----
#     def _bench_delta(g, ref, col="median_gross_trim"):
#         ref_val = g.loc[g["bucket"] == ref, col]
#         ref_val = ref_val.iloc[0] if len(ref_val) else np.nan
#         g[f"delta_vs_{ref}"] = g[col] - ref_val
#         return g

#     for ref in ["Medicare", "Self-Pay"]:
#         if ref in payer_bucket_medians["bucket"].astype(str).unique():
#             payer_bucket_medians = (
#                 payer_bucket_medians
#                 .groupby(["source_file", "code"], as_index=False)
#                 .apply(lambda g: _bench_delta(g, ref, col="median_gross_trim"))
#                 .reset_index(drop=True)
#             )

#     print("\n[7] Payer-bucket medians with deltas (TRIMMED) — sample:")
#     print(payer_bucket_medians.head(20))

#     # ---------------- Optional RAW (untrimmed) comparison ----------------
#     if RUN_RAW_COMPARISON_7:
#         _raw = _pe.copy()
#         # use the original untrimmed column
#         payer_bucket_medians_raw = (
#             _raw.groupby(["source_file", "code", "bucket"], dropna=False)["standard_charge_gross"]
#                 .median()
#                 .reset_index(name="median_gross_raw")
#         )

#         def _bench_delta_raw(g, ref, col="median_gross_raw"):
#             ref_val = g.loc[g["bucket"] == ref, col]
#             ref_val = ref_val.iloc[0] if len(ref_val) else np.nan
#             g[f"delta_vs_{ref}_raw"] = g[col] - ref_val
#             return g

#         for ref in ["Medicare", "Self-Pay"]:
#             if ref in payer_bucket_medians_raw["bucket"].astype(str).unique():
#                 payer_bucket_medians_raw = (
#                     payer_bucket_medians_raw
#                     .groupby(["source_file", "code"], as_index=False)
#                     .apply(lambda g: _bench_delta_raw(g, ref, col="median_gross_raw"))
#                     .reset_index(drop=True)
#                 )

#         print("\n[7] Payer-bucket medians with deltas — RAW (untrimmed) — sample:")
#         print(payer_bucket_medians_raw.head(20))

# else:
#     print("\n[7] Skipped: need columns", required_7)
        
        
# ##################################
# #    OTHER TYPES OF ANALYSES    #
# ###################################

# # Detect weird pricing patterns per hospital+code using features that actually mean something
# # (not just single values).

# # ==== [8] Anomaly detection on (hospital, code) ====

# TRIM_LO, TRIM_HI = 0.01, 0.99
# USE_PER_SLICE_TRIM = True          # True: trim within each (hospital, code)
# CONTAM = 0.02                      # expected anomaly fraction (2%)

# assert {"source_file", "code", "standard_charge_gross"}.issubset(df.columns), \
#     "Need columns: source_file, code, standard_charge_gross."

# Z = df.copy()
# Z["standard_charge_gross"] = pd.to_numeric(Z["standard_charge_gross"], errors="coerce")

# # Bucket is already in standardized_cpt_columns; just normalize it.
# if "bucket" in Z.columns:
#     Z["bucket"] = Z["bucket"].fillna("UNKNOWN")
# else:
#     Z["bucket"] = "UNKNOWN"

# # ---- Trimming (winsorize) ----
# if USE_PER_SLICE_TRIM:
#     Z["gross_trim"] = (
#         Z.groupby(["source_file", "code"])["standard_charge_gross"]
#           .transform(lambda s: s.clip(*s.quantile([TRIM_LO, TRIM_HI])) if s.notna().any() else s)
#     )
# else:
#     lo, hi = Z["standard_charge_gross"].quantile([TRIM_LO, TRIM_HI])
#     Z["gross_trim"] = Z["standard_charge_gross"].clip(lo, hi)

# # ---- Build features per (hospital, code) ----
# def q(p):
#     return lambda x: x.quantile(p)

# agg = (
#     Z.groupby(["source_file", "code"])
#      .agg(
#          rows=("gross_trim", "count"),
#          med=("gross_trim", "median"),
#          q25=("gross_trim", q(0.25)),
#          q75=("gross_trim", q(0.75)),
#          p05=("gross_trim", q(0.05)),
#          p95=("gross_trim", q(0.95)),
#          vmin=("gross_trim", "min"),
#          vmax=("gross_trim", "max"),
#          mean=("gross_trim", "mean"),
#          std=("gross_trim", "std"),
#          payer_count=("payer_name", "nunique"),
#          bucket_count=("bucket", "nunique"),
#      )
#      .reset_index()
# )

# # Derived features
# agg["iqr"] = agg["q75"] - agg["q25"]
# agg["iqr_over_med"] = agg["iqr"] / agg["med"].replace(0, np.nan)
# agg["p95_over_p05"] = agg["p95"] / agg["p05"].replace(0, np.nan)
# agg["range_ratio"] = agg["vmax"] / agg["vmin"].replace(0, np.nan)
# agg["cv"] = agg["std"] / agg["mean"].replace(0, np.nan)

# # Optional: bucket mix proportions (only if buckets exist)
# if "bucket" in Z.columns:
#     mix = Z.pivot_table(
#         index=["source_file", "code"],
#         columns="bucket",
#         values="gross_trim",
#         aggfunc="count",
#         fill_value=0,
#     )
#     mix = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0).add_prefix("mix_")
#     agg = agg.merge(mix, on=["source_file", "code"], how="left")

# # ---- Train IsolationForest ----
# feature_cols = [
#     "rows", "med", "iqr", "iqr_over_med", "p95_over_p05",
#     "range_ratio", "cv", "payer_count", "bucket_count",
# ] + [c for c in agg.columns if c.startswith("mix_")]

# X = agg[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
# X = StandardScaler().fit_transform(X)

# iso = IsolationForest(n_estimators=300, contamination=CONTAM, random_state=42)
# agg["anomaly_flag"] = (iso.fit_predict(X) == -1)
# agg["anomaly_score"] = -iso.decision_function(X)  # higher = more anomalous

# # ---- Output: top anomalies ----
# anomalies = agg.sort_values(
#     ["anomaly_flag", "anomaly_score"],
#     ascending=[False, False]
# ).head(50)

# print("\n[8] Top (hospital, code) anomalies:")
# print(
#     anomalies[
#         [
#             "source_file", "code", "rows", "med",
#             "iqr_over_med", "p95_over_p05", "cv",
#             "anomaly_score", "anomaly_flag",
#         ]
#     ].head(50)
# )

# # If you want raw rows for the #1 anomaly:
# if not anomalies.empty:
#     _h, _c = anomalies.iloc[0][["source_file", "code"]]
#     suspect_rows = Z[(Z["source_file"] == _h) & (Z["code"] == _c)]
#     print(f"\nSample rows for anomaly: hospital={_h}, code={_c}")
#     print(suspect_rows.head(20))
    
    
    
    
# =========================
# 8) VISUAL: BELL-CURVE DISTRIBUTION ACROSS CPT CODES
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