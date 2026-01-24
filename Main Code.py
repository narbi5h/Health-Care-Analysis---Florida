import pandas as pd
from sqlalchemy import create_engine, text, bindparam
from dotenv import load_dotenv
import os
import numpy as np
import re
# import payer_plan_distinct_pull
# import full_standardization
import Negotiated_Rate_Calculator as nrc # import the module with compute_rates function

#### Database Connection Setup
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DB_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DB_URL, pool_pre_ping=True)
print("[INFO] Database engine created.")

### IDENTIFY CPT CODES FOR QUERIES DRIVEN BY CPT CODES CSV FILE
curr_path = os.getcwd()
cpts = pd.read_csv(f"{curr_path}/dimensions/cpt codes mapping.csv")


cpt_list = cpts['cpt_code'].astype(str).str.strip().tolist()
# if your CPT codes are numeric in the DB, you can convert them to int:
# cpt_list = [int(x) if x.isdigit() else x for x in cpt_list]

if not cpt_list:
	raise ValueError("cpt_list is empty; check the CSV path and the 'cpt_code' column.")

cpts['cpt_code'] = cpts['cpt_code'].astype(str)

###### Add Bucketing and Specification Columns to Payer/Plan Standardization Pipeline, consider moving to separate script
# #### payer/plan export and standardization runs
# ### Run payer/plan distinct export
# if os.getenv("RUN_DISTINCT_PULL", "false").lower() == "true":
#     print("\n[INFO] Running payer_plan_distinct_pull script...")
#     # Call the script as a module; it will execute its own logic
#     payer_plan_distinct_pull.main()
# ## 2 CSVs will be created in the current directory:
# #   - distinct_payers.csv
# #   - distinct_plans.csv
# ## need to manually bucket and spec these files before running full_standardization
# ### Run full standardization / bucket-spec pipeline
# ## If you want to run to write in DB, then set the variable in "full_standardization.py" to "DRY_RUN = false"
# ## If there are no bucket/spec columns in DB, set the variable in "full_standardization.py" to "ADD_BUCKET_SPEC_COLUMNS", "true"
# # Default: DRY_RUN = True, "ADD_BUCKET_SPEC_COLUMNS", "false"
# if os.getenv("RUN_STANDARDIZATION", "false").lower() == "true":
#     print("\n[INFO] Running full_standardization script...")
#     full_standardization.main()

### SQL QUERY

sql = text("""
select b.hospital_name, b.hospital_address, 
		   hcc.description, hcc.code, hcc.setting, hcc.modifiers, hcc.standard_charge_gross, hcc.standard_charge_discounted_cash, hcc.payer_name, hcc.plan_name, 
		   hcc.standard_charge_negotiated_dollar, standard_charge_negotiated_percentage, hcc.estimated_amount, hcc.standard_charge_min, hcc.standard_charge_max,
           COALESCE(ps.bucket, ys.bucket) as bucket,
           COALESCE(ps.specification, ys.specification) as specification_plan,
           ys.specification as specification_payer
           		   
from hospital_cpt_charges hcc
join hospital_metadata b
    ON hcc.source_file = b.source_file
left join plan_specs ps ON ps.name = hcc.plan_name
left join payer_specs ys ON ys.name = hcc.payer_name
WHERE hcc.code IN :cpts
""").bindparams(bindparam("cpts", expanding=True))


df = pd.read_sql_query(sql, engine, params={"cpts": cpt_list})

merge_df = pd.merge(df, cpts, left_on='code', right_on='cpt_code', how='left')


### Update Charge Columns to Numeric
cols_to_float = [
    'standard_charge_gross',
    'standard_charge_negotiated_dollar',
    'standard_charge_negotiated_percentage',
    'estimated_amount',
    'standard_charge_min',
    'standard_charge_max',
    'standard_charge_discounted_cash'
]


# Clean and convert to float safely
print("[INFO] Converting columns to float...")
for col in cols_to_float:
    merge_df[col] = (
        merge_df[col]
        .astype(str)                           # ensure string type
        .str.replace(',', '', regex=False)     # remove commas like "1,000"
        .str.replace('$', '', regex=False)     # remove dollar signs if any
        .replace(['', 'None', 'nan', 'NaN'], np.nan)  # treat blanks as NaN
        .astype(float)
    )

#### Determine Rate Amount of the Procedures

merge_df = nrc.compute_rates(merge_df)



#### Format Hospital Address ZIP Codes
print("[INFO] Formatting hospital address ZIP codes...")
merge_df['hospital_address'] = merge_df['hospital_address'].str.replace(
    r'(\b\d{5})(\d{4}\b)',
    r'\1-\2',
    regex=True
)

#### SPLIT MULTI-ADDRESS ROWS INTO MULTIPLE ROWS

# make a list of addresses per row (split on | with or without spaces) ---
print("[INFO] Exploding multi-address rows into single-address rows...")
merge_df = merge_df.copy()

merge_df['address_list'] = (
    merge_df['hospital_address']
      .fillna('')
      .apply(lambda x: [a.strip() for a in re.split(r'\s*\|\s*', str(x)) if a.strip()])
)

# how many locations were in that row
merge_df['num_locations'] = merge_df['address_list'].apply(len)

# --- 2) explode to one row per address ---
exploded = (
    merge_df
      .explode('address_list', ignore_index=True)
      .rename(columns={'address_list': 'hospital_address_single'})
)

# flag whether the price was system-applied (multiple addresses) or facility-specific
exploded['price_scope'] = np.where(exploded['num_locations'] > 1, 'system_applied', 'facility_specific')

# --- 3) improved ZIP extraction (take LAST ZIP, not first) ---
# find all 5-digit or ZIP+4 patterns
zip_candidates = exploded['hospital_address_single'].str.findall(r'\b\d{5}(?:-\d{4})?\b')

# ZIP4 = last candidate in each address, if any exist
exploded['ZIP4'] = zip_candidates.apply(lambda xs: xs[-1] if isinstance(xs, list) and len(xs) else pd.NA)

# ZIP = first 5 digits of ZIP4
exploded['ZIP'] = exploded['ZIP4'].astype('string').str.slice(0, 5)

# ensure text type (important for leading zeros)
exploded['ZIP']  = exploded['ZIP'].astype('string')
exploded['ZIP4'] = exploded['ZIP4'].astype('string')

# --- 4) (optional) tidy columns/order ---
print("[INFO] Finalizing exploded DataFrame...")
cols_front = ['hospital_name', 'hospital_address_single', 'ZIP', 'ZIP4', 'num_locations', 'price_scope']
other_cols = [c for c in exploded.columns if c not in cols_front]
exploded = exploded[cols_front + other_cols]

### Update inconsistent Bucket and Specification
exploded['specification'] = np.where(
    exploded['bucket'].eq('Government'),
    exploded['specification_plan'],
    exploded['specification_payer']
)

cols_keep = [
    'hospital_name', 'hospital_address_single', 'ZIP4', 'setting', 'modifiers', 'standard_charge_gross',
    'standard_charge_discounted_cash', 'payer_name', 'plan_name',
    'standard_charge_negotiated_dollar', 'standard_charge_negotiated_percentage', 
    'standard_charge_min', 'standard_charge_max', 'bucket', 'specification','specification_plan', 'cpt_code', 'Description', 'Simple Description',
    'Specialty', 'Category', 'Rate_using_min', 'Rate_using_max'
]

missing = [c for c in cols_keep if c not in exploded.columns]

exploded = exploded[cols_keep].copy()


# # write rows to CSV
# exploded.to_csv(f"{curr_path}/exploded_cpt_data.csv", index=False)
# print("[OK] Wrote exploded CPT data to 'exploded_cpt_data.csv'")

# Create a new table in the database to store the exploded CPT data
exploded.to_sql('standardized_cpt_columns', engine, if_exists='replace', index=True)
print("[OK] Wrote exploded CPT data to 'standardized_cpt_columns' table in the database.")




