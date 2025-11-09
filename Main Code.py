import pandas as pd
from sqlalchemy import create_engine, text, bindparam
import os
import numpy as np


#### Database Connection Setup



DB = "postgresql+psycopg2://postgres:verdansk2020!@iamr007.ddns.net:2345/hospital_db"
engine = create_engine(DB, pool_pre_ping=True)



### IDENTIFY CPT CODES FOR QUERIES DRIVEN BY CPT CODES CSV FILE
curr_path = os.getcwd()
cpts = pd.read_csv(f"{curr_path}/dimensions/cpt codes mapping.csv")


cpt_list = cpts['cpt_code'].astype(str).str.strip().tolist()
# if your CPT codes are numeric in the DB, you can convert them to int:
# cpt_list = [int(x) if x.isdigit() else x for x in cpt_list]

if not cpt_list:
	raise ValueError("cpt_list is empty; check the CSV path and the 'cpt_code' column.")

cpts['cpt_code'] = cpts['cpt_code'].astype(str)

### SQL QUERY

sql = text("""
select b.hospital_name, b.hospital_address, 
		   hcc.description, hcc.code, hcc.setting, hcc.modifiers, hcc.standard_charge_gross, hcc.standard_charge_discounted_cash, hcc.payer_name, hcc.plan_name, 
		   hcc.standard_charge_negotiated_dollar, standard_charge_negotiated_percentage, hcc.estimated_amount, hcc.standard_charge_min, hcc.standard_charge_max		   
from hospital_cpt_charges hcc
join hospital_metadata b
on hcc.source_file=b.source_file
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
    'standard_charge_max'
]

# Clean and convert to float safely
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


merge_df['Rate'] = np.select(
    [
        # 1️⃣ If standard_charge_negotiated_dollar is not null
        merge_df['standard_charge_negotiated_dollar'].notna(),

        # 2️⃣ If percentage + gross both exist
        merge_df['standard_charge_negotiated_percentage'].notna() & merge_df['standard_charge_gross'].notna(),

        # 3️⃣ If percentage + valid estimated_amount exist
        (merge_df['standard_charge_negotiated_percentage'].notna()) &
        (merge_df['estimated_amount'].notna()) &
        (merge_df['estimated_amount'] != 999999999.0),

        # 4️⃣ If percentage + standard_charge_max exist
        merge_df['standard_charge_negotiated_percentage'].notna() & merge_df['standard_charge_max'].notna(),
    ],
    [
        # Corresponding calculations
        merge_df['standard_charge_negotiated_dollar'],

        merge_df['standard_charge_gross'] * (merge_df['standard_charge_negotiated_percentage'] / 100),

        merge_df['estimated_amount'] * (merge_df['standard_charge_negotiated_percentage'] / 100),

        merge_df['standard_charge_max'] * (merge_df['standard_charge_negotiated_percentage'] / 100),
    ],
    default=np.nan
)


#### Check Missing Values



#### Descriptive Analytics



#### Data Visualization



#### Predictive Modeling


