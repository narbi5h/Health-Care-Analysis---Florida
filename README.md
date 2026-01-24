[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cN1yryt_)


# Florida Hospital Pricing Analytics Pipeline

## Project Overview
This project focuses on building an end-to-end data pipeline to standardize, clean, and analyze publicly available hospital pricing data from **254 hospitals across Florida**. The pipeline extracts heterogeneous hospital pricing files, applies a consistent schema, loads the standardized output into PostgreSQL, and prepares analytics-ready tables for downstream descriptive analysis and Power BI visualization.

---

## Project Objectives
- Consolidate and standardize public hospital pricing data from multiple sources.  
- Build a reproducible data engineering pipeline using Python and PostgreSQL.  
- Simplify CPT-code descriptions and isolate relevant CPT-based pricing records.  
- Create curated analytical tables used for descriptive analytics and visualization.  
- Enable team-wide self-service exploration through an interactive Power BI dashboard.

---

## Data Source
The raw dataset consists of hospital “Standard Charges” files published by **254 Florida hospitals**. These files vary widely in structure, naming, payer formats, and metadata—requiring custom normalization steps to consolidate them into a unified schema.

Hospital Prices can be obtained at https://hospitalpricingfiles.org/

Hospital Price Transparency Github page for additional information at https://github.com/CMSgov/hospital-price-transparency


---

## Technical Approach

## 1. Data Pipeline (postgres pipeline scripts\Pipeline Script.ipynb)
We developed a Python-based extraction and standardization pipeline that:
- Reads raw hospital pricing files (CSV, XLSX, and other formats).  
- Normalizes column names, payer groups, rate formats, code systems, and metadata.  
- Harmonizes pricing fields such as:  
  - `standard_charge_gross`  
  - negotiated rates  
  - discounted cash price  
  - estimated amount  
- Loads standardized outputs into a PostgreSQL database using SQLAlchemy.

---

## 2. PostgreSQL Database Structure

### a. Master Standardized Table
All hospitals' pricing data are consolidated into a unified table with a consistent schema.

### b. CPT-Only Table
A filtered table created where `code_type = 'CPT'`, isolating CPT-based procedures and removing other code systems (HCPCS, DRG, Rev Codes, etc.).

### c. ETL Script Summary (Main Code.py)

This script performs the end-to-end ETL process for CPT-based hospital pricing data. It securely connects to PostgreSQL, loads CPT mappings, and executes parameterized SQL queries to extract all relevant CPT records from hospital pricing, metadata, payer, and plan tables. The script cleans and standardizes numeric charge fields, applies hierarchical logic to calculate final rates, and includes special handling for Self-Pay pricing.

The ETL process also prepares a Clean Curated CPT Table, where:

Only selected CPT codes of analytical interest (e.g., Knee Replacement) are retained.

Descriptions are standardized and simplified for consistent interpretation.

Final rates and payer-specific values are aligned and imputed according to project rules.

Additionally, the script normalizes hospital addresses, extracts ZIP and ZIP+4 codes, expands multi-location hospitals into single-address rows, and produces an analytics-ready dataset. The final output is written back to PostgreSQL as standardized_cpt_columns for downstream descriptive analytics and Power BI visualization.

This final clean table is used throughout the analytics phase.

---

## Analytics & Visualization

### Descriptive Analytics (Descriptive Analytics.ipynb)
Using the clean CPT table, we performed:
- Price distribution analysis  
- Rate comparisons across hospitals and counties  
- Outlier detection  
- Exploratory insights around payer mixes and negotiated pricing variability  

### Power BI Dashboard (visuals\healthcare_pbi.pbix)
The curated CPT dataset powers an interactive Power BI report, enabling:
- Dynamic filtering by county, hospital, procedure, or payer  
- Comparative cost visualizations  
- Summary statistics and contextual insights  

Power BI deployment allows each team member to interact with the visualization and conduct **self-service analysis**, enriching our collective contextual understanding of hospital pricing patterns across Florida.

---

## Repository Structure
```text
├── dimensions/                 # Dimensions and Lookup tables
├── notebook/                   # Descriptive Analytics jupyter notebooks
├── postgres pipeline scripts/  # PostgreSQL DDL, transformations, and table creation scripts  
├── visuals/                    # Power BI assets or exported visuals  
└── README.md                   # Project documentation
