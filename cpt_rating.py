import pandas as pd
import numpy as np

def build_cpt_rating_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary dataframe by CPT code and hospital rating.

    Requires columns:
      - 'cpt_code'
      - 'county'  (if you want to count counties differently)
      - 'Hospital overall rating'
      - 'hospital_name'
      - 'bucket'  (values 'Government' / 'Commercial')
      - 'rate_using_max'
      - (optionally) 'plan_name' if you want to count plans differently
    """
    # work on a copy to avoid SettingWithCopyWarnings
    d = df.copy()

    # make sure the price column is numeric
    d["rate_using_max"] = pd.to_numeric(d["rate_using_max"], errors="coerce")

    def agg_group(g: pd.DataFrame) -> pd.Series:
        # count of negotiated plans = number of rows in this group
        count_plans = len(g)

        # distinct hospitals
        distinct_hospitals = g["hospital_name"].nunique()

        # median government price (Rate_using_max)
        govt_vals = g.loc[g["bucket"] == "Government", "rate_using_max"].dropna()
        median_govt = govt_vals.median() if not govt_vals.empty else np.nan

        # median commercial price (Rate_using_max)
        comm_vals = g.loc[g["bucket"] == "Commercial", "rate_using_max"].dropna()
        median_comm = comm_vals.median() if not comm_vals.empty else np.nan

        # commercial vs govt deviation (as a percentage)
        if pd.notna(median_govt) and median_govt != 0 and pd.notna(median_comm):
            deviation = (median_comm - median_govt) / median_govt * 100.0
        else:
            deviation = np.nan

        return pd.Series(
            {
                "Count of Negotiated Plans": count_plans,
                "Distinct Hospitals": distinct_hospitals,
                "Median Price Govt": median_govt,
                "Median Price Comm": median_comm,
                "Commercial vs Govt deviation median": deviation,
            }
        )

    summary = (
        d.groupby(["cpt_code", "county", "Hospital overall rating"], dropna=False)
        .apply(agg_group)
        .reset_index()
        .sort_values(["cpt_code", "county", "Hospital overall rating"])
    )

    return summary