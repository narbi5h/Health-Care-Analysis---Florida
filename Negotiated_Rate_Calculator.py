import numpy as np
import pandas as pd

SENTINEL = 999999999.0

def compute_rates(merge_df: pd.DataFrame, sentinel: float = SENTINEL) -> pd.DataFrame:
    """
    Compute Rate_using_max and Rate_using_min columns based on negotiated,
    percentage, and fallback logic. Modifies merge_df in place and returns it.

    Extra rules:
    1) If standard_charge_negotiated_dollar < 1 and gross is available,
       treat it as a rate (multiplier) on standard_charge_gross.
    2) If standard_charge_negotiated_dollar > standard_charge_max,
       cap it at standard_charge_max.
    """

    pct   = merge_df['standard_charge_negotiated_percentage']
    gross = merge_df['standard_charge_gross']
    est   = merge_df['estimated_amount']
    smax  = merge_df['standard_charge_max']
    smin  = merge_df['standard_charge_min']

    # Start from raw negotiated dollar
    doll = merge_df['standard_charge_negotiated_dollar'].copy()

    # 1) If negotiated "dollar" < 1, treat as a rate on gross
    #    (only when gross is available)
    mask_doll_is_rate = doll.notna() & (doll < 1) & gross.notna()
    doll.loc[mask_doll_is_rate] = gross[mask_doll_is_rate] * doll[mask_doll_is_rate]

    # 2) If negotiated "dollar" > standard_charge_max, cap at max
    mask_doll_above_max = doll.notna() & smax.notna() & (doll > smax)
    doll.loc[mask_doll_above_max] = smax[mask_doll_above_max]

    # If your percentages are like 55.0 for 55%, keep /100. If already 0.55, remove /100.
    pct_factor = pct / 100.0

    # Masks (now using cleaned doll)
    has_dollar      = doll.notna()
    has_pct_gross   = pct.notna() & gross.notna()
    has_pct_est     = pct.notna() & est.notna() & (est != 0) & (est != sentinel)
    has_pct_max     = pct.notna() & smax.notna()
    has_pct_min     = pct.notna() & smin.notna()

    no_negotiated   = doll.isna() & pct.isna()
    fallback_est    = no_negotiated & est.notna() & (est != 0) & (est != sentinel)
    fallback_gross  = no_negotiated & gross.notna()
    fallback_max    = no_negotiated & smax.notna()
    fallback_min    = no_negotiated & smin.notna()

    # Max path
    merge_df['Rate_using_max'] = np.select(
        [
            has_dollar,      # cleaned negotiated dollar
            has_pct_gross,
            has_pct_est,
            has_pct_max,
            fallback_est,
            fallback_max,
            fallback_gross,
        ],
        [
            doll,
            gross * pct_factor,
            est   * pct_factor,
            smax  * pct_factor,
            est,
            smax,
            gross,
        ],
        default=np.nan
    )

    # Min path
    merge_df['Rate_using_min'] = np.select(
        [
            has_dollar,
            has_pct_gross,
            has_pct_est,
            has_pct_min,
            fallback_est,
            fallback_min,
            fallback_gross,
        ],
        [
            doll,
            gross * pct_factor,
            est   * pct_factor,
            smin  * pct_factor,
            est,
            smin,
            gross,
        ],
        default=np.nan
    )

    # Self Pay override
    self_pay = merge_df['bucket'] == 'Self Pay'
    has_neg_dollar = doll.notna()  # use cleaned doll here as well

    self_pay_rate = np.where(
        has_neg_dollar,
        doll,  # already adjusted for <1 (rate) and >max (cap)
        merge_df['standard_charge_discounted_cash']
    )

    merge_df.loc[self_pay, 'Rate_using_max'] = self_pay_rate[self_pay]
    merge_df.loc[self_pay, 'Rate_using_min'] = self_pay_rate[self_pay]

    return merge_df
