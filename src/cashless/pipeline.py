"""Clean and transform the raw Kenyan monthly series into the dissertation's analysis dataset.

Raw input: one sheet, 132 monthly rows (2015m01 to 2025m12), assembled by hand from
Central Bank of Kenya and KNBS releases. It has the usual problems of hand-assembled
data: numbers stored as text with units ("2,366.6b", "459,277ton"), a CPI series that
switches base year in 2020, a policy rate entered as percentages for some months and
decimals for others, a mis-dated dummy, missing months, and three mistyped values.

Every fix is a separate function, so each one can be tested on its own, and
``validate`` stops the run if the output breaks an expected invariant.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = {
    "date": "date_str",
    "Currency in Circulation": "m0_raw",
    "mpesa_value": "mpesa_value",
    "cement_consumption": "cement_raw",
    "consumer price index": "cpi_index",
    "infl_rate": "infl_rate",
    "cbr_rate": "cbr_rate",
    "covid_dummy": "covid_dummy",
}

# Oct-Dec 2025 M0 was typed roughly 10x too large (e.g. "2,366.6b"). Values checked by hand
# against the CBK release. Any OTHER month that looks anomalous makes the pipeline fail.
M0_CORRECTIONS = {"2025m10": 306.7, "2025m11": 312.2, "2025m12": 323.2}
M0_PLAUSIBLE_MAX = 500.0  # KES bn; real values are 150-330 over the sample

# Feb 2019 CPI on the OLD base (Feb 2009 = 100). Used to splice the two series.
CPI_FEB2019_OLD_BASE = 195.780
CPI_NEW_BASE_STARTS = pd.Timestamp("2020-01-01")

COVID_START = pd.Timestamp("2020-03-01")

# Annual nominal GDP, KES million, current prices (Central Bank of Kenya).
CBK_ANNUAL_GDP = {
    2015: 6_884_317, 2016: 7_594_064, 2017: 8_483_396, 2018: 9_340_307,
    2019: 10_237_727, 2020: 10_715_070, 2021: 12_027_662, 2022: 13_489_642,
    2023: 15_033_612, 2024: 16_224_477,
}


def parse_number(series, unit):
    """'2,366.6b' -> 2366.6 ; '459,277ton' -> 459277.0. Numbers already numeric pass through."""
    return (series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(unit, "", case=False, regex=False)
            .str.strip()
            .astype(float))


def load_raw(path):
    raw = pd.read_excel(path)
    df = raw[list(RAW_COLUMNS)].rename(columns=RAW_COLUMNS)
    df["date"] = pd.to_datetime(df["date_str"].str.replace("m", "-", regex=False), format="%Y-%m")
    return df.sort_values("date").reset_index(drop=True)


def fix_m0(df, corrections=M0_CORRECTIONS):
    df = df.copy()
    df["m0_nominal"] = parse_number(df["m0_raw"], "b")
    flagged = set(df.loc[df["m0_nominal"] > M0_PLAUSIBLE_MAX, "date_str"])
    unexplained = flagged - set(corrections)
    if unexplained:
        raise ValueError(f"Implausible M0 values with no documented correction: {sorted(unexplained)}")
    for period, value in corrections.items():
        df.loc[df["date_str"] == period, "m0_nominal"] = value
    return df


def fix_cement(df):
    df = df.copy()
    df["cement_cons"] = parse_number(df["cement_raw"], "ton")
    return df


def rebase_cpi(df):
    """Splice the 2009-base series onto the 2019 base, then fill the two gaps.

    - 2015-2019 values are multiplied by 100 / CPI(Feb 2019, old base).
    - Jan 2020 is missing (transition month): linear interpolation between Dec 2019 and Feb 2020.
    - Dec 2025 is missing (not yet released): carried forward from Nov 2025.
    """
    df = df.copy()
    old_base = df["date"] < CPI_NEW_BASE_STARTS
    df.loc[old_base, "cpi_index"] *= 100 / CPI_FEB2019_OLD_BASE
    df["cpi_index"] = df["cpi_index"].interpolate(method="linear", limit_area="inside")
    df["cpi_index"] = df["cpi_index"].ffill()
    return df


def fix_rates_and_dummy(df):
    df = df.copy()
    as_percent = df["cbr_rate"] > 1  # Jan-Jul 2015 typed as 8.5 rather than 0.085
    df.loc[as_percent, "cbr_rate"] /= 100
    df["covid_dummy"] = (df["date"] >= COVID_START).astype(int)  # raw sheet started it in Feb 2020
    return df


def extrapolate_gdp(annual, through_year=2025, window=3):
    """Extend annual GDP with the trailing average growth rate (flagged as an estimate)."""
    annual = dict(annual)
    while max(annual) < through_year:
        years = sorted(annual)
        growth = [annual[y] / annual[p] - 1 for p, y in zip(years[:-1], years[1:])]
        annual[years[-1] + 1] = round(annual[years[-1]] * (1 + np.mean(growth[-window:])))
    return annual


def denton_proportional(indicator, annual, years, max_iter=300, tol=1e-4):
    """Distribute annual totals across months in proportion to a monthly indicator.

    Months get the indicator's within-year share, then each year is rescaled until it
    sums exactly to its benchmark.
    """
    indicator = np.asarray(indicator, dtype=float)
    years = np.asarray(years)
    out = np.zeros_like(indicator)
    for year, total in annual.items():
        m = years == year
        out[m] = total * indicator[m] / indicator[m].sum()
    for _ in range(max_iter):
        prev = out.copy()
        for year, total in annual.items():
            m = years == year
            out[m] *= total / out[m].sum()
        if np.max(np.abs(out - prev)) < tol:
            break
    return out


def derive(df, annual_gdp=None):
    df = df.copy()
    annual = extrapolate_gdp(annual_gdp or CBK_ANNUAL_GDP)
    df["year"] = df["date"].dt.year
    df["real_m0"] = df["m0_nominal"] / (df["cpi_index"] / 100)
    df["monthly_gdp_mn"] = denton_proportional(df["cement_cons"], annual, df["year"])
    df["monthly_gdp_bn"] = df["monthly_gdp_mn"] / 1000
    df["Vm"] = df["mpesa_value"] / df["monthly_gdp_bn"]  # mobile-money intensity
    for col, src in [("ln_m0", "real_m0"), ("ln_mpesa", "mpesa_value"),
                     ("ln_cement", "cement_cons"), ("ln_Vm", "Vm")]:
        df[col] = np.log(df[src])
    return df, annual


OUTPUT_COLUMNS = ["date", "date_str", "m0_nominal", "real_m0", "ln_m0", "mpesa_value", "ln_mpesa",
                  "cement_cons", "ln_cement", "cpi_index", "infl_rate", "cbr_rate", "covid_dummy",
                  "year", "monthly_gdp_mn", "monthly_gdp_bn", "Vm", "ln_Vm"]


def validate(df, annual):
    """Raise if the output breaks any invariant the analysis depends on."""
    problems = []
    if len(df) != 132 or df["date"].diff().dropna().dt.days.gt(31).any():
        problems.append("expected 132 consecutive months")
    nans = df[OUTPUT_COLUMNS].isna().sum()
    if nans.any():
        problems.append(f"missing values: {nans[nans > 0].to_dict()}")
    for year, total in annual.items():
        got = df.loc[df["year"] == year, "monthly_gdp_mn"].sum()
        if abs(got / total - 1) > 1e-9:
            problems.append(f"monthly GDP for {year} does not sum to the annual benchmark")
    if not df["cbr_rate"].between(0, 0.3).all() or not df["infl_rate"].between(0, 0.3).all():
        problems.append("rates must be decimals between 0 and 0.3")
    if not df["m0_nominal"].between(100, M0_PLAUSIBLE_MAX).all():
        problems.append("nominal M0 outside the plausible range")
    # Splice check: the CPI must not jump at the base change (raw data falls ~48% there)
    jump = df.set_index("date")["cpi_index"].pct_change().abs().max()
    if jump > 0.05:
        problems.append(f"CPI has a {jump:.0%} month-on-month jump; rebasing failed")
    if problems:
        raise ValueError("; ".join(problems))
    return df


def build(raw_path):
    df = load_raw(raw_path)
    for step in (fix_m0, fix_cement, rebase_cpi, fix_rates_and_dummy):
        df = step(df)
    df, annual = derive(df)
    return validate(df[OUTPUT_COLUMNS + []], annual)


def summary_table(df):
    cols = {"m0_nominal": "M0 Nominal (KES bn)", "real_m0": "Real M0 (KES bn)", "ln_m0": "ln(Real M0)",
            "Vm": "Vm (fintech intensity)", "mpesa_value": "M-Pesa Value (KES bn)",
            "cpi_index": "CPI Index (Feb 2019=100)", "infl_rate": "Inflation Rate", "cbr_rate": "CBR Rate"}
    s = df[list(cols)].agg(["count", "mean", "std", "min", "max", "median"]).T
    return s.rename(index=cols)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    out = build(root / "data" / "raw" / "kenya_monthly_raw.xlsx")
    dest = root / "data" / "processed" / "cashless_paradox_analysis.csv"
    out.to_csv(dest, index=False)
    print(f"wrote {len(out)} rows to {dest.relative_to(root)}\n")
    print(summary_table(out).round(3).to_string())
