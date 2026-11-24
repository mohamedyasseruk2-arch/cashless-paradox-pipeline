from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cashless import pipeline as p

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "kenya_monthly_raw.xlsx"

# Table 3.0 of the dissertation: (mean, std, min, max, median)
TABLE_3_0 = {
    "M0 Nominal (KES bn)": (229.16, 37.55, 157.70, 323.20, 223.50),
    "Real M0 (KES bn)": (206.35, 13.90, 153.17, 241.08, 205.29),
    "ln(Real M0)": (5.327, 0.069, 5.032, 5.485, 5.324),
    "Vm (fintech intensity)": (0.484, 0.077, 0.368, 0.709, 0.467),
    "M-Pesa Value (KES bn)": (477.66, 182.07, 208.13, 790.80, 421.58),
    "CPI Index (Feb 2019=100)": (111.77, 20.84, 78.37, 147.08, 108.53),
    "Inflation Rate": (0.060, 0.017, 0.027, 0.117, 0.058),
    "CBR Rate": (0.094, 0.017, 0.070, 0.130, 0.095),
}


@pytest.fixture(scope="module")
def built():
    return p.build(RAW)


@pytest.mark.parametrize("variable", TABLE_3_0)
def test_output_reproduces_dissertation_table_3_0(built, variable):
    stats = p.summary_table(built).loc[variable]
    expected = TABLE_3_0[variable]
    got = (stats["mean"], stats["std"], stats["min"], stats["max"], stats["median"])
    for e, g in zip(expected, got):
        decimals = len(str(e).split(".")[1])
        assert round(g, decimals) == pytest.approx(e, abs=10 ** -decimals), (variable, expected, got)
    assert stats["count"] == 132


def test_parse_number_handles_units_commas_and_case():
    s = pd.Series(["2,366.6b", "171.3B ", "459,277ton", " 868875", 12.5])
    assert p.parse_number(s[:2], "b").tolist() == [2366.6, 171.3]
    assert p.parse_number(s[2:], "ton").tolist() == [459277.0, 868875.0, 12.5]


def test_fix_m0_applies_documented_corrections():
    df = pd.DataFrame({"date_str": ["2025m09", "2025m10"], "m0_raw": ["292.5b", "2320.3b"]})
    out = p.fix_m0(df, corrections={"2025m10": 306.7})
    assert out["m0_nominal"].tolist() == [292.5, 306.7]


def test_fix_m0_rejects_undocumented_anomalies():
    df = pd.DataFrame({"date_str": ["2024m01"], "m0_raw": ["2,500b"]})
    with pytest.raises(ValueError, match="2024m01"):
        p.fix_m0(df, corrections={})


def test_cpi_rebase_removes_base_year_break_and_fills_gaps():
    dates = pd.to_datetime(["2019-02-01", "2019-12-01", "2020-01-01", "2020-02-01", "2020-03-01"])
    df = pd.DataFrame({"date": dates, "cpi_index": [195.78, 204.77, np.nan, 107.174, np.nan]})
    out = p.rebase_cpi(df)["cpi_index"]
    assert out[0] == pytest.approx(100.0)                      # Feb 2019 = 100 by construction
    assert out[1] == pytest.approx(204.77 * 100 / 195.78)
    assert out[2] == pytest.approx((out[1] + out[3]) / 2)     # Jan 2020 interpolated
    assert out[4] == pytest.approx(107.174)                   # trailing gap carried forward


def test_policy_rate_units_and_covid_dummy():
    df = pd.DataFrame({"date": pd.to_datetime(["2015-07-01", "2015-08-01", "2020-02-01", "2020-03-01"]),
                       "cbr_rate": [11.5, 0.115, 0.0825, 0.0725], "covid_dummy": [0, 0, 1, 1]})
    out = p.fix_rates_and_dummy(df)
    assert out["cbr_rate"].tolist() == pytest.approx([0.115, 0.115, 0.0825, 0.0725])
    assert out["covid_dummy"].tolist() == [0, 0, 0, 1]  # starts March 2020, not February


def test_denton_hits_annual_benchmarks_and_keeps_monthly_shape():
    indicator = np.array([1, 2, 3, 4, 10, 10, 20, 20], dtype=float)
    years = np.array([2020] * 4 + [2021] * 4)
    out = p.denton_proportional(indicator, {2020: 100, 2021: 600}, years)
    assert out[:4].sum() == pytest.approx(100) and out[4:].sum() == pytest.approx(600)
    assert out[:4] == pytest.approx([10, 20, 30, 40])


def test_gdp_extrapolation_uses_trailing_three_year_growth():
    annual = {2021: 100, 2022: 110, 2023: 121, 2024: 133.1}
    out = p.extrapolate_gdp(annual, through_year=2025)
    assert out[2025] == round(133.1 * 1.10)


def test_validate_catches_unrebased_cpi(built):
    broken = built.copy()
    broken.loc[broken["date"] < p.CPI_NEW_BASE_STARTS, "cpi_index"] *= p.CPI_FEB2019_OLD_BASE / 100
    annual = p.extrapolate_gdp(p.CBK_ANNUAL_GDP)
    with pytest.raises(ValueError, match="rebasing failed"):
        p.validate(broken, annual)


def test_validate_catches_missing_values(built):
    broken = built.copy()
    broken.loc[5, "real_m0"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        p.validate(broken, p.extrapolate_gdp(p.CBK_ANNUAL_GDP))
