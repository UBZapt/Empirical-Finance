"""
Empirical Finance - Task 5: Panel Data Analysis
Loads and cleans the CCM firm-year panel, runs six regression specifications
for each of two dependent variables (investment rate and next-year returns),
and exports all results to Excel.
"""

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, PooledOLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE  = Path("CCM_sample.dta")
OUTPUT_FILE = Path("cleaned_data.xlsx")

REQUIRED_COLS = ["permno", "year", "bm", "I2ppegt", "logME", "gpa", "blev", "g_sale", "ret_A"]
REGRESSORS    = ["logme", "bm", "g_sale", "blev"]

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
raw = pd.read_stata(DATA_FILE)

# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

# Standardize column names: lowercase, strip whitespace, spaces -> underscores
raw.columns = (
    raw.columns.str.strip()
    .str.lower()
    .str.replace(r"[\s\-/]+", "_", regex=True)
    .str.replace(r"[^\w]", "", regex=True)
)

# Extract integer year from the datetime column (Stata stores year dates as
# Jan 1 of that year; we only need the integer year for panel analysis)
raw["year"] = raw["year"].dt.year.astype(int)

# Cast permno to int (stored as float64 with no fractional part)
raw["permno"] = raw["permno"].astype(int)

# Trim whitespace in any string columns
str_cols = raw.select_dtypes(include="object").columns
raw[str_cols] = raw[str_cols].apply(lambda s: s.str.strip())

# Sort and construct one-period-ahead return per firm.
# ret_a_lead at (firm, t) is ret_a at (firm, t+1); the last year per firm is NaN.
raw = raw.sort_values(["permno", "year"]).reset_index(drop=True)
raw["ret_a_lead"] = raw.groupby("permno")["ret_a"].shift(-1)

# Drop exact duplicate rows (none expected)
n_before = len(raw)
raw = raw.drop_duplicates()
n_after  = len(raw)

Cleaned_data = raw.copy()

# ---------------------------------------------------------------------------
# 3. Verification tests
# ---------------------------------------------------------------------------
print("=" * 60)
print("VERIFICATION TESTS")
print("=" * 60)

print(f"\n[Row count] Before: {n_before:,}  After: {n_after:,}")
rows_ok = n_after == n_before
print(f"  Duplicates dropped: {n_before - n_after}  -> {'PASS' if rows_ok else 'WARN'}")

dup_count = Cleaned_data.duplicated().sum()
print(f"\n[Duplicates] Post-cleaning: {dup_count}  -> {'PASS' if dup_count == 0 else 'FAIL'}")

missing_cols = [c for c in [col.lower() for col in REQUIRED_COLS] if c not in Cleaned_data.columns]
print(f"\n[Required columns] Missing: {missing_cols if missing_cols else 'none'}  -> {'FAIL' if missing_cols else 'PASS'}")

type_checks = {"permno": "int", "year": "int"}
type_results = []
for col, expected in type_checks.items():
    actual = str(Cleaned_data[col].dtype)
    ok = expected in actual
    type_results.append(ok)
    print(f"  {col}: dtype={actual}  expected '{expected}' -> {'PASS' if ok else 'FAIL'}")
print(f"[Dtypes] -> {'PASS' if all(type_results) else 'FAIL'}")

mv     = Cleaned_data.isnull().sum()
mv_pct = (mv / len(Cleaned_data) * 100).round(2)
print("\n[Missing values] (flagged only — not imputed or dropped):")
for col in mv.index:
    flag = " ** NOTE" if mv[col] > 0 else ""
    print(f"  {col:15s}: {mv[col]:6,}  ({mv_pct[col]:.2f}%){flag}")

print("\n[Invalid value flags] (flagged only — not removed):")
print(f"  i2ppegt < 0 : {(Cleaned_data['i2ppegt'] < 0).sum():,} rows")
print(f"  |bm| > 10   : {(Cleaned_data['bm'].abs() > 10).sum():,} rows  (possible outliers)")
print(f"  i2ppegt > 10: {(Cleaned_data['i2ppegt'] > 10).sum():,} rows  (possible outliers)")

print(f"\n[Coverage]")
print(f"  Year range  : {Cleaned_data['year'].min()} - {Cleaned_data['year'].max()}")
print(f"  Unique firms: {Cleaned_data['permno'].nunique():,}")
print(f"  Total obs   : {len(Cleaned_data):,}")

last_obs_nan = (
    Cleaned_data.groupby("permno")["ret_a_lead"]
    .apply(lambda s: s.iloc[-1])
    .isna().all()
)
print(f"\n[ret_a_lead] Last obs per firm is NaN: {'PASS' if last_obs_nan else 'FAIL'}")

all_pass = all([rows_ok, dup_count == 0, not missing_cols, all(type_results), last_obs_nan])
print(f"\n{'=' * 60}")
print(f"OVERALL: {'PASS' if all_pass else 'FAIL (see warnings above)'}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 4. Panel Regressions
# ---------------------------------------------------------------------------

# -- Helper functions -------------------------------------------------------

def stars(pval: float) -> str:
    """Return significance stars for a two-tailed p-value."""
    if pval < 0.01: return "***"
    if pval < 0.05: return "**"
    if pval < 0.10: return "*"
    return ""


def build_reg_table(specs: list[dict]) -> pd.DataFrame:
    """
    Build a regression summary DataFrame from a list of specification dicts.

    Each dict must contain:
        res       – fitted linearmodels result object
        firm_fe   – bool, whether firm fixed effects are included
        year_fe   – bool, whether year fixed effects are included
        cluster   – str describing the clustering level (e.g. "None", "Firm")
        r2_type   – "overall" for pooled OLS, "within" for FE models

    Rows: coefficient and (t-stat) for each regressor, then R2, FE indicators,
          cluster label, and N.
    Columns: specification labels (1) through (n).

    Significance convention: * p<0.10, ** p<0.05, *** p<0.01
    """
    LABELS = {"logme": "logME", "bm": "bm", "g_sale": "g_sale", "blev": "blev"}
    col_labels = [f"({i + 1})" for i in range(len(specs))]

    rows: OrderedDict = OrderedDict()

    for var in REGRESSORS:
        label = LABELS[var]
        coef_col, tstat_col = [], []
        for s in specs:
            res = s["res"]
            if var in res.params.index:
                c = float(res.params[var])
                p = float(res.pvalues[var])
                t = float(res.tstats[var])
                coef_col.append(f"{c:.3f}{stars(p)}")
                tstat_col.append(f"({t:.3f})")
            else:
                coef_col.append("")
                tstat_col.append("")
        rows[label]         = coef_col
        rows[f"({label})"]  = tstat_col   # t-stat row labelled "(varname)"

    r2_col = []
    for s in specs:
        r2 = s["res"].rsquared if s["r2_type"] == "overall" else s["res"].rsquared_within
        r2_col.append(f"{float(r2):.3f}")
    rows["R2"]         = r2_col
    rows["Firm FE"]    = ["Yes" if s["firm_fe"] else "No"  for s in specs]
    rows["Year FE"]    = ["Yes" if s["year_fe"] else "No"  for s in specs]
    rows["Cluster SE"] = [s["cluster"]                     for s in specs]
    rows["N"]          = [f"{int(s['res'].nobs):,}"        for s in specs]

    df = pd.DataFrame(rows).T
    df.columns = col_labels
    df.index.name = None
    return df


# Set up multi-level panel index required by linearmodels
panel = Cleaned_data.set_index(["permno", "year"])

# -- Investment-rate regressions (Dependent variable: i2ppegt) --------------
# Sample: drop rows where any required column is missing
inv_sample = panel[["i2ppegt"] + REGRESSORS].dropna()
y_inv = inv_sample["i2ppegt"]
X_inv = inv_sample[REGRESSORS]

print("\nRunning investment regressions...")

# Spec 1: Pooled OLS with conventional standard errors
res_inv_1 = PooledOLS(y_inv, sm.add_constant(X_inv)).fit(cov_type="unadjusted")

# Spec 2: Firm fixed effects, conventional SEs
mod_inv_firm = PanelOLS(y_inv, X_inv, entity_effects=True, time_effects=False)
res_inv_2    = mod_inv_firm.fit(cov_type="unadjusted")

# Specs 3-6 share the same two-way FE model; only the covariance estimator changes
mod_inv_fe2 = PanelOLS(y_inv, X_inv, entity_effects=True, time_effects=True)
res_inv_3   = mod_inv_fe2.fit(cov_type="unadjusted")
res_inv_4   = mod_inv_fe2.fit(cov_type="clustered", cluster_entity=True)
res_inv_5   = mod_inv_fe2.fit(cov_type="clustered", cluster_time=True)
res_inv_6   = mod_inv_fe2.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

inv_specs = [
    {"res": res_inv_1, "firm_fe": False, "year_fe": False, "cluster": "None",      "r2_type": "overall"},
    {"res": res_inv_2, "firm_fe": True,  "year_fe": False, "cluster": "None",      "r2_type": "within"},
    {"res": res_inv_3, "firm_fe": True,  "year_fe": True,  "cluster": "None",      "r2_type": "within"},
    {"res": res_inv_4, "firm_fe": True,  "year_fe": True,  "cluster": "Firm",      "r2_type": "within"},
    {"res": res_inv_5, "firm_fe": True,  "year_fe": True,  "cluster": "Year",      "r2_type": "within"},
    {"res": res_inv_6, "firm_fe": True,  "year_fe": True,  "cluster": "Firm+Year", "r2_type": "within"},
]
inv_table = build_reg_table(inv_specs)

# -- Return regressions (Dependent variable: ret_a_lead) --------------------
ret_sample = panel[["ret_a_lead"] + REGRESSORS].dropna()
y_ret = ret_sample["ret_a_lead"]
X_ret = ret_sample[REGRESSORS]

print("Running return regressions...")

res_ret_1 = PooledOLS(y_ret, sm.add_constant(X_ret)).fit(cov_type="unadjusted")

mod_ret_firm = PanelOLS(y_ret, X_ret, entity_effects=True, time_effects=False)
res_ret_2    = mod_ret_firm.fit(cov_type="unadjusted")

mod_ret_fe2 = PanelOLS(y_ret, X_ret, entity_effects=True, time_effects=True)
res_ret_3   = mod_ret_fe2.fit(cov_type="unadjusted")
res_ret_4   = mod_ret_fe2.fit(cov_type="clustered", cluster_entity=True)
res_ret_5   = mod_ret_fe2.fit(cov_type="clustered", cluster_time=True)
res_ret_6   = mod_ret_fe2.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

ret_specs = [
    {"res": res_ret_1, "firm_fe": False, "year_fe": False, "cluster": "None",      "r2_type": "overall"},
    {"res": res_ret_2, "firm_fe": True,  "year_fe": False, "cluster": "None",      "r2_type": "within"},
    {"res": res_ret_3, "firm_fe": True,  "year_fe": True,  "cluster": "None",      "r2_type": "within"},
    {"res": res_ret_4, "firm_fe": True,  "year_fe": True,  "cluster": "Firm",      "r2_type": "within"},
    {"res": res_ret_5, "firm_fe": True,  "year_fe": True,  "cluster": "Year",      "r2_type": "within"},
    {"res": res_ret_6, "firm_fe": True,  "year_fe": True,  "cluster": "Firm+Year", "r2_type": "within"},
]
ret_table = build_reg_table(ret_specs)

# Print tables to console
print("\n" + "=" * 70)
print("TABLE 1: Investment Rate (I2ppegt) Regressions")
print("Cols: (1) Pooled OLS  (2) Firm FE  (3)-(6) Firm+Year FE")
print("Significance: * p<0.10  ** p<0.05  *** p<0.01  (t-stats in parentheses)")
print("=" * 70)
print(inv_table.to_string())

print("\n" + "=" * 70)
print("TABLE 2: Next-Year Return (ret_A_lead) Regressions")
print("Cols: (1) Pooled OLS  (2) Firm FE  (3)-(6) Firm+Year FE")
print("Significance: * p<0.10  ** p<0.05  *** p<0.01  (t-stats in parentheses)")
print("=" * 70)
print(ret_table.to_string())

# ---------------------------------------------------------------------------
# 5. Export
# ---------------------------------------------------------------------------
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    Cleaned_data.to_excel(writer, sheet_name="Cleaned_Data", index=False)
    inv_table.to_excel(writer, sheet_name="Investment_Regressions")
    ret_table.to_excel(writer, sheet_name="Returns_Regressions")

print(f"\nExported to '{OUTPUT_FILE}'")
print("  Sheets: Cleaned_Data, Investment_Regressions, Returns_Regressions")
