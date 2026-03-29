"""
Empirical Finance - Task 5: Panel Data Analysis
Loads and cleans the CCM firm-year panel, runs six regression specifications
for each of two dependent variables (investment rate and next-year returns),
and exports all results to Excel.
"""

from pathlib import Path
import sys

import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, PooledOLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_CANDIDATES = [Path("CCM_sample.dta"), Path("CCM sample.dta")]
OUTPUT_FILE     = Path("task5_panel_results.xlsx")

# Columns that must be present after name standardization (gpa is optional)
REQUIRED_COLS = ["permno", "year", "bm", "i2ppegt", "logme", "blev", "g_sale", "ret_a"]
REGRESSORS    = ["logme", "bm", "g_sale", "blev"]


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_data(candidates: list[Path]) -> pd.DataFrame:
    """Try each candidate path in order; exit with a clear error if none exists."""
    for path in candidates:
        if path.exists():
            return pd.read_stata(path)
    tried = ", ".join(str(p) for p in candidates)
    sys.exit(f"ERROR: data file not found. Tried: {tried}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names, strip whitespace, normalize separators to underscores."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def convert_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract integer year safely.
    Uses .dt.year for datetime-typed columns (Stata stores year dates as Jan 1 of that year);
    otherwise coerces directly to numeric and validates before converting to int.
    """
    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["year"]):
        df["year"] = df["year"].dt.year.astype(int)
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        invalid = df["year"].isna().sum()
        if invalid > 0:
            sys.exit(f"ERROR: {invalid} non-numeric values in 'year' after coercion.")
        df["year"] = df["year"].astype(int)
    return df


def validate_panel(df: pd.DataFrame) -> None:
    """
    Check that (permno, year) keys are unique before any lead construction.
    Duplicate panel keys would silently corrupt the lead-return variable.
    """
    dup_keys = df.duplicated(subset=["permno", "year"]).sum()
    if dup_keys > 0:
        sys.exit(
            f"ERROR: {dup_keys} duplicate (permno, year) rows found. "
            "Resolve before constructing lead returns."
        )


def make_lead_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ret_a_lead: within-firm one-period-ahead return (shift(-1) within permno).
    The final observation per firm is NaN by construction.
    """
    df = df.sort_values(["permno", "year"]).reset_index(drop=True)
    df["ret_a_lead"] = df.groupby("permno")["ret_a"].shift(-1)
    return df


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_verification(df: pd.DataFrame, n_before: int, n_after: int) -> None:
    """Run required verification tests and print a PASS/FAIL summary."""
    print("=" * 60)
    print("VERIFICATION TESTS")
    print("=" * 60)

    results: list[bool] = []

    # Row count
    rows_ok = n_after == n_before
    results.append(rows_ok)
    print(
        f"[Row count]        Before: {n_before:,}  After: {n_after:,}  "
        f"(dropped: {n_before - n_after})  -> {'PASS' if rows_ok else 'WARN'}"
    )

    # Exact duplicates
    dup_count = df.duplicated().sum()
    results.append(dup_count == 0)
    print(f"[Duplicates]       Post-cleaning: {dup_count}  -> {'PASS' if dup_count == 0 else 'FAIL'}")

    # Panel key uniqueness
    dup_keys = df.duplicated(subset=["permno", "year"]).sum()
    results.append(dup_keys == 0)
    print(f"[Panel keys]       Duplicate (permno, year): {dup_keys}  -> {'PASS' if dup_keys == 0 else 'FAIL'}")

    # Required columns
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    results.append(not missing_cols)
    print(
        f"[Required cols]    Missing: {missing_cols if missing_cols else 'none'}  "
        f"-> {'FAIL' if missing_cols else 'PASS'}"
    )

    # Dtype checks
    type_ok = ("int" in str(df["permno"].dtype)) and ("int" in str(df["year"].dtype))
    results.append(type_ok)
    print(
        f"[Dtypes]           permno={df['permno'].dtype}  year={df['year'].dtype}  "
        f"-> {'PASS' if type_ok else 'FAIL'}"
    )

    # All regression variables must be numeric
    non_numeric = [
        c for c in REGRESSORS + ["i2ppegt", "ret_a"]
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    results.append(not non_numeric)
    print(
        f"[Numeric vars]     Non-numeric regression cols: "
        f"{non_numeric if non_numeric else 'none'}  -> {'FAIL' if non_numeric else 'PASS'}"
    )

    # Missing-value summary (flagged only — not dropped or imputed)
    check_cols = REQUIRED_COLS + ["ret_a_lead"]
    mv = df[check_cols].isnull().sum()
    print("[Missing values]   (flagged — not imputed or dropped):")
    for col, cnt in mv.items():
        note = " **" if cnt > 0 else ""
        print(f"  {col:15s}: {cnt:,}{note}")

    # Lead-return: last obs per firm must be NaN
    last_obs_nan = df.groupby("permno")["ret_a_lead"].apply(lambda s: s.iloc[-1]).isna().all()
    results.append(last_obs_nan)
    print(f"[ret_a_lead]       Last obs per firm is NaN: {'PASS' if last_obs_nan else 'FAIL'}")

    # Invalid value flags (flagged only)
    print("[Invalid flags]    (flagged — not removed):")
    print(f"  i2ppegt < 0 : {(df['i2ppegt'] < 0).sum():,}")
    print(f"  |bm| > 10   : {(df['bm'].abs() > 10).sum():,}")
    print(f"  i2ppegt > 10: {(df['i2ppegt'] > 10).sum():,}")

    # Coverage summary
    print(
        f"[Coverage]         Year range: {df['year'].min()}–{df['year'].max()}  |  "
        f"Firms: {df['permno'].nunique():,}  |  Obs: {len(df):,}"
    )

    overall = all(results)
    print(f"\n{'=' * 60}")
    print(f"OVERALL: {'PASS' if overall else 'FAIL (see above)'}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def stars(pval: float) -> str:
    """Return significance stars for a two-tailed p-value."""
    if pval < 0.01: return "***"
    if pval < 0.05: return "**"
    if pval < 0.10: return "*"
    return ""


def build_reg_table(specs: list[dict]) -> pd.DataFrame:
    """
    Build a regression summary table from a list of specification dicts.

    Each dict must have:
        res       – fitted linearmodels result
        firm_fe   – bool
        year_fe   – bool
        cluster   – str label (e.g. "None", "Firm", "Firm+Year")
        r2_type   – "overall" (pooled OLS) or "within" (FE models)

    Rows: coefficient + (t-stat) for each regressor, R2, FE indicators,
          cluster label, N.  Columns: spec labels (1)–(n).
    Significance: * p<0.10  ** p<0.05  *** p<0.01
    """
    LABELS    = {"logme": "logME", "bm": "bm", "g_sale": "g_sale", "blev": "blev"}
    col_labels = [f"({i + 1})" for i in range(len(specs))]
    rows: dict = {}

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
        rows[label]        = coef_col
        rows[f"({label})"] = tstat_col

    r2_col = [
        f"{float(s['res'].rsquared if s['r2_type'] == 'overall' else s['res'].rsquared_within):.3f}"
        for s in specs
    ]
    rows["R2"]         = r2_col
    rows["Firm FE"]    = ["Yes" if s["firm_fe"] else "No" for s in specs]
    rows["Year FE"]    = ["Yes" if s["year_fe"] else "No" for s in specs]
    rows["Cluster SE"] = [s["cluster"]                    for s in specs]
    rows["N"]          = [f"{int(s['res'].nobs):,}"       for s in specs]

    df = pd.DataFrame(rows).T
    df.columns    = col_labels
    df.index.name = None
    return df


def run_regression_family(
    panel: pd.DataFrame, dep_var: str
) -> tuple[pd.DataFrame, int, int]:
    """
    Run the six required specifications for dep_var against REGRESSORS.
    Prints sample size and rows dropped for missing values.
    Returns (result table, N used, N dropped).
    """
    sample   = panel[[dep_var] + REGRESSORS].dropna()
    y        = sample[dep_var]
    X        = sample[REGRESSORS]
    n_used   = len(sample)
    n_dropped = len(panel) - n_used

    print(f"  {dep_var}: {n_used:,} obs used  ({n_dropped:,} dropped for missing values)")

    # Spec 1: Pooled OLS, conventional SEs
    res_1 = PooledOLS(y, sm.add_constant(X)).fit(cov_type="unadjusted")

    # Spec 2: Firm FE, conventional SEs
    mod_firm = PanelOLS(y, X, entity_effects=True, time_effects=False)
    res_2    = mod_firm.fit(cov_type="unadjusted")

    # Specs 3–6: Firm+Year FE; only covariance estimator varies
    mod_fe2 = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res_3   = mod_fe2.fit(cov_type="unadjusted")
    res_4   = mod_fe2.fit(cov_type="clustered", cluster_entity=True)
    res_5   = mod_fe2.fit(cov_type="clustered", cluster_time=True)
    res_6   = mod_fe2.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    specs = [
        {"res": res_1, "firm_fe": False, "year_fe": False, "cluster": "None",      "r2_type": "overall"},
        {"res": res_2, "firm_fe": True,  "year_fe": False, "cluster": "None",      "r2_type": "within"},
        {"res": res_3, "firm_fe": True,  "year_fe": True,  "cluster": "None",      "r2_type": "within"},
        {"res": res_4, "firm_fe": True,  "year_fe": True,  "cluster": "Firm",      "r2_type": "within"},
        {"res": res_5, "firm_fe": True,  "year_fe": True,  "cluster": "Year",      "r2_type": "within"},
        {"res": res_6, "firm_fe": True,  "year_fe": True,  "cluster": "Firm+Year", "r2_type": "within"},
    ]
    return build_reg_table(specs), n_used, n_dropped


def export_results(
    data: pd.DataFrame,
    inv_table: pd.DataFrame,
    ret_table: pd.DataFrame,
    meta: dict,
    output_path: Path,
) -> None:
    """Export cleaned data, both regression tables, and a metadata sheet to Excel."""
    INV_TITLE = (
        "Table 1: Investment Rate (I2ppegt) Regressions — "
        "(1) Pooled OLS  (2) Firm FE  (3)–(6) Firm+Year FE  |  "
        "* p<0.10  ** p<0.05  *** p<0.01  |  t-stats in parentheses  |  "
        "Within-R² for FE models; overall R² for Pooled OLS"
    )
    RET_TITLE = (
        "Table 2: Next-Year Return (ret_A_lead) Regressions — "
        "(1) Pooled OLS  (2) Firm FE  (3)–(6) Firm+Year FE  |  "
        "* p<0.10  ** p<0.05  *** p<0.01  |  t-stats in parentheses  |  "
        "Within-R² for FE models; overall R² for Pooled OLS"
    )
    meta_df = pd.DataFrame(list(meta.items()), columns=["Item", "Value"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Cleaned_Data", index=False)

        inv_table.to_excel(writer, sheet_name="Investment_Regressions", startrow=2)
        writer.sheets["Investment_Regressions"].cell(row=1, column=1).value = INV_TITLE

        ret_table.to_excel(writer, sheet_name="Returns_Regressions", startrow=2)
        writer.sheets["Returns_Regressions"].cell(row=1, column=1).value = RET_TITLE

        meta_df.to_excel(writer, sheet_name="Metadata", index=False)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

# 1. Load raw data (tries CCM_sample.dta then CCM sample.dta)
raw = load_data(DATA_CANDIDATES)

# 2. Standardize column names to lowercase with underscores
raw = standardize_columns(raw)

# 3. Convert year to integer (handles both datetime and numeric imports)
raw = convert_year(raw)

# 4. Cast permno to int (Stata stores it as float64 with no fractional part)
raw["permno"] = raw["permno"].astype(int)

# 5. Trim whitespace in any string columns
str_cols = raw.select_dtypes(include="object").columns
raw[str_cols] = raw[str_cols].apply(lambda s: s.str.strip())

# 6. Drop exact duplicate rows before panel validation
n_before = len(raw)
raw = raw.drop_duplicates()
n_after  = len(raw)

# 7. Validate panel keys — must be unique before constructing lead return
validate_panel(raw)

# 8. Construct one-period-ahead return within each firm
raw = make_lead_return(raw)

# 9. Run verification tests
run_verification(raw, n_before, n_after)

# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------
print("\nRunning regressions...")
panel = raw.set_index(["permno", "year"])

inv_table, inv_n, inv_dropped = run_regression_family(panel, "i2ppegt")
ret_table, ret_n, ret_dropped = run_regression_family(panel, "ret_a_lead")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
data_file_used = next((str(p) for p in DATA_CANDIDATES if p.exists()), "unknown")
meta = {
    "Dataset":                          data_file_used,
    "Total rows after cleaning":        f"{n_after:,}",
    "Exact duplicates dropped":         f"{n_before - n_after:,}",
    "Year range":                       f"{raw['year'].min()}–{raw['year'].max()}",
    "Unique firms":                     f"{raw['permno'].nunique():,}",
    "Investment sample (N)":            f"{inv_n:,}",
    "Investment rows dropped (missing)":f"{inv_dropped:,}",
    "Returns sample (N)":               f"{ret_n:,}",
    "Returns rows dropped (missing)":   f"{ret_dropped:,}",
}

export_results(raw, inv_table, ret_table, meta, OUTPUT_FILE)
print(f"\nExported to '{OUTPUT_FILE}'")
print("  Sheets: Cleaned_Data, Investment_Regressions, Returns_Regressions, Metadata")
