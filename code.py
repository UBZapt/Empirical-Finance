"""
Empirical Finance - Task 5: Panel Data Analysis
Loads and cleans the CCM firm-year panel, runs six regression specifications
for each of two dependent variables (investment rate and next-year returns),
displays results in the terminal, and exports all results to Excel.

Regression engine: pyfixest feols(), replicating Stata's reghdfe.
"""

from pathlib import Path
import sys
import warnings

# Had an error with the file being named code.py, so sys.path[0] not in ("",) was added.
if sys.path and sys.path[0] not in ("",):
    sys.path.pop(0)

import numpy as np
import pandas as pd
from pyfixest.estimation import feols

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_CANDIDATES     = [Path("CCM_sample.dta"), Path("CCM sample.dta")]
OUTPUT_FILE         = Path("task5_panel_results.xlsx")

REQUIRED_COLS = ["permno", "year", "bm", "i2ppegt", "logme", "blev", "g_sale", "ret_a"]
REGRESSORS    = ["logme", "bm", "g_sale", "blev"]
REG_LABELS    = {"logme": "logME", "bm": "bm", "g_sale": "g_sale", "blev": "blev"}

# Display metadata for each of the six specifications.
# Only contains keys consumed by build_reg_table; actual formulas/vcov are
# in run_regression_family.
SPEC_META: list[dict] = [
    {"firm_fe": False, "year_fe": False, "cluster": "None",      "r2_type": "overall"},
    {"firm_fe": True,  "year_fe": False, "cluster": "None",      "r2_type": "within"},
    {"firm_fe": True,  "year_fe": True,  "cluster": "None",      "r2_type": "within"},
    {"firm_fe": True,  "year_fe": True,  "cluster": "Firm",      "r2_type": "within"},
    {"firm_fe": True,  "year_fe": True,  "cluster": "Year",      "r2_type": "within"},
    {"firm_fe": True,  "year_fe": True,  "cluster": "Firm+Year", "r2_type": "within"},
]

# Suppress pyfixest's singleton-FE UserWarning — these are expected in unbalanced panels and do not affect coefficient estimates or standard errors.
warnings.filterwarnings(
    "ignore",
    message=".*singleton fixed effect.*",
    category=UserWarning,
)

SEP = "-" * 80


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_data(candidates: list[Path]) -> tuple[pd.DataFrame, Path]:
    """Try each possible data file name; exit with a clear error if none exists."""
    for path in candidates:
        if path.exists():
            return pd.read_stata(path), path
    tried = ", ".join(str(p) for p in candidates)
    print(f"ERROR  data file not found. Tried: {tried}")
    sys.exit(1)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names, strip whitespace, normalize separators to underscores."""
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
    Uses .dt.year for datetime-typed columns;
    otherwise coerces to numeric and validates before converting to int.
    """
    if pd.api.types.is_datetime64_any_dtype(df["year"]):
        df["year"] = df["year"].dt.year.astype(int)
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        invalid = df["year"].isna().sum()
        if invalid > 0:
            print(f"ERROR  {invalid} non-numeric values in 'year' after coercion.")
            sys.exit(1)
        df["year"] = df["year"].astype(int)
    return df


def validate_and_coerce_permno(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate permno: coerce to numeric, check for missing/non-numeric and
    fractional values, then convert to int.
    """
    numeric = pd.to_numeric(df["permno"], errors="coerce")
    n_invalid = numeric.isna().sum()
    if n_invalid > 0:
        print(f"ERROR  {n_invalid} missing or non-numeric values in 'permno'.")
        sys.exit(1)
    fractional = int((numeric != numeric.round()).sum())
    if fractional > 0:
        print(f"ERROR  {fractional} non-integer values in 'permno' (fractional parts detected).")
        sys.exit(1)
    df["permno"] = numeric.astype(int)
    return df


def validate_panel(df: pd.DataFrame) -> None:
    """
    Check that (permno, year) keys are unique before ret a lead construction.
    Called after drop_duplicates() — any remaining duplicates here are non-identical
    rows sharing the same panel key and cannot be resolved by deduplication.
    """
    dup_keys = df.duplicated(subset=["permno", "year"]).sum()
    if dup_keys > 0:
        print(
            f"ERROR  {dup_keys} duplicate (permno, year) rows remain after "
            "deduplication. Resolve before proceeding."
        )
        sys.exit(1)


def make_lead_return(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, int, int]:
    """
    Add ret_a_lead: within-firm one-period-ahead return.
    Only assigns a lead value where the next firm observation is exactly year t+1;
    rows where a year gap is detected are set to NaN.

    Returns (df, last_idx, gap_mask, n_boundary_nan, n_gap_nan) so later 
    code can reuse these without repeating the groupby.
    """
    df = df.sort_values(["permno", "year"]).reset_index(drop=True)
    # Single groupby, shift both columns at once
    shifted = df.groupby("permno")[["year", "ret_a"]].shift(-1)
    df["ret_a_lead"] = shifted["ret_a"]
    gap_mask = shifted["year"].notna() & (shifted["year"] != df["year"] + 1)
    df.loc[gap_mask, "ret_a_lead"] = np.nan

    # Precompute diagnostics reused by verification and main-script reporting
    last_idx = df.groupby("permno")["year"].idxmax()
    n_boundary_nan = int(df.loc[last_idx, "ret_a_lead"].isna().sum())
    n_gap_nan = int(df["ret_a_lead"].isna().sum()) - n_boundary_nan

    return df, last_idx, gap_mask, n_boundary_nan, n_gap_nan


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_verification(
    df: pd.DataFrame,
    n_before: int,
    n_after: int,
    n_exact_dupes: int,
    last_idx: pd.Series,
    gap_mask: pd.Series,
) -> bool:
    """
    Run required verification tests; returns True if all checks pass.
    """
    results: list[tuple[str, str, bool]] = []

    # Row count — PASS if the only reduction matches intentional deduplication
    rows_ok = (n_before - n_after) == n_exact_dupes
    results.append((
        "Row count",
        f"Before: {n_before:,}  ->  After: {n_after:,}  (deduped: {n_exact_dupes:,})",
        rows_ok,
    ))

    dup_count = df.duplicated().sum()
    results.append(("Duplicates", f"Post-cleaning: {dup_count:,}", dup_count == 0))

    dup_keys = df.duplicated(subset=["permno", "year"]).sum()
    results.append(("Panel keys", f"Duplicate (permno, year): {dup_keys:,}", dup_keys == 0))

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    results.append((
        "Required cols",
        "All present" if not missing_cols else f"Missing: {missing_cols}",
        not missing_cols,
    ))

    type_ok = ("int" in str(df["permno"].dtype)) and ("int" in str(df["year"].dtype))
    results.append(("Dtypes", f"permno={df['permno'].dtype}  year={df['year'].dtype}", type_ok))

    non_numeric = [
        c for c in REGRESSORS + ["i2ppegt", "ret_a"]
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    results.append((
        "Numeric vars",
        "All numeric" if not non_numeric else f"Non-numeric: {non_numeric}",
        not non_numeric,
    ))

    # Reuse precomputed last_idx from make_lead_return
    last_obs_nan = df.loc[last_idx, "ret_a_lead"].isna().all()
    results.append(("ret_a_lead tail", "Last obs per firm is NaN", last_obs_nan))

    # Reuse precomputed gap_mask from make_lead_return
    n_gaps = int(gap_mask.sum())
    gaps_are_nan = df.loc[gap_mask, "ret_a_lead"].isna().all() if n_gaps > 0 else True
    results.append((
        "ret_a_lead gaps",
        f"Gap rows: {n_gaps:,}  --  all NaN: {gaps_are_nan}",
        gaps_are_nan,
    ))

    # Print verification results
    print(f"  {'Check':<20} {'Detail':<50} {'Status':>6}")
    print(f"  {'-'*20} {'-'*50} {'-'*6}")
    for name, detail, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<20} {detail:<50} {status:>6}")

    # Missing values (flagged only — not removed)
    check_cols = REQUIRED_COLS + ["ret_a_lead"]
    mv      = df[check_cols].isnull().sum()
    n_total = len(df)

    print(f"\n  Missing Values (flagged only - not removed)")
    print(f"  {'Column':<16} {'Missing':>10} {'% of N':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10}")
    for col, cnt in mv.items():
        print(f"  {col:<16} {cnt:>10,} {cnt / n_total * 100:>9.1f}%")

    all_ok = all(ok for _, _, ok in results)
    n_checks = len(results)
    if all_ok:
        print(f"\n  All {n_checks} checks passed\n")
    else:
        n_fail = sum(1 for _, _, ok in results if not ok)
        print(f"\n  {n_fail} of {n_checks} checks failed - review above\n")

    return all_ok


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def stars(pval: float) -> str:
    """Return significance stars for a two-tailed p-value."""
    if pval < 0.01: return "***"
    if pval < 0.05: return "**"
    if pval < 0.10: return "*"
    return ""



def _extract_fit_data(fit: object, r2_type: str) -> dict:
    """Extract all needed data from a pyfixest fit into a lightweight dict."""
    r2_within_val = getattr(fit, "_r2_within", None)
    r2_within = float(r2_within_val) if r2_within_val is not None else float(fit._r2)
    adj_r2_val = getattr(fit, "_adj_r2", np.nan)
    return {
        "coefs": fit.coef(),
        "tstats": fit.tstat(),
        "pvals": fit.pvalue(),
        "r2": float(fit._r2),
        "r2_within": r2_within if r2_type == "within" else float(fit._r2),
        "adj_r2": float(adj_r2_val) if pd.notna(adj_r2_val) else None,
        "N": int(fit._N),
    }


def build_reg_table(
    cached: list[dict], specs: list[dict], dep_var: str
) -> pd.DataFrame:
    """
    Build the multi-model summary DataFrame from pre-extracted fit data dicts.
    Rows: coefficient + t-stat per regressor, _cons for pooled OLS only,
    R² variants, FE flags, cluster level, N.
    Columns: specification labels (1)-(6).
    """
    col_labels = [f"({i + 1})" for i in range(len(cached))]
    rows: dict = {}

    for var in REGRESSORS:
        label = REG_LABELS[var]
        coef_col: list[str] = []
        tstat_col: list[str] = []
        for c_data in cached:
            coefs, tstats, pvals = c_data["coefs"], c_data["tstats"], c_data["pvals"]
            if var in coefs.index:
                coef_col.append(f"{float(coefs[var]):.3f}{stars(float(pvals[var]))}")
                tstat_col.append(f"({float(tstats[var]):.3f})")
            else:
                coef_col.append("")
                tstat_col.append("")
        rows[label] = coef_col
        rows[f"({label})"] = tstat_col

    # _cons: pooled OLS intercept only; FE models leave cells empty
    cons_coef: list[str] = []
    cons_tstat: list[str] = []
    for c_data in cached:
        coefs, tstats, pvals = c_data["coefs"], c_data["tstats"], c_data["pvals"]
        if "Intercept" in coefs.index:
            cons_coef.append(f"{float(coefs['Intercept']):.3f}{stars(float(pvals['Intercept']))}")
            cons_tstat.append(f"({float(tstats['Intercept']):.3f})")
        else:
            cons_coef.append("")
            cons_tstat.append("")
    rows["_cons"] = cons_coef
    rows["(_cons)"] = cons_tstat

    rows["R2 (Within)"]  = [f"{d['r2_within']:.3f}" for d in cached]
    rows["R2 (Overall)"] = [f"{d['r2']:.3f}" for d in cached]
    rows["Adj. R2"] = [
        f"{d['adj_r2']:.3f}" if d["adj_r2"] is not None else "" for d in cached
    ]

    rows["Firm FE"]    = ["Yes" if s["firm_fe"] else "No" for s in specs]
    rows["Year FE"]    = ["Yes" if s["year_fe"] else "No" for s in specs]
    rows["Cluster SE"] = [s["cluster"]                    for s in specs]
    rows["N"]          = [f"{d['N']:,}"                   for d in cached]

    df = pd.DataFrame(rows).T
    df.columns = col_labels
    df.index.name = None
    return df


def print_family_table(table_df: pd.DataFrame, title: str) -> None:
    """Print a regression summary DataFrame as a plain-text table."""
    col_w = 14
    label_w = 16
    header = f"  {'':>{label_w}}" + "".join(f"{c:>{col_w}}" for c in table_df.columns)

    print(f"\n  {title}")
    print(f"  {'=' * (label_w + col_w * len(table_df.columns))}")
    print(header)
    print(f"  {'-' * (label_w + col_w * len(table_df.columns))}")

    for idx_val, row in table_df.iterrows():
        label = str(idx_val)
        # Divider before R² block
        if label == "R2 (Within)":
            print(f"  {'-' * (label_w + col_w * len(table_df.columns))}")
        vals = "".join(f"{v:>{col_w}}" for v in row.tolist())
        print(f"  {label:>{label_w}}{vals}")

    print(f"  {'-' * (label_w + col_w * len(table_df.columns))}")
    print(f"  * p<0.10  ** p<0.05  *** p<0.01  |  t-stats in parentheses")
    print()


def run_regression_family(
    dep_var: str, data: pd.DataFrame
) -> tuple[pd.DataFrame, int, int]:
    """
    Run all six specifications for dep_var.

    Only 3 distinct coefficient models are estimated:
      - pooled OLS (spec 1)
      - firm FE    (spec 2)
      - firm+year FE (specs 3-6, different vcov only)
    Specs 4-6 reuse the firm+year FE fit by calling .vcov() in-place and
    extracting results.
    """
    cols_needed = [dep_var] + REGRESSORS + ["permno", "year"]
    sample      = data[cols_needed].dropna()
    n_used      = len(sample)
    n_dropped   = len(data) - n_used

    base = f"{dep_var} ~ {' + '.join(REGRESSORS)}"

    # 3 unique model fits
    fit_pooled    = feols(base,                      data=sample, vcov="iid")
    fit_firm      = feols(f"{base} | permno",        data=sample, vcov="iid")
    fit_firm_year = feols(f"{base} | permno + year", data=sample, vcov="iid")

    # Extract data from the first three fits
    cached = [
        _extract_fit_data(fit_pooled, "overall"),
        _extract_fit_data(fit_firm, "within"),
        _extract_fit_data(fit_firm_year, "within"),
    ]

    # Specs 4-6: mutate vcov on the same fit object, extract after each change.
    # No deepcopy needed — .vcov() only recomputes standard errors in-place.
    for vcov_spec in [{"CRV1": "permno"}, {"CRV1": "year"}, {"CRV1": "permno + year"}]:
        fit_firm_year.vcov(vcov_spec)
        cached.append(_extract_fit_data(fit_firm_year, "within"))

    table = build_reg_table(cached, SPEC_META, dep_var)
    return table, n_used, n_dropped


def export_results(
    inv_table: pd.DataFrame,
    ret_table: pd.DataFrame,
    meta: dict,
    output_path: Path,
) -> None:
    """Export per-family regression tables and metadata to Excel."""
    INV_CAPTION = (
        "Table 1: Investment Rate (I2ppegt) — "
        "(1) Pooled OLS  (2) Firm FE  (3)-(6) Firm+Year FE  |  "
        "* p<0.10  ** p<0.05  *** p<0.01  |  t-stats in parentheses  |  "
        "Within-R2 for FE models; overall R2 for Pooled OLS"
    )
    RET_CAPTION = (
        "Table 2: Next-Year Return (ret_A_lead) — "
        "(1) Pooled OLS  (2) Firm FE  (3)-(6) Firm+Year FE  |  "
        "* p<0.10  ** p<0.05  *** p<0.01  |  t-stats in parentheses  |  "
        "Within-R2 for FE models; overall R2 for Pooled OLS  |  "
        "N excludes boundary obs (last firm-year, no t+1) and gap obs "
        "(year discontinuity — multi-year lookaheads suppressed)"
    )

    meta_df = pd.DataFrame(list(meta.items()), columns=["Item", "Value"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        inv_table.to_excel(writer, sheet_name="Investment_Regressions", startrow=2)
        writer.sheets["Investment_Regressions"].cell(row=1, column=1).value = INV_CAPTION

        ret_table.to_excel(writer, sheet_name="Returns_Regressions", startrow=2)
        writer.sheets["Returns_Regressions"].cell(row=1, column=1).value = RET_CAPTION

        meta_df.to_excel(writer, sheet_name="Metadata", index=False)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

# 1. Load raw data
print(f"\n{SEP}\n  DATA LOADING\n{SEP}")
raw, data_path = load_data(DATA_CANDIDATES)
print(f"  {data_path}    {raw.shape[0]:,} rows    {raw.shape[1]} cols\n")

# 2. Clean and prepare the panel
print(f"{SEP}\n  DATA CLEANING\n{SEP}")

raw = standardize_columns(raw)
raw = convert_year(raw)
raw = validate_and_coerce_permno(raw)

# Compact integer dtypes reduce memory and speed up FE factorization.
raw["permno"] = raw["permno"].astype("int32")
raw["year"]   = raw["year"].astype("int32")

str_cols   = raw.select_dtypes(include="object").columns
n_str_cols = len(str_cols)
raw[str_cols] = raw[str_cols].apply(lambda s: s.str.strip())

# Surface duplicate panel keys before any row removal
n_dup_keys_before = int(raw.duplicated(subset=["permno", "year"]).sum())

n_before = len(raw)
raw      = raw.drop_duplicates()
n_after  = len(raw)
n_exact_dupes = n_before - n_after

validate_panel(raw)

# make_lead_return returns precomputed diagnostics to avoid redundant groupby
raw, last_idx, gap_mask, n_boundary_nan, n_gap_nan = make_lead_return(raw)

n_unique_firms = raw["permno"].nunique()

print(f"  Column names standardized             lowercase with underscores")
print(f"  Year column converted                 int32")
print(f"  permno validated and coerced          int32")
print(f"  String columns trimmed                {n_str_cols} column(s)")
print(f"  Duplicate (permno, year) before dedup {n_dup_keys_before:,} found")
print(f"  Exact duplicate rows removed          {n_exact_dupes:,} dropped")
print(f"  ret_a_lead: boundary NaN (last obs)   {n_boundary_nan:,} rows")
print(f"  ret_a_lead: gap NaN (year discontin.) {n_gap_nan:,} rows")
print(f"\n  Panel   {raw['year'].min()}-{raw['year'].max()}  |  "
      f"{n_unique_firms:,} firms  |  {len(raw):,} obs\n")

# 3. Run verification tests
print(f"{SEP}\n  VERIFICATION\n{SEP}")
verification_passed = run_verification(
    raw, n_before, n_after, n_exact_dupes, last_idx, gap_mask
)

# 4. Run regressions — pyfixest feols, 6 specifications x 2 dependent variables
print(f"{SEP}\n  REGRESSIONS\n{SEP}")
print("  pyfixest feols -- 6 specifications x 2 dependent variables\n")

inv_table, inv_n, inv_dropped = run_regression_family("i2ppegt",    raw)
ret_table, ret_n, ret_dropped = run_regression_family("ret_a_lead", raw)

print(f"  {'Dependent Variable':<35} {'N Used':>10} {'Dropped':>10}")
print(f"  {'-'*35} {'-'*10} {'-'*10}")
print(f"  {'I2ppegt (investment rate)':<35} {inv_n:>10,} {inv_dropped:>10,}")
print(f"  {'ret_A_lead (next-year return)':<35} {ret_n:>10,} {ret_dropped:>10,}")

print_family_table(inv_table, "Table 1 -- Investment Rate  |  Dep. Var: I2ppegt")
print_family_table(ret_table, "Table 2 -- Next-Year Return  |  Dep. Var: ret_A_lead")

# 5. Export results to Excel
print(f"\n{SEP}\n  EXPORT\n{SEP}")
meta = {
    "Dataset":                           str(data_path),
    "Regression engine":                 "pyfixest feols (reghdfe equivalent)",
    "Total rows after cleaning":         f"{n_after:,}",
    "Exact duplicates dropped":          f"{n_exact_dupes:,}",
    "Year range":                        f"{raw['year'].min()}-{raw['year'].max()}",
    "Unique firms":                      f"{n_unique_firms:,}",
    "Investment sample (N)":                        f"{inv_n:,}",
    "Investment rows dropped (missing)":            f"{inv_dropped:,}",
    "Returns sample (N)":                           f"{ret_n:,}",
    "Returns rows dropped (missing)":               f"{ret_dropped:,}",
    "ret_a_lead boundary NaN (last obs per firm)":  f"{n_boundary_nan:,}",
    "ret_a_lead gap NaN (year discontinuity)":      f"{n_gap_nan:,}",
}
export_results(inv_table, ret_table, meta, OUTPUT_FILE)

print(f"  {OUTPUT_FILE}")
print(f"    Investment_Regressions  Table 1 -- I2ppegt, 6 specifications")
print(f"    Returns_Regressions     Table 2 -- ret_A_lead, 6 specifications")
print(f"    Metadata                Dataset info and sample sizes\n")

if verification_passed:
    print(f"{SEP}\n  Complete\n{SEP}")
else:
    print(f"{SEP}\n  Complete - VERIFICATION FAILED\n{SEP}")
