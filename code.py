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
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_CANDIDATES = [Path("CCM_sample.dta"), Path("CCM sample.dta")]
OUTPUT_FILE     = Path("task5_panel_results.xlsx")

# Columns that must be present after name standardization (gpa is optional)
REQUIRED_COLS = ["permno", "year", "bm", "i2ppegt", "logme", "blev", "g_sale", "ret_a"]
REGRESSORS    = ["logme", "bm", "g_sale", "blev"]

console = Console(legacy_windows=False)


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_data(candidates: list[Path]) -> tuple[pd.DataFrame, Path]:
    """Try each candidate path in order; exit with a clear error if none exists."""
    for path in candidates:
        if path.exists():
            return pd.read_stata(path), path
    tried = ", ".join(str(p) for p in candidates)
    console.print(f"[bold red]ERROR[/]  data file not found. Tried: {tried}")
    sys.exit(1)


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
            console.print(f"[bold red]ERROR[/]  {invalid} non-numeric values in 'year' after coercion.")
            sys.exit(1)
        df["year"] = df["year"].astype(int)
    return df


def validate_panel(df: pd.DataFrame) -> None:
    """
    Check that (permno, year) keys are unique before any lead construction.
    Duplicate panel keys would silently corrupt the lead-return variable.
    """
    dup_keys = df.duplicated(subset=["permno", "year"]).sum()
    if dup_keys > 0:
        console.print(
            f"[bold red]ERROR[/]  {dup_keys} duplicate (permno, year) rows found. "
            "Resolve before constructing lead returns."
        )
        sys.exit(1)


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
    """Run required verification tests and display results as Rich tables."""
    results: list[bool] = []

    def _status(ok: bool, warn_if_false: bool = False) -> str:
        if not ok and warn_if_false:
            return "[bold yellow]WARN[/]"
        return "[bold green]PASS[/]" if ok else "[bold red]FAIL[/]"

    # --- Verification checks table ---
    checks = Table(box=box.SIMPLE_HEAD, header_style="bold", expand=False, padding=(0, 1))
    checks.add_column("Check",  style="dim", min_width=16)
    checks.add_column("Detail",              min_width=50)
    checks.add_column("Status", justify="center", min_width=6)

    # Row count (WARN if rows dropped — exact-duplicate removal is expected)
    rows_ok = n_after == n_before
    results.append(rows_ok)
    checks.add_row(
        "Row count",
        f"Before: {n_before:,}  ->  After: {n_after:,}  (dropped: {n_before - n_after:,})",
        _status(rows_ok, warn_if_false=True),
    )

    # Exact duplicates post-clean
    dup_count = df.duplicated().sum()
    results.append(dup_count == 0)
    checks.add_row("Duplicates", f"Post-cleaning: {dup_count:,}", _status(dup_count == 0))

    # Panel key uniqueness
    dup_keys = df.duplicated(subset=["permno", "year"]).sum()
    results.append(dup_keys == 0)
    checks.add_row("Panel keys", f"Duplicate (permno, year): {dup_keys:,}", _status(dup_keys == 0))

    # Required columns present
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    results.append(not missing_cols)
    checks.add_row(
        "Required cols",
        "All present" if not missing_cols else f"Missing: {missing_cols}",
        _status(not missing_cols),
    )

    # Dtype checks (permno and year must be integer)
    type_ok = ("int" in str(df["permno"].dtype)) and ("int" in str(df["year"].dtype))
    results.append(type_ok)
    checks.add_row(
        "Dtypes",
        f"permno={df['permno'].dtype}  year={df['year'].dtype}",
        _status(type_ok),
    )

    # All regression variables must be numeric
    non_numeric = [
        c for c in REGRESSORS + ["i2ppegt", "ret_a"]
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    results.append(not non_numeric)
    checks.add_row(
        "Numeric vars",
        "All numeric" if not non_numeric else f"Non-numeric: {non_numeric}",
        _status(not non_numeric),
    )

    # Lead-return: last observation per firm must be NaN
    last_obs_nan = df.groupby("permno")["ret_a_lead"].apply(lambda s: s.iloc[-1]).isna().all()
    results.append(last_obs_nan)
    checks.add_row("ret_a_lead", "Last obs per firm is NaN", _status(last_obs_nan))

    console.print(checks)

    # --- Missing values table (flagged only — not dropped or imputed) ---
    check_cols = REQUIRED_COLS + ["ret_a_lead"]
    mv = df[check_cols].isnull().sum()
    n_total = len(df)

    mv_table = Table(
        title="Missing Values",
        caption="flagged only - not imputed or dropped",
        box=box.SIMPLE_HEAD, header_style="bold", expand=False, padding=(0, 1),
    )
    mv_table.add_column("Column",  style="dim", min_width=16)
    mv_table.add_column("Missing", justify="right", min_width=9)
    mv_table.add_column("% of N",  justify="right", min_width=7)
    for col, cnt in mv.items():
        mv_table.add_row(
            col,
            f"{cnt:,}",
            f"{cnt / n_total * 100:.1f}%",
            style="yellow" if cnt > 0 else "",
        )
    console.print(mv_table)

    # --- Invalid value flags (flagged only — not removed) ---
    flags_data = [
        ("i2ppegt < 0",  int((df["i2ppegt"] < 0).sum())),
        ("|bm| > 10",    int((df["bm"].abs() > 10).sum())),
        ("i2ppegt > 10", int((df["i2ppegt"] > 10).sum())),
    ]

    flags_table = Table(
        title="Invalid Value Flags",
        caption="flagged only - not removed",
        box=box.SIMPLE_HEAD, header_style="bold", expand=False, padding=(0, 1),
    )
    flags_table.add_column("Flag",  style="dim", min_width=16)
    flags_table.add_column("Count", justify="right", min_width=9)
    for flag, cnt in flags_data:
        flags_table.add_row(flag, f"{cnt:,}", style="yellow" if cnt > 0 else "")
    console.print(flags_table)

    # --- Coverage summary ---
    console.print(
        f"\n  [dim]Coverage[/]   "
        f"{df['year'].min()}-{df['year'].max()}  [dim]|[/]  "
        f"Firms: {df['permno'].nunique():,}  [dim]|[/]  "
        f"Obs: {len(df):,}\n"
    )

    # --- Overall result ---
    overall = all(results)
    if overall:
        console.print(Panel("[bold green]  OVERALL: PASS  [/]", expand=False, border_style="green", padding=(0, 2)))
    else:
        console.print(Panel("[bold red]  OVERALL: FAIL — review checks above  [/]", expand=False, border_style="red", padding=(0, 2)))


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
    Returns (result table, N used, N dropped).
    """
    sample    = panel[[dep_var] + REGRESSORS].dropna()
    y         = sample[dep_var]
    X         = sample[REGRESSORS]
    n_used    = len(sample)
    n_dropped = len(panel) - n_used

    console.print(
        f"  [dim]>[/] [bold]{dep_var}[/]"
        f"  [dim]{n_used:,} obs  ({n_dropped:,} dropped for missing values)[/]"
    )

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

# Stage 1: Load raw data (tries CCM_sample.dta then CCM sample.dta)
console.rule("[bold gold1] DATA LOADING [/]")
raw, data_path = load_data(DATA_CANDIDATES)
console.print(f"  [dim]source[/]   {data_path}  ({raw.shape[0]:,} rows x {raw.shape[1]} cols)")

# Stage 2: Clean and prepare the panel
console.rule("[bold gold1] DATA CLEANING [/]")

# Standardize column names to lowercase with underscores
raw = standardize_columns(raw)

# Convert year to integer (handles both datetime and numeric imports)
raw = convert_year(raw)

# Cast permno to int (Stata stores it as float64 with no fractional part)
raw["permno"] = raw["permno"].astype(int)

# Trim whitespace in any string columns
str_cols = raw.select_dtypes(include="object").columns
raw[str_cols] = raw[str_cols].apply(lambda s: s.str.strip())

# Drop exact duplicate rows before panel validation
n_before = len(raw)
raw = raw.drop_duplicates()
n_after  = len(raw)

# Validate panel keys — must be unique before constructing lead return
validate_panel(raw)

# Construct one-period-ahead return within each firm
raw = make_lead_return(raw)

console.print(
    f"  Columns standardized  |  Year converted  |  permno cast to int  |  "
    f"Exact duplicates dropped: {n_before - n_after:,}  |  Lead return constructed"
)

# Stage 3: Run verification tests
console.rule("[bold gold1] VERIFICATION [/]")
console.print()
run_verification(raw, n_before, n_after)
console.print()

# Stage 4: Run regressions
console.rule("[bold gold1] REGRESSIONS [/]")
console.print()
panel = raw.set_index(["permno", "year"])

inv_table, inv_n, inv_dropped = run_regression_family(panel, "i2ppegt")
ret_table, ret_n, ret_dropped = run_regression_family(panel, "ret_a_lead")

# Sample size summary table
sample_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold", expand=False, padding=(0, 1))
sample_tbl.add_column("Dependent Variable", style="dim", min_width=30)
sample_tbl.add_column("N Used",             justify="right", min_width=9)
sample_tbl.add_column("Dropped (missing)",  justify="right", min_width=18)
sample_tbl.add_row("I2ppegt (investment rate)",     f"{inv_n:,}", f"{inv_dropped:,}")
sample_tbl.add_row("ret_A_lead (next-year return)", f"{ret_n:,}", f"{ret_dropped:,}")
console.print()
console.print(sample_tbl)
console.print()

# Stage 5: Export results to Excel
console.rule("[bold gold1] EXPORT [/]")
meta = {
    "Dataset":                           str(data_path),
    "Total rows after cleaning":         f"{n_after:,}",
    "Exact duplicates dropped":          f"{n_before - n_after:,}",
    "Year range":                        f"{raw['year'].min()}–{raw['year'].max()}",
    "Unique firms":                      f"{raw['permno'].nunique():,}",
    "Investment sample (N)":             f"{inv_n:,}",
    "Investment rows dropped (missing)": f"{inv_dropped:,}",
    "Returns sample (N)":                f"{ret_n:,}",
    "Returns rows dropped (missing)":    f"{ret_dropped:,}",
}

export_results(raw, inv_table, ret_table, meta, OUTPUT_FILE)
console.print(f"  [dim]output[/]   {OUTPUT_FILE}")
console.print(f"  [dim]sheets[/]   Cleaned_Data  |  Investment_Regressions  |  Returns_Regressions  |  Metadata")
console.print()
