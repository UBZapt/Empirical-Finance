"""
Empirical Finance - Task 5: Panel Data Analysis
Loads and cleans the CCM firm-year panel, runs six regression specifications
for each of two dependent variables (investment rate and next-year returns),
displays results in the terminal, and exports all results to Excel.

Regression engine: pyfixest feols(), replicating Stata's reghdfe.
"""

from pathlib import Path
import sys

# pyfixest's dependency chain (feols → IPython → pdb) does 'import code'.
# Because this script is named code.py and Python adds its directory to
# sys.path[0] at startup, the interpreter finds this file before the stdlib
# module, causing a circular-import crash. Removing the script directory
# before pyfixest loads prevents the collision. No local-module imports are
# used, so this is safe.
if sys.path and sys.path[0] not in ("",):
    sys.path.pop(0)

import pandas as pd
from pyfixest.estimation import feols
from rich.console import Console
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_CANDIDATES = [Path("CCM_sample.dta"), Path("CCM sample.dta")]
OUTPUT_FILE     = Path("task5_panel_results.xlsx")

REQUIRED_COLS = ["permno", "year", "bm", "i2ppegt", "logme", "blev", "g_sale", "ret_a"]
REGRESSORS    = ["logme", "bm", "g_sale", "blev"]
REG_LABELS    = {"logme": "logME", "bm": "bm", "g_sale": "g_sale", "blev": "blev"}

# Six specifications mirroring Panel_Examples.do (reghdfe).
# formula_fe: the absorb/FE part appended to the base formula (empty = pooled OLS).
# vcov: pyfixest variance estimator — "iid" for conventional, CRV1 dict for clustered.
SPEC_META: list[dict] = [
    {"formula_fe": "",                 "vcov": "iid",                             "firm_fe": False, "year_fe": False, "cluster": "None",      "r2_type": "overall"},
    {"formula_fe": "| permno",         "vcov": "iid",                             "firm_fe": True,  "year_fe": False, "cluster": "None",      "r2_type": "within"},
    {"formula_fe": "| permno + year",  "vcov": "iid",                             "firm_fe": True,  "year_fe": True,  "cluster": "None",      "r2_type": "within"},
    {"formula_fe": "| permno + year",  "vcov": {"CRV1": "permno"},                "firm_fe": True,  "year_fe": True,  "cluster": "Firm",      "r2_type": "within"},
    {"formula_fe": "| permno + year",  "vcov": {"CRV1": "year"},                  "firm_fe": True,  "year_fe": True,  "cluster": "Year",      "r2_type": "within"},
    {"formula_fe": "| permno + year",  "vcov": {"CRV1": "permno + year"},         "firm_fe": True,  "year_fe": True,  "cluster": "Firm+Year", "r2_type": "within"},
]

console = Console(width=100, legacy_windows=False)


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_data(candidates: list[Path]) -> tuple[pd.DataFrame, Path]:
    """Try each candidate path in order; exit with a clear error if none exists."""
    for path in candidates:
        if path.exists():
            return pd.read_stata(path), path
    tried = ", ".join(str(p) for p in candidates)
    console.print(f"[red]ERROR[/]  data file not found. Tried: {tried}")
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
    Uses .dt.year for datetime-typed columns (Stata stores annual dates as Jan 1);
    otherwise coerces to numeric and validates before converting to int.
    """
    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["year"]):
        df["year"] = df["year"].dt.year.astype(int)
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        invalid = df["year"].isna().sum()
        if invalid > 0:
            console.print(f"[red]ERROR[/]  {invalid} non-numeric values in 'year' after coercion.")
            sys.exit(1)
        df["year"] = df["year"].astype(int)
    return df


def validate_panel(df: pd.DataFrame) -> None:
    """
    Check that (permno, year) keys are unique before lead construction.
    Duplicate panel keys would silently corrupt ret_a_lead.
    """
    dup_keys = df.duplicated(subset=["permno", "year"]).sum()
    if dup_keys > 0:
        console.print(
            f"[red]ERROR[/]  {dup_keys} duplicate (permno, year) rows found. "
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

def run_verification(df: pd.DataFrame, n_before: int, n_after: int) -> bool:
    """Run required verification tests; returns True if all checks pass."""
    results: list[bool] = []

    def _status(ok: bool, warn_if_false: bool = False) -> str:
        if not ok and warn_if_false:
            return "[yellow]WARN[/]"
        return "[green]PASS[/]" if ok else "[red]FAIL[/]"

    checks = Table(box=box.SIMPLE_HEAD, header_style="bold", expand=True, padding=(0, 1))
    checks.add_column("Check",  min_width=16)
    checks.add_column("Detail")
    checks.add_column("Status", justify="center", min_width=6, no_wrap=True)

    rows_ok = n_after == n_before
    results.append(rows_ok)
    checks.add_row(
        "Row count",
        f"Before: {n_before:,}  ->  After: {n_after:,}  (dropped: {n_before - n_after:,})",
        _status(rows_ok, warn_if_false=True),
    )

    dup_count = df.duplicated().sum()
    results.append(dup_count == 0)
    checks.add_row("Duplicates", f"Post-cleaning: {dup_count:,}", _status(dup_count == 0))

    dup_keys = df.duplicated(subset=["permno", "year"]).sum()
    results.append(dup_keys == 0)
    checks.add_row("Panel keys", f"Duplicate (permno, year): {dup_keys:,}", _status(dup_keys == 0))

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    results.append(not missing_cols)
    checks.add_row(
        "Required cols",
        "All present" if not missing_cols else f"Missing: {missing_cols}",
        _status(not missing_cols),
    )

    type_ok = ("int" in str(df["permno"].dtype)) and ("int" in str(df["year"].dtype))
    results.append(type_ok)
    checks.add_row(
        "Dtypes",
        f"permno={df['permno'].dtype}  year={df['year'].dtype}",
        _status(type_ok),
    )

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

    last_obs_nan = df.groupby("permno")["ret_a_lead"].apply(lambda s: s.iloc[-1]).isna().all()
    results.append(last_obs_nan)
    checks.add_row("ret_a_lead", "Last obs per firm is NaN", _status(last_obs_nan))

    console.print(checks)

    # Missing values (flagged only — not removed)
    check_cols = REQUIRED_COLS + ["ret_a_lead"]
    mv      = df[check_cols].isnull().sum()
    n_total = len(df)

    mv_table = Table(
        title="[underline]Missing Values[/underline]",
        caption="flagged only - not removed",
        box=box.SIMPLE_HEAD, header_style="bold", expand=True, padding=(0, 1),
    )
    mv_table.add_column("Column",  min_width=16)
    mv_table.add_column("Missing", justify="right")
    mv_table.add_column("% of N",  justify="right")
    for col, cnt in mv.items():
        mv_table.add_row(col, f"{cnt:,}", f"{cnt / n_total * 100:.1f}%")
    console.print(mv_table)
    console.print()

    # Potential invalid value flags (flagged only — not removed)
    flags_data = [
        ("i2ppegt < 0",  int((df["i2ppegt"] < 0).sum())),
        ("|bm| > 10",    int((df["bm"].abs() > 10).sum())),
        ("i2ppegt > 10", int((df["i2ppegt"] > 10).sum())),
    ]

    flags_table = Table(
        title="[underline]Potential Invalid Value Flags[/underline]",
        caption="flagged only - not removed",
        box=box.SIMPLE_HEAD, header_style="bold", expand=True, padding=(0, 1),
    )
    flags_table.add_column("Flag",  min_width=16)
    flags_table.add_column("Count", justify="right")
    for flag, cnt in flags_data:
        flags_table.add_row(flag, f"{cnt:,}")
    console.print(flags_table)

    n_pass   = sum(results)
    n_checks = len(results)
    if all(results):
        console.print(f"\n  [green]All {n_checks} checks passed[/]\n")
    else:
        console.print(f"\n  [red]{n_checks - n_pass} of {n_checks} checks failed - review above[/]\n")

    return all(results)


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

def stars(pval: float) -> str:
    """Return significance stars for a two-tailed p-value."""
    if pval < 0.01: return "***"
    if pval < 0.05: return "**"
    if pval < 0.10: return "*"
    return ""


def run_spec(dep_var: str, spec: dict, data: pd.DataFrame) -> object:
    """
    Fit one feols specification.
    Constructs the formula from REGRESSORS and the FE part, then calls feols().
    pyfixest takes a flat DataFrame; permno and year are referenced by name in the formula.
    """
    base    = f"{dep_var} ~ {' + '.join(REGRESSORS)}"
    formula = f"{base} {spec['formula_fe']}".strip()
    return feols(formula, data=data, vcov=spec["vcov"])


def get_r2(fit: object, r2_type: str) -> float:
    """Return within-R² for FE models or overall R² for pooled OLS."""
    if r2_type == "within":
        r2w = getattr(fit, "_r2_within", None)
        return float(r2w) if r2w is not None else float(fit._r2)
    return float(fit._r2)


def build_reg_table(fits: list, specs: list[dict]) -> pd.DataFrame:
    """
    Build the multi-model summary DataFrame from a list of fitted pyfixest results.
    Rows: coefficient + t-stat per regressor, R², FE flags, cluster level, N.
    Columns: specification labels (1)-(6).
    """
    col_labels = [f"({i + 1})" for i in range(len(fits))]
    rows: dict = {}

    for var in REGRESSORS:
        label    = REG_LABELS[var]
        coef_col: list[str] = []
        tstat_col: list[str] = []
        for fit in fits:
            coefs  = fit.coef()
            tstats = fit.tstat()
            pvals  = fit.pvalue()
            if var in coefs.index:
                c = float(coefs[var])
                t = float(tstats[var])
                p = float(pvals[var])
                coef_col.append(f"{c:.3f}{stars(p)}")
                tstat_col.append(f"({t:.3f})")
            else:
                coef_col.append("")
                tstat_col.append("")
        rows[label]        = coef_col
        rows[f"({label})"] = tstat_col

    rows["R\u00b2"]       = [f"{get_r2(fit, s['r2_type']):.3f}" for fit, s in zip(fits, specs)]
    rows["Firm FE"]    = ["Yes" if s["firm_fe"] else "No" for s in specs]
    rows["Year FE"]    = ["Yes" if s["year_fe"] else "No" for s in specs]
    rows["Cluster SE"] = [s["cluster"]                    for s in specs]
    rows["N"]          = [f"{int(fit._N):,}"              for fit in fits]

    df = pd.DataFrame(rows).T
    df.columns    = col_labels
    df.index.name = None
    return df


def print_rich_family_table(table_df: pd.DataFrame, title: str) -> None:
    """Render a regression summary DataFrame as a Rich console table."""
    tbl = Table(
        title=f"[bold]{title}[/bold]",
        caption="* p<0.10  ** p<0.05  *** p<0.01  |  t-stats in parentheses",
        box=box.SIMPLE_HEAD,
        header_style="bold",
        expand=True,
        padding=(0, 1),
    )
    tbl.add_column("", min_width=14)
    for col in table_df.columns:
        tbl.add_column(col, justify="right", min_width=10)

    # Insert a section divider before R² to separate coefficients from model statistics
    SECTION_BEFORE = {"\u00b2"}  # matches "R²"
    for idx_val, row in table_df.iterrows():
        label    = str(idx_val)
        is_tstat = label.startswith("(")
        if label in SECTION_BEFORE or label == "R\u00b2":
            tbl.add_section()
        tbl.add_row(label, *row.tolist(), style="dim" if is_tstat else "")

    console.print(tbl)
    console.print()


def run_regression_family(
    dep_var: str, data: pd.DataFrame
) -> tuple[pd.DataFrame, list, int, int]:
    """
    Run all six feols specifications for dep_var.
    Returns (summary_df, list_of_fits, n_used, n_dropped).
    permno and year are kept in the sample so pyfixest can resolve FE and cluster variables.
    """
    cols_needed = [dep_var] + REGRESSORS + ["permno", "year"]
    sample      = data[cols_needed].dropna()
    n_used      = len(sample)
    n_dropped   = len(data) - n_used

    fits  = [run_spec(dep_var, spec, sample) for spec in SPEC_META]
    table = build_reg_table(fits, SPEC_META)
    return table, fits, n_used, n_dropped


def export_results(
    data: pd.DataFrame,
    inv_table: pd.DataFrame,
    ret_table: pd.DataFrame,
    meta: dict,
    output_path: Path,
) -> None:
    """Export cleaned data, per-family tables, combined summary, and metadata to Excel."""
    INV_CAPTION = (
        "Table 1: Investment Rate (I2ppegt) — "
        "(1) Pooled OLS  (2) Firm FE  (3)-(6) Firm+Year FE  |  "
        "* p<0.10  ** p<0.05  *** p<0.01  |  t-stats in parentheses  |  "
        "Within-R\u00b2 for FE models; overall R\u00b2 for Pooled OLS"
    )
    RET_CAPTION = (
        "Table 2: Next-Year Return (ret_A_lead) — "
        "(1) Pooled OLS  (2) Firm FE  (3)-(6) Firm+Year FE  |  "
        "* p<0.10  ** p<0.05  *** p<0.01  |  t-stats in parentheses  |  "
        "Within-R\u00b2 for FE models; overall R\u00b2 for Pooled OLS"
    )
    COMBINED_CAPTION = (
        "Combined Panel Regression Results — "
        "* p<0.10  ** p<0.05  *** p<0.01  |  t-stats in parentheses  |  "
        "Within-R\u00b2 for FE models; overall R\u00b2 for Pooled OLS"
    )

    # Combined summary: both families stacked with a blank separator row
    gap = pd.DataFrame(
        [[""] * len(inv_table.columns)],
        columns=inv_table.columns,
        index=[""],
    )
    combined = pd.concat([inv_table, gap, ret_table])

    meta_df = pd.DataFrame(list(meta.items()), columns=["Item", "Value"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="Cleaned_Data", index=False)

        inv_table.to_excel(writer, sheet_name="Investment_Regressions", startrow=2)
        writer.sheets["Investment_Regressions"].cell(row=1, column=1).value = INV_CAPTION

        ret_table.to_excel(writer, sheet_name="Returns_Regressions", startrow=2)
        writer.sheets["Returns_Regressions"].cell(row=1, column=1).value = RET_CAPTION

        combined.to_excel(writer, sheet_name="Combined_Summary", startrow=4)
        ws = writer.sheets["Combined_Summary"]
        ws.cell(row=1, column=1).value = COMBINED_CAPTION
        ws.cell(row=2, column=1).value = (
            "Top panel: Investment Rate (I2ppegt).  "
            "Bottom panel: Next-Year Return (ret_A_lead)."
        )

        meta_df.to_excel(writer, sheet_name="Metadata", index=False)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

# 1. Load raw data
console.rule("[bold gold1] DATA LOADING [/]")
raw, data_path = load_data(DATA_CANDIDATES)
console.print(f"\n  {data_path}    {raw.shape[0]:,} rows    {raw.shape[1]} cols\n")

# 2. Clean and prepare the panel
console.rule("[bold gold1] DATA CLEANING [/]")

raw = standardize_columns(raw)
raw = convert_year(raw)

permno_original_dtype = str(raw["permno"].dtype)
raw["permno"] = raw["permno"].astype(int)

str_cols   = raw.select_dtypes(include="object").columns
n_str_cols = len(str_cols)
raw[str_cols] = raw[str_cols].apply(lambda s: s.str.strip())

n_before = len(raw)
raw      = raw.drop_duplicates()
n_after  = len(raw)

validate_panel(raw)
raw = make_lead_return(raw)

cleaning_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold", expand=True, padding=(0, 1))
cleaning_tbl.add_column("Step")
cleaning_tbl.add_column("Result", justify="right")
cleaning_tbl.add_row("Column names standardized",        "lowercase with underscores")
cleaning_tbl.add_row("Year column converted",            "int64")
cleaning_tbl.add_row("Firm ID (permno) coerced to int", f"{permno_original_dtype} -> int64")
cleaning_tbl.add_row("String columns trimmed",           f"{n_str_cols} column(s)")
cleaning_tbl.add_row("Exact duplicate rows removed",     f"{n_before - n_after:,} dropped")
cleaning_tbl.add_row("Lead return constructed",          "ret_a_lead added (within-firm, t+1)")
console.print(cleaning_tbl)

console.print(
    f"  Panel   {raw['year'].min()}-{raw['year'].max()}  |  "
    f"{raw['permno'].nunique():,} firms  |  "
    f"{len(raw):,} obs\n"
)

# 3. Run verification tests
console.rule("[bold gold1] VERIFICATION [/]")
verification_passed = run_verification(raw, n_before, n_after)

# 4. Run regressions — pyfixest feols, 6 specifications x 2 dependent variables
console.rule("[bold gold1] REGRESSIONS [/]")
console.print()
console.print("  pyfixest feols \u2014 6 specifications \u00d7 2 dependent variables\n")

inv_table, inv_fits, inv_n, inv_dropped = run_regression_family("i2ppegt",    raw)
ret_table, ret_fits, ret_n, ret_dropped = run_regression_family("ret_a_lead", raw)

reg_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold", expand=True, padding=(0, 1))
reg_tbl.add_column("Dependent Variable")
reg_tbl.add_column("N Used",            justify="right")
reg_tbl.add_column("Dropped (missing)", justify="right")
reg_tbl.add_row("I2ppegt (investment rate)",     f"{inv_n:,}", f"{inv_dropped:,}")
reg_tbl.add_row("ret_A_lead (next-year return)", f"{ret_n:,}", f"{ret_dropped:,}")
console.print(reg_tbl)
console.print()

# Print per-family regression result tables
print_rich_family_table(
    inv_table,
    "Table 1 \u2014 Investment Rate  |  Dep. Var: I2ppegt  |  Within-R\u00b2 for FE models",
)
print_rich_family_table(
    ret_table,
    "Table 2 \u2014 Next-Year Return  |  Dep. Var: ret_A_lead  |  Within-R\u00b2 for FE models",
)

# 5. Export results to Excel
console.rule("[bold gold1] EXPORT [/]")
meta = {
    "Dataset":                           str(data_path),
    "Regression engine":                 "pyfixest feols (reghdfe equivalent)",
    "Total rows after cleaning":         f"{n_after:,}",
    "Exact duplicates dropped":          f"{n_before - n_after:,}",
    "Year range":                        f"{raw['year'].min()}-{raw['year'].max()}",
    "Unique firms":                      f"{raw['permno'].nunique():,}",
    "Investment sample (N)":             f"{inv_n:,}",
    "Investment rows dropped (missing)": f"{inv_dropped:,}",
    "Returns sample (N)":                f"{ret_n:,}",
    "Returns rows dropped (missing)":    f"{ret_dropped:,}",
}
export_results(raw, inv_table, ret_table, meta, OUTPUT_FILE)

sheets_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold", expand=True, padding=(0, 1))
sheets_tbl.add_column("Sheet")
sheets_tbl.add_column("Contents", justify="right")
sheets_tbl.add_row("Cleaned_Data",           f"Panel data: {n_after:,} obs")
sheets_tbl.add_row("Investment_Regressions", "Table 1 \u2014 I2ppegt, 6 specifications")
sheets_tbl.add_row("Returns_Regressions",    "Table 2 \u2014 ret_A_lead, 6 specifications")
sheets_tbl.add_row("Combined_Summary",       "Both families stacked, 6 specifications each")
sheets_tbl.add_row("Metadata",               "Dataset info and sample sizes")
console.print(f"  {OUTPUT_FILE}\n")
console.print(sheets_tbl)
console.print()

if verification_passed:
    console.rule("[bold green] Complete [/]", style="green")
else:
    console.rule("[bold red] Complete - verification failed [/]", style="red")
