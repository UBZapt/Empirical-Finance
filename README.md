# Empirical-Finance

Empirical Finance groupwork, Task 5 - Panel Data Analysis

## Dependencies

Python 3.10+ is required. Install all third-party packages with:

Windows - run in command prompt | Mac - run in terminal:

```bash
pip install numpy pandas pyfixest rich openpyxl
```


| Package    | Purpose                                                          |
| ---------- | ---------------------------------------------------------------- |
| `numpy`    | Numerical operations                                             |
| `pandas`   | DataFrame handling and `.dta` file loading via `read_stata`      |
| `pyfixest` | Panel regression engine (`feols`) — replicates Stata's `reghdfe` |
| `rich`     | Formatted console output (tables, progress)                      |
| `openpyxl` | Excel `.xlsx` export engine used by pandas                       |


`pandas.read_stata` requires no additional package for `.dta` files.

---

## Data

`CCM_sample.dta` is not included in this repository (WRDS/CRSP/Compustat licence prohibits redistribution). Download`CCM_sample.dta` and place it in the same folder as`code.py`. 

## Output

Results are written to `task5_panel_results.xlsx` with three sheets:

- **Investment_Regressions** — Table 1 (six specifications for I2ppegt)
- **Returns_Regressions** — Table 2 (six specifications for ret_A_lead)
- **Metadata** — dataset used, regression engine, row counts, and sample sizes

Both regression tables are also printed to the terminal.

## Assumptions

- **Year column**: If `year` imports as a datetime, the integer year is extracted via `.dt.year`; otherwise the column is stored directly to integer and validated for non-numeric values before proceeding.
- **permno**: Cast from `float64` to `int64`.
- **gpa**: not listed as a regressor in the project brief.
- **Missing values**: Flagged and reported but not dropped or imputed — downstream regression routines handle listwise deletion automatically.
- `ret_a_lead`: Constructed as the within-firm one-period-ahead `ret_a`. The lead return is set to `NaN` wherever the next available firm observation is not exactly `year + 1`, ensuring multi-year lookahead returns are never silently used as the next-year return. 
  - NOTE: The Stata code in week 9 assigns the next row's return regardless of year continuity and can produce multi-year lookaheads which doesn't fit the assignment instructions.

## Regression notes

- Regression engine switched from `linearmodels` to `pyfixest` (`feols()`), which replicates Stata's `reghdfe` including its Frisch–Waugh–Lovell FE absorption and two-way clustering.
- Specs (3)–(6) share the same absorbed FE structure (`| permno + year`); only the covariance estimator differs, so point estimates are identical across those four columns.

## Non-obvious functions / methods

- `validate_and_coerce_permno`: validates `permno` for missing values and fractional parts before `astype(int)`, so failures are caught with a count and a clear message rather than an opaque NumPy error.
- `validate_panel`: checks for duplicate `(permno, year)` keys *after* deduplication and exits with an error before lead-return construction; duplicate keys would silently corrupt `ret_a_lead`.
- `make_lead_return`: computes a `lead_year` alongside the shifted return and sets `ret_a_lead = NaN` where `lead_year != year + 1`, preventing gap-spanning returns from being used as the next-year return.
- `pd.api.types.is_datetime64_any_dtype(df["year"])`: used to branch year conversion safely instead of blindly calling `.dt.year`, which raises `AttributeError` on non-datetime columns.
- `df.groupby("permno")["ret_a"].shift(-1)`: shifts `ret_a` up by one row *within each firm group*, so row `t` receives the return from year `t+1` of the same firm.
- `feols("y ~ x | fe", data=df, vcov=...)`: pyfixest formula interface — the `|` separator introduces the high-dimensional fixed effects to absorb, replicating Stata's `reghdfe y x, absorb(fe)`.
- `vcov={"CRV1": "permno + year"}`: Cameron–Gelbach–Miller two-way clustered sandwich estimator, equivalent to Stata's `cluster(permno year)` option in `reghdfe`; using a `+`-separated formula string activates multi-way clustering.
- `fit._r2_within`: within-R², computed after absorbing fixed effects (on demeaned residuals); reported in the `R² (Within)` row for FE models.
- `fit._r2` / `fit._r2_adj`: overall R² and adjusted overall R², reported for all specifications in the `R² (Overall)` and `Adj. R²` rows respectively.
- `get_r2(fit, r2_type)`: wrapper that selects `_r2_within` for FE models and `_r2` for pooled OLS, with a fallback if `_r2_within` is `None`.
- `df.groupby("permno")["year"].idxmax()`: used in verification to find the last row per firm; retrieves the integer index of the maximum year within each group, which is faster than `apply(lambda s: s.iloc[-1])`.
- `sys.path.pop(0)` guard at the top of the script: pyfixest's dependency chain (`feols → IPython → pdb`) does `import code`; because the script is named `code.py`, Python finds it before the stdlib module and crashes with a circular-import error. Removing the script directory from `sys.path` before pyfixest loads resolves this without side effects, since no local modules are imported.

