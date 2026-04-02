# Empirical-Finance
Empirical Finance groupwork, Task 5 - Panel Data Analysis

## Data

`CCM_sample.dta` is not included in this repository (WRDS/CRSP/Compustat licence prohibits redistribution). Download it from WRDS and place it in the project root before running `code.py`. The loader also accepts `CCM sample.dta` (with a space) — whichever exists is used; if neither is found, the script exits with a clear error.

## Output

Results are written to `task5_panel_results.xlsx` with three sheets:
- **Investment_Regressions** — Table 1 (six specifications for I2ppegt)
- **Returns_Regressions** — Table 2 (six specifications for ret_A_lead)
- **Metadata** — dataset used, regression engine, row counts, and sample sizes

Both regression tables are also printed to the terminal as Rich console tables.

## Assumptions

- **Year column**: If `year` imports as a datetime (Stata stores annual dates as Jan 1 of that year), the integer year is extracted via `.dt.year`; otherwise the column is coerced directly to integer and validated for non-numeric values before proceeding.
- **permno**: Cast from `float64` to `int64`; all values had zero fractional part.
- **gpa**: Present in the raw data and retained in `Cleaned_Data` but excluded from `REQUIRED_COLS` and all regressions — it is not listed as a regressor in the project brief.
- **Missing values**: Flagged and reported but not dropped or imputed — downstream regression routines handle listwise deletion automatically.
- **`ret_a_lead`**: Constructed as the within-firm one-period-ahead `ret_a` (sorted by `permno`, `year`). The lead return is set to `NaN` wherever the next available firm observation is not exactly `year + 1`, ensuring gap-spanning returns are never silently used as the next-year return. The final observation for each firm is `NaN` by construction. This differs from Stata's `by permno: gen ret_A_lead = ret_A[_n+1]`, which assigns the physically next row's return regardless of year continuity and can produce multi-year lookaheads in unbalanced panels. The resulting sample for `ret_A_lead` regressions is therefore smaller than the Stata baseline: the exact breakdown between boundary NaNs (last obs per firm) and gap NaNs (year discontinuity) is printed to the console and recorded in the Excel Metadata sheet.
- **Duplicate (permno, year) keys**: Reported as a data-integrity diagnostic before `drop_duplicates()` is called. Exact duplicate rows are then removed; any duplicate panel keys that survive deduplication (non-identical rows sharing the same key) cause the script to exit with an error.
- **`permno` validation**: Missing values and non-integer fractional parts are checked explicitly before the `astype(int)` coercion; the script exits with a count and clear message if either condition is found.

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
- `print_rich_family_table`: renders the summary DataFrame as a Rich console table with a section divider separating coefficients from model-level statistics.
- `df.groupby("permno")["year"].idxmax()`: used in verification to find the last row per firm; retrieves the integer index of the maximum year within each group, which is faster than `apply(lambda s: s.iloc[-1])`.
- `warnings.filterwarnings("ignore", message=".*singleton fixed effects.*")`: suppresses pyfixest's `UserWarning` about singleton groups dropped during FE absorption; these are expected in unbalanced panels and do not affect estimates.
- `sys.path.pop(0)` guard at the top of the script: pyfixest's dependency chain (`feols → IPython → pdb`) does `import code`; because the script is named `code.py`, Python finds it before the stdlib module and crashes with a circular-import error. Removing the script directory from `sys.path` before pyfixest loads resolves this without side effects, since no local modules are imported.

## Regression notes

- Regression engine switched from `linearmodels` to `pyfixest` (`feols()`), which replicates Stata's `reghdfe` including its Frisch–Waugh–Lovell FE absorption and two-way clustering.
- `gpa` (gross profitability) is present in the dataset and retained in `Cleaned_Data` but excluded from all regressions — it is not listed as a regressor in the project brief.
- Specs (3)–(6) share the same absorbed FE structure (`| permno + year`); only the covariance estimator differs, so point estimates are identical across those four columns.
- pyfixest takes a flat (non-indexed) DataFrame; `permno` and `year` are referenced by column name in the formula rather than used as a MultiIndex.
