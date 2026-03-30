# Empirical-Finance
Empirical Finance groupwork, Task 5 - Panel Data Analysis

## Data

`CCM_sample.dta` is not included in this repository (WRDS/CRSP/Compustat licence prohibits redistribution). Download it from WRDS and place it in the project root before running `code.py`. The loader also accepts `CCM sample.dta` (with a space) — whichever exists is used; if neither is found, the script exits with a clear error.

## Output

Results are written to `task5_panel_results.xlsx` with four sheets:
- **Cleaned_Data** — the cleaned panel used for all regressions
- **Investment_Regressions** — Table 1 (six specifications for I2ppegt)
- **Returns_Regressions** — Table 2 (six specifications for ret_A_lead)
- **Metadata** — dataset used, row counts, and sample sizes for each regression family

## Assumptions

- **Year column**: If `year` imports as a datetime (Stata stores annual dates as Jan 1 of that year), the integer year is extracted via `.dt.year`; otherwise the column is coerced directly to integer and validated for non-numeric values before proceeding.
- **permno**: Cast from `float64` to `int64`; all values had zero fractional part.
- **gpa**: Present in the raw data and retained in `Cleaned_Data` but excluded from `REQUIRED_COLS` and all regressions — it is not listed as a regressor in the project brief.
- **Missing values**: Flagged and reported but not dropped or imputed — downstream regression routines handle listwise deletion automatically.
- **Negative `i2ppegt`** and **extreme `bm`/`i2ppegt` outliers**: Flagged in validation output but retained; no outlier removal instruction was given. Thresholds used — `i2ppegt < 0` (investment rates cannot be negative under standard COMPUSTAT construction; any negative value signals a data error or unusual event such as asset disposal exceeding gross investment) and `i2ppegt > 10` (a ratio above 10× implies gross investment exceeded ten times lagged capital, which is implausible for an ongoing firm and almost always indicates a near-zero denominator); `|bm| > 10` (book-to-market ratios of this magnitude arise almost exclusively from firms with near-zero or negative book equity, not genuine value characteristics, and are standard screen thresholds in the academic literature). All three flags follow the conservative convention of identifying likely data artefacts while leaving the analyst to decide on removal.
- **`ret_a_lead`**: Constructed as the within-firm one-period-ahead `ret_a` (sorted by `permno`, `year`). Panel key uniqueness is validated before this step to prevent silent corruption. The final observation for each firm is `NaN` by construction.

## Non-obvious functions / methods

- `validate_panel`: checks for duplicate `(permno, year)` keys and exits with an error before lead-return construction; duplicate keys would silently corrupt `ret_a_lead`.
- `pd.api.types.is_datetime64_any_dtype(df["year"])`: used to branch year conversion safely instead of blindly calling `.dt.year`, which raises `AttributeError` on non-datetime columns.
- `df.groupby("permno")["ret_a"].shift(-1)`: shifts `ret_a` up by one row *within each firm group*, so row `t` receives the return from year `t+1` of the same firm.
- `PanelOLS(..., entity_effects=True, time_effects=True)`: linearmodels within estimator absorbing both firm and year fixed effects via two-way demeaning; coefficients identified from within-firm, within-year variation only.
- `mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)`: Cameron–Gelbach–Miller two-way clustered sandwich estimator; called on the same model object as specs 3–6 so coefficients are identical and only standard errors change.
- `res.rsquared_within`: within-R², computed on the demeaned dependent variable and regressors; reported for all FE specifications in place of the overall R² used for pooled OLS.

## Regression notes

- `gpa` (gross profitability) is present in the dataset and retained in `Cleaned_Data` but excluded from all regressions — it is not listed as a regressor in the project brief.
- Specs (3)–(6) use the same fitted model object; only the covariance estimator differs, so coefficients are exactly identical across those four columns.
