# Empirical-Finance
Empirical Finance groupwork, Task 5 - Panel Data Analysis

## Data

`CCM_sample.dta` is not included in this repository (WRDS/CRSP/Compustat licence prohibits redistribution). Download it from WRDS and place it in the project root before running `code.py`.

## Assumptions

- **Year column**: The raw `.dta` file stores `year` as a datetime (`Jan 1, YYYY`). The integer year is extracted; no data is lost.
- **permno**: Cast from `float64` to `int64`; all values had zero fractional part.
- **gpa**: Present in the raw data but not referenced in the project brief. Retained as-is without modification.
- **Missing values**: Flagged and reported but not dropped or imputed — downstream regression routines handle listwise deletion automatically.
- **Negative `i2ppegt`** (24 rows) and **extreme `bm`/`i2ppegt` outliers**: Flagged in validation output but retained; no outlier removal instruction was given.
- **`ret_a_lead`**: Constructed as the within-firm one-period-ahead `ret_a` (sorted by `permno`, `year`). The final observation for each firm is `NaN` by construction.

## Non-obvious functions / methods

- `groupby("permno")["ret_a_lead"].apply(lambda s: s.iloc[-1])`: retrieves the last observation of `ret_a_lead` per firm to verify it is `NaN` — confirms the lead shift was applied correctly at firm boundaries.
- `df.groupby("permno")["ret_a"].shift(-1)`: shifts `ret_a` up by one row *within each firm group*, so row `t` receives the return from year `t+1` of the same firm.
- `PanelOLS(..., entity_effects=True, time_effects=True)`: linearmodels within estimator that absorbs both firm and year fixed effects via the two-way demeaning transformation; coefficients are identified from within-firm, within-year variation only.
- `mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)`: Cameron–Gelbach–Miller two-way clustered sandwich estimator; called on the same model object as specs 3–6 so coefficients are identical and only standard errors change.
- `res.rsquared_within`: the within-R², computed on the demeaned dependent variable and regressors; reported for all FE specifications in place of the overall R² used for pooled OLS.

## Regression notes

- `gpa` (gross profitability) is present in the dataset and retained in `Cleaned_Data` but is excluded from all regressions — it is not listed as a regressor in the project brief.
- Investment sample: 104,972 observations (listwise deletion on i2ppegt, logme, bm, g\_sale, blev).
- Returns sample: 95,551 observations (listwise deletion on ret\_a\_lead plus the four regressors; the final year per firm is always missing for ret\_a\_lead by construction).
- Specs (3)–(6) use the same fitted model object; only the covariance estimator differs, so coefficients are exactly identical across those four columns.
