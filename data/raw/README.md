# Raw Data

This folder contains the frozen raw inputs for the S&P 500 factor investing project.

Included raw inputs:

- `sp500_prices_long.csv`
- `sp500_fundamentals_daily_long.csv`
- `sp500_fundamentals_statements_long.csv`
- `sp500_constituents.csv`
- `fundamentals_field_definitions.csv`
- `DATA_DESCRIPTION.md`
- `sp500_index_yahoo.csv`

The three stock-level Tiingo CSVs are large, so a normal GitHub repository may require Git LFS or a separate data archive. The project code reads these files directly from this folder and filters all analysis to observations on or before `2026-05-31`.
