# Reproducibility

Run from a fresh Python session:

```powershell
cd C:\Users\asus\Desktop\sp500_factor_investing
py -3.10 -m pip install -r requirements.txt
py -3.10 src\run_project.py
```

The code reads frozen CSV files from `data/raw/`. It does not redownload stock data. It downloads `^GSPC` only if `data/raw/sp500_index_yahoo.csv` is missing or stale; after that the benchmark is frozen and reused.

Expected regenerated folders:

- `data/processed/`
- `results/`
- `figures/`
- `presentation/`
