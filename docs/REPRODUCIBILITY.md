# Reproducibility

## One-time setup

```powershell
git clone https://github.com/omerfyldz/factor_investing_in_sp500.git
cd factor_investing_in_sp500
py -3.10 -m pip install -r requirements.txt
git lfs install
git lfs pull       # materialize the large raw CSVs
```

## Full pipeline (slow — about 56 minutes)

```powershell
py -3.10 src\run_project.py
```

Rebuilds the processed factor panel, runs base + improveds 1-3 (vector + Backtrader), runs FMP analysis, computes Monte Carlo + block bootstrap + walk-forward + benchmark comparison, regenerates all figures, regenerates the PDF presentation, and rewrites the strategy history doc.

## Focused per-strategy reruns (faster)

Each focused script reads the already-processed factor panel and reruns only its strategy. Useful for iterating on one strategy without the full pipeline cost.

```powershell
py -3.10 src\run_improved_4_stop_take_sensitivity.py     # ~36 min (6x6 grid)
py -3.10 src\run_improved_5_regime_filter.py              # ~21 min
py -3.10 src\run_improved_6_hzz_trend.py                  # ~25 min
py -3.10 src\run_improved_7_costs.py                      # ~5 min (vector only)
py -3.10 src\run_improved_8_top_n_sizing.py               # ~25 min
```

## After code changes

```powershell
py -3.10 -m py_compile src\project_core.py src\run_project.py ^
    src\run_improved_4_stop_take_sensitivity.py ^
    src\run_improved_5_regime_filter.py ^
    src\run_improved_6_hzz_trend.py ^
    src\run_improved_7_costs.py ^
    src\run_improved_8_top_n_sizing.py

py -3.10 -c "import sys; sys.path.insert(0, 'src'); import project_core; print('import OK')"
```

## Reproducibility guarantees

- Raw data CSVs are frozen and committed via Git LFS.
- The Yahoo `^GSPC` download is auto-skipped if `data/raw/sp500_index_yahoo.csv` already covers `>= 2026-05-29`.
- All random-number generator seeds are fixed (`RNG_SEED = 5811` in `project_core.py`).
- All cutoff dates are constants (`CUTOFF = 2026-05-31`, `EVALUATION_START = 2016-05-31`).
- All hyperparameters are constants in source code.
- Processed-data outputs are deterministic given the raw inputs.

Running the pipeline today on the same machine should produce bit-identical CSV and PDF outputs.

## Expected regenerated folders

- `data/processed/`
- `results/base_strategy/`, `results/improved_strategy/`, `results/improved_strategy_2/3/4/5/6/7/8/`
- `results/comparison/`, `results/fmp_analysis/`
- `figures/`
- `presentation/`
- `docs/` (auto-regenerated: `STRATEGY_HISTORY.md`, `PROJECT_REPORT.md`)
