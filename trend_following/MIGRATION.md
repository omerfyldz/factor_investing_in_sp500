# Migration: `trend_following/` → standalone repo `omerfyldz/tred_on_etfs`

**Why this exists:** the trend-following strategy was built inside
`factor_investing_in_sp500/trend_following/` because the original session was
locked to that one repo. The goal is to move it into its own repo,
`omerfyldz/tred_on_etfs`, where `trend_following/`'s contents become the repo
root. This doc is the authoritative, self-contained checklist (the session that
runs it is a fresh container with no prior chat history).

**Prerequisite:** run this from a Claude Code session that has BOTH
`omerfyldz/factor_investing_in_sp500` and `omerfyldz/tred_on_etfs` in scope
(add them in the environment settings, then start the session). Verify with
`git ls-remote <proxy>/git/omerfyldz/tred_on_etfs` returning refs (not 403).

## Decisions (confirmed by the user)
- Route: new session with both repos in scope.
- After verification, **delete** `trend_following/` + the two `trend_*`
  workflows from `factor_investing_in_sp500` (true move, not a copy).

## Steps

1. **Clone the target** (it is empty/near-empty):
   `git clone <proxy>/git/omerfyldz/tred_on_etfs`

2. **Copy contents of `trend_following/` into the ROOT of `tred_on_etfs`:**
   `core.py`, `backtest/`, `live/`, `tests/`, `data/` (including the cached
   `prices.csv`), `results/`, `logs/.gitkeep`, `requirements.txt`, `README.md`.
   Do NOT copy this `MIGRATION.md`.

3. **Move the two workflows** to `tred_on_etfs/.github/workflows/` and fix paths
   (they are no longer under a subfolder). In BOTH `trend_backtest.yml` and
   `trend_rebalance.yml`:
   - `pip install -r trend_following/requirements.txt` → `pip install -r requirements.txt`
   - delete every `working-directory: trend_following` line (repo root is now cwd)
   - commit-step `git add` paths lose the `trend_following/` prefix:
     - backtest: `git add results/ data/prices.csv`
     - rebalance: `git add data/portfolio_state.json logs/`
   - No Python changes: `python -m backtest.run_backtest`, `python -m live.run_monthly`,
     and `import core` all resolve from the repo root unchanged.

4. **README:** drop the `trend_following/` prefix in the layout section.

5. **Commit + push** to `tred_on_etfs` `main` (author as the user:
   `--author="omerfyldz <yildiz.omer.faruk1464@gmail.com>"`).

6. **Secrets:** add to the **`tred_on_etfs`** repo: `ALPACA_TREND_KEY`,
   `ALPACA_TREND_SECRET` (separate Alpaca paper account; reset its balance to
   $1,000,000 so the 9 sleeves fit).

7. **Verify** in `tred_on_etfs`: dispatch `trend_backtest.yml`, and
   `trend_rebalance.yml` with `dry_run=true`. Expect the same green results:
   clean backtest (≈2007-06 start, MaxDD ≈ −7.7% vs SPY −50.8%) and a dry-run
   that prints completed-month target weights summing to 1.0.

8. **Cleanup the factor repo** (the actual "move"): delete `trend_following/`
   (including this file) and `.github/workflows/trend_backtest.yml` +
   `.github/workflows/trend_rebalance.yml` from `factor_investing_in_sp500`;
   commit + push. Leave `paper_trading/` and `monthly_rebalance.yml` untouched.

## Sanity checks after migration
- `cd tred_on_etfs && python tests/test_core.py && python tests/test_lookahead.py && python tests/test_signal_parity.py` all pass.
- Repo root contains `core.py` (not nested under `trend_following/`).
- The factor repo no longer references trend following anywhere
  (`grep -ri trend .github/ | wc -l` → 0).
