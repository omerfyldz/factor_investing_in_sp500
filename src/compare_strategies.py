from __future__ import annotations

import pandas as pd

import project_core as core


def main() -> None:
    """Create a clean staged comparison from saved result folders."""
    core.ensure_dirs()
    base = pd.read_csv(core.BASE_RESULTS_DIR / "vector_metrics.csv")
    improved = pd.read_csv(core.IMPROVED_RESULTS_DIR / "vector_metrics.csv")
    improved_2 = pd.read_csv(core.IMPROVED_2_RESULTS_DIR / "vector_metrics.csv")
    improved_3 = pd.read_csv(core.IMPROVED_3_RESULTS_DIR / "vector_metrics.csv")
    comparison = pd.concat(
        [
            base.assign(role="base"),
            improved.assign(role="improved_1"),
            improved_2.assign(role="improved_2"),
            improved_3.assign(role="improved_3"),
        ],
        ignore_index=True,
    )
    core.save_csv(comparison, core.COMPARISON_RESULTS_DIR / "base_vs_improved_metrics.csv")


if __name__ == "__main__":
    main()
