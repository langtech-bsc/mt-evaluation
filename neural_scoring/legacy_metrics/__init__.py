"""Expose the original metric directory without importing the lm_eval runtime."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / "lm_eval" / "extra_metrics")]
