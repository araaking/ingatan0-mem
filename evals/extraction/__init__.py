"""The extraction eval: how much of what a model extracts is correct, and how much invented.

Run with `python -m evals.extraction`. It needs the model configured under `[llm]`, so CI
does not run it; `tests/evals/test_extraction_metrics.py` covers the judging without one.
"""

from __future__ import annotations
