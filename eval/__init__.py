"""Cross-method evaluation: metrics, plotting, comparison.

Stateless: every entry point reads run artifacts (run_summary.yaml + npz)
and emits figures or comparison tables. Nothing in this package runs
training or inference.
"""
