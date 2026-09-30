"""Canonical names for every entry a run's metrics dict can carry.

Several `BaseMetrics` subclasses default to one of these names, and readers
elsewhere (the modifier's prompt-history exclusion list, the report
renderers, trainingLoop's own bookkeeping) key off the same string. Importing
the constant instead of retyping the literal is what keeps a rename from
silently desyncing a reader from the place the name is actually assigned.
"""

accuracy_metrics = "Accuracy-Metrics"
negative_predicted_metrics = "Negative-Predicted-Metrics"
chain_token_usage_metrics = "Chain-Token-Usage-Metrics"
parsing_metrics = "Parsing-Metrics"
modification_metrics = "Modification-Metrics"
text_length_metrics = "TextLength-Metrics"
language_metrics = "Language-Metrics"
labeller_metrics = "Labeller-Metrics"

# Not produced by any BaseMetrics subclass - trainingLoop writes these two
# straight into a run's report data for the modifier LLM's own bookkeeping,
# rather than through ModifierMetrics.get_metrics().
modifier_token_usage_metrics = "Modifier-Token-Usage-Metrics"
modifier_run_info = "Modifier-Run-Info"
