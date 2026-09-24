# Archived result-directory code

Files under `results/` record historical experiment outputs and the exact scripts copied into those run directories. They are provenance records, not maintained entry points. They may contain stale paths, interfaces, or dependencies and must not be used to regenerate paper claims.

Run maintained commands from `scripts/`, with explicit model, data, and output arguments. A result is usable only after its current validator accepts the complete run artifact. Historical external-baseline outputs that do not preserve one trace identity per ground-truth row require a new run; aggregate values alone are insufficient.
