# Interval-representation measurements

JSON benchmark reports for `CL-overtake` on the first 100 merged CLEVRER videos
(`clevrer_merged/video_00000+99more`). Both builds use `max_gap=1`, zero fitting
error, replication factor 1, and three query executions:

- Compact: `--features instrument`
- Enumerated: `--features instrument,naive-intervals`

Both return 222 tuples with identical pair sets. The smaller dataset is used
because the enumerated build exceeded 75 GB RSS on the full dataset before
being stopped.

Render with `scripts/interval_ablation.py` from `plotter/`; see the
[plotter README](../../README.md).
