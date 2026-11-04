# Recorded inputs

These are the recorded inputs used to produce the paper's figures.
They are independent of newly generated runs in
`results/suite/`.

| Input | Used by |
|---|---|
| `results-dedup.json` | Compression comparison's deduplicated DuckDB baseline |
| `results-rekall.json` | Rekall comparison |
| `vlm/results-vlm*.json`, `vlm/ingest.json` | VLM latency, accuracy, and amortization plots |
| `cooccurrence_stats.json` | Co-occurrence plot |
| `ablation/*.db` | Polynomial-degree ablation: clean runs |
| `ablation-instrumented/*.db` | Polynomial-degree ablation: stage proportions |
| `interval-ablation/*.json` | Compact versus naive interval-set ablation |
| `miris_timeline_plot.pdf` | Preserved MIRIS comparison, Figure 7 |

The MIRIS PDF is copied unchanged; its generator and source measurements are
not included. Other figures use the root `bench.db` or the inputs above.
See the [plotter README](../README.md) for rendering and selecting new inputs,
and the [VLM README](../../suite/vlm/README.md) for ingestion-cost assumptions.
