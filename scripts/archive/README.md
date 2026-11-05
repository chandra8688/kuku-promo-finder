# Archive Index — One-Off & Internal Diagnostic Scripts

This directory preserves historical, experimental, and diagnostic scripts developed during the project's development, ablation studies, and evaluation experiments. They are retained for evidence and auditability.

## Script Index

| Script | Purpose & Output | Related Report / Deliverable |
|---|---|---|
| `build_final_outputs.py` | Formats final ranked CSVs and JSON manifests from raw episode outputs | Part B Primary Deliverable |
| `compute_cost_metrics.py` | Aggregates token usage, API latency, and estimated cost per episode | Part B Cost & Latency Table |
| `diagnose_gold.py` | Diagnoses gold reference window IoU coverage and gate drop reasons | Finding 4 & 5 Diagnostics |
| `evaluate.py` | Legacy V1 evaluation harness over the initial 7-window gold set | Superseded by `scripts/eval_v2.py` |
| `extract_judged_metrics.py` | Parses judged candidates JSON to compute rubric score distributions | Part B & Part D Analysis |
| `extract_part_e.py` | Extracts ablation table metrics across pipeline stages and gates | Part E Ablations |
| `list_models.py` | Queries live Gemini API endpoints to discover available models and limits | Initial Setup Probe |
| `probe_models.py` | Probes model roles for output length limits and token formatting | Model Configuration Table |
| `report_part_b.py` | Compiles primary pipeline extraction and baseline comparison metrics | Part B Delivery Summary |
| `run_listwise_strong.py` | Runs strong-model (`gemini-3.6-flash`) listwise reranking across top promos | Part F Control / Sensitivity |
| `run_model_swap.py` | Evaluates candidate stability under model substitution | Part G Model Substitution |
| `run_part_d_validity.py` | Evaluates judge ranking validity with swapped candidate pairs | Part D Judge Validity Study |
| `run_part_f_controls.py` | Tests judge position bias and order sensitivity | Part F Sensitivity Controls |
| `run_part_f_judge.py` | Evaluates standalone judge scoring over candidate windows | Part F Judge Ablation |
| `run_part_g_swap.py` | Compiles model-swap candidate overlap and rank correlations | Part G Cross-Model Overlap |
| `run_posthoc_diagnostics.py` | Deep dive into gate drop distributions and survivor counts | Diagnostic Audit Log |
| `run_primary.py` | Batch execution harness for 50-episode primary pipeline run | Primary Show Processing |
| `run_sensitivity_dev.py` | Diagnostic sensitivity tests on dev episode candidate windows | Weight Tuning Diagnostics |
| `run_stability.py` | Runs 3-sample stability tests and computes field agreement metrics | Stability & Consistency Audit |
| `show_clip.py` | CLI viewer to inspect parsed beats and timing for a specific clip | Terminal Debugging Tool |
| `show_window.py` | Detailed window inspector displaying start/end beats and scoring terms | Feature Extraction Debugging |
