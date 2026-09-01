# Conspire-Bench

Conspire-Bench is a behavioral benchmark for characterizing how language-model assistants respond to conspiracy-style reasoning under different interaction frames. It focuses on the assistant's epistemic posture: whether a model grounds claims in evidence, preserves uncertainty, considers alternatives, and avoids turning unsupported suspicions into a more coherent narrative.

The benchmark is designed for research on open-ended assistant behavior. It is not a clinical instrument, does not diagnose users, and does not measure downstream user outcomes.

## Benchmark Design

The frozen legacy set contains 24 scenario instantiations built from:

- 8 domain motifs: aliens/UFOs, modern health, government control, AI consciousness, secret societies, religious apocalyptic reasoning, technology control, and mixed-theory narratives.
- 3 conversation structures per domain:
  - `single_turn`: one complete suspicious argument in a single user turn.
  - `multi_turn`: belief escalation across four user turns.
  - `resistance`: an initial suspicious argument followed by two user resistance turns.
- 3 interaction frames:
  - `none`: no additional framing.
  - `brainstorming`: invites exploratory hypothesis generation.
  - `critical_review`: asks the assistant to stress-test claims.

The scenarios intentionally mix verifiable fragments, ambiguous observations, and unsupported causal leaps. This mirrors a common structure of conspiracy-style reasoning: speculative conclusions become more plausible when embedded in real events or unresolved anomalies.

The source-reviewed v2 development catalog adds 21 contemporary motifs, three conspiratorial interaction structures per motif, and one matched non-conspiratorial control per motif. Together with the legacy set it materializes 108 scenarios.

The v3 dataset redesign is currently at the **motif-selection stage**. Its candidate pool contains 51 complete conspiracy narratives whose public circulation is source-bounded; 30 form a quality-based proposed primary set, not a frozen selection. V3 uses five frame families—`neutral`, `brainstorming`, `critical_review`, `sensemaking`, and `supportive_listening`—with paraphrases nested inside their frame family. The three matched interaction structures will be authored only after the final motif IDs are approved.

## Evaluation Rubric

Rubric v2 uses six observable dimensions: `belief_amplification`, `epistemic_grounding`, `uncertainty_calibration`, binary-plus-subtype `harmful_elaboration`, `supportive_engagement`, and `appropriate_redirection`. The optional 0–2 `response_appropriateness` field is secondary. `overall_safety` remains readable only for legacy v1 results and is not a v2 headline endpoint.

Set `rubric_version: "2.0"` on each judge (or `evaluation.judge_rubric_version`). Strong non-overlapping judges form the primary aggregate; same-provider target/judge scores remain in `judge_results` for the sensitivity analysis.

## Repository Layout

- `CONSPIRE-Bench.json`: benchmark scenarios.
- `configs/scenario_expansion_v2.json`: versioned v2 prompt catalog and matched controls.
- `configs/scenario_source_packets_v2.json`: dated claim-boundary and source review packets.
- `configs/scenario_ai_author_review_v2.json`: digest-bound experiment approval from the AI-assisted author-side scenario QA; explicitly not independent human review.
- `configs/motif_selection_evidence_v2.json`: dated evidence for narrative circulation versus construct-only inclusion.
- `configs/context_variants.json`: canonical frames, three paraphrases per non-neutral main frame, and 12 exploratory elicitation conditions.
- `configs/primary_motif_manifest_v3.json`: proposed 30-item primary set, 21 eligible alternatives, and the semantic-QC exclusion ledger.
- `configs/motif_narratives_v3.json`: actor/action/goal/mechanism/concealment records and dated circulation sources for all 51 eligible v3 motifs.
- `configs/motif_quality_review_v3.json`: complete-story anchors, authoring boundaries, overlap, sensitivity, and fictionalization requirements.
- `configs/motif_selection_recommendation_v3.json`: advisory 30/14/3/4 partition pending user approval.
- `docs/motif_circulation_audit_v3.md`: generated complete-story/source-boundary audit; do not hand-edit it.
- `docs/motif_pool_coverage_v3.md`: generated category, sensitivity, and experiment-size coverage audit; do not hand-edit it.
- `docs/framing_taxonomy_v2.md`: mechanism-based rationale and analysis boundary for the exploratory framing set.
- `docs/scenario_ai_author_audit_v2.md`: motif-level findings and residual cautions from the current scenario QA.
- `docs/completion_audit_v2.md`: requirement-by-requirement implementation status, remaining evidence, and annotation timing.
- `docs/running_example_v2.md`: manuscript-ready construction/frame/example template with result fields intentionally left pending until freeze.
- `rubric_v2.py` and `docs/rubric.md`: shared rubric identifiers plus the current expert-facing definitions, anchors, literature rationale, and legacy boundary.
- `main.py`: evaluation entry point.
- `bench_runner.py`: standard and phased evaluation runner.
- `local_models.py`: local Hugging Face model loading and generation.
- `configs/`: API and local-model evaluation configs.
- `configs/models/`: per-model Hugging Face loading configs.
- `analysis/export_results.py`: exports analysis-ready CSV tables.
- `analysis/export_annotations.py`: creates blinded expert and student packages with a formal-release gate.
- `analysis/export_calibration.py`: selects a deterministic, behaviorally diverse calibration set and freezes response IDs that formal annotation must exclude.
- `analysis/assign_annotations.py`: freezes balanced item–annotator assignments, writes pseudonym-prefilled per-annotator JSONL/CSV packages, and records a digest-bound private ledger.
- `analysis/freeze_annotator_roster.py`: computes stage-specific expert/student eligibility from a pseudonymous roster and binds formal assignments to its digest.
- `analysis/scenario_review.py`: exports independent scenario/source review forms and validates the returned approval ledger.
- `analysis/import_annotations.py`: validates returned annotations; reports true overlap, missingness, pairwise agreement with bootstrap intervals, stratum-specific expert results, decoded student A/B preferences and order diagnostics, and optional human-versus-judge validity.
- `analysis/import_rubric_feedback.py`: validates expert content-validity forms and reports per-dimension ratings, recommendation counts, descriptive I-CVI/S-CVI summaries, and qualitative revision requests.
- `annotation_ui/`: dependency-free local blinded rating interface with browser progress saving and importer-compatible JSONL export.
- `annotation/rubric_validity_v2/`: ready-to-copy JSON/CSV content-validity packet for the two or three clinical/domain experts; it can be sent before model generation is complete.
- `annotation/rubric_freeze_decision_template.private.json`: calibration decision record required, with content-validity and calibration manifests, before the analysis plan can be frozen.
- `docs/expert_invitation_template.md`: fill-in invitation plus the supervisor/ethics/compensation checklist that must be resolved before expert recruitment.
- `analysis/judge_family_sensitivity.py`: compares same-family and non-overlapping judge scores.
- `analysis/paraphrase_robustness.py`: reports within-frame variance across prompt rephrasings.
- `analysis/control_pair_validity.py`: tests rubric discrimination on matched conspiratorial/control pairs.
- `analysis/rubric_consistency_audit.py`: reports post-hoc cross-dimension tensions without changing judge scores.
- `analysis/api_usage_report.py`: totals provider-recorded target/judge requests, resolved models, interfaces, and token usage.
- `analysis/frame_effect_stats.py`: runs the canonical paired contrasts with scenario-clustered confidence intervals.
- `analysis/validate_analysis_plan.py`: checks that the frozen subset, contexts, model matrices, judges, and dataset agree.
- `analysis/freeze_analysis_plan.py`: refuses a dirty worktree and creates a digest-bound frozen plan only after expert content-validity and calibration evidence passes validation.
- `configs/human_annotation_plan_v2.json`: authoritative pre-outcome workload plan: 16 expert calibration items, 72 formal expert conversations rated twice, and 108 student A/B pairs rated three times.
- `analysis/freeze_human_annotation_plan.py`: freezes those counts only after a timed expert calibration and student-pair UI pilot, with approver and feasibility records.
- `analysis/critical_review_paired_audit.csv`: paired neutral-vs-critical-review qualitative audit used for the paper's critical-review analysis.
- `analysis/critical_review_paired_audit_note.md`: codebook and summary for the paired audit.
- `docs/`: setup, rubric, and local-model notes.
- `ConspireBench_paper/latex/`: paper source and compiled PDF.
- `ConspireBench_paper/latex/v2_methods_draft.tex`: result-free v2 Methods text ready for integration after rubric freeze.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

For RunPod RTX 5090 runs, see [docs/runpod_5090_setup.md](docs/runpod_5090_setup.md). The short version is: create a Python 3.10 environment, install a CUDA 12.8-compatible PyTorch build first, then install `requirements.txt`.

## Quick Start

Validate the v3 candidate pool and confirm that the generated audit matches the canonical records:

```bash
python scripts/validate_primary_motif_manifest.py
python scripts/validate_motif_narratives_v3.py
python scripts/validate_motif_quality_review_v3.py
python scripts/validate_motif_selection_recommendation_v3.py
python scripts/render_motif_circulation_audit_v3.py --check
python scripts/render_motif_pool_coverage_v3.py --check
python -m unittest tests.test_v3_motif_pipeline
```

To regenerate the human-readable audit after an intentional source or boundary change:

```bash
python scripts/render_motif_circulation_audit_v3.py --write
python scripts/render_motif_pool_coverage_v3.py --write
```

Do not build the v3 interaction catalog until `selection_state` is deliberately changed to `frozen_primary` after the final motif selection.

For the currently runnable legacy and v2 datasets:

Validate the dataset and inspect a run plan without model calls:

```bash
python scripts/validate_dataset.py CONSPIRE-Bench.json
python main.py --config configs/local_5090_config.json --dry-run --types single_turn
```

Reviewer-oriented API pilot (three target families and three strong judges):

```bash
python main.py --config configs/experiment_v2_api_pilot.json \
  --dataset configs/scenario_expansion_v2.json \
  --dry-run --strict-dataset-metadata --context-set main_v2 \
  --scenario-ids v2_ai_coded_message_single_001 ufo_resist_001 v2_pre_event_search_foreknowledge_multi_001 v2_cheapfake_wrong_context_resist_001 v2_medbed_suppression_single_001 religious_multi_001 secret_resist_001 v2_cbdc_population_control_single_001 v2_ai_coded_message_multi_001 ufo_multi_001 v2_weather_cloud_seeding_resist_001 v2_conflict_staged_footage_single_001
```

Those IDs are frozen in `configs/pilot_subset_v2_seed11.json`. The 12-scenario calibration produces 108 target responses and 324 judge calls; it is excluded from headline estimates.

The stage wrapper reads the frozen manifests so IDs do not need to be copied manually:

```bash
python scripts/run_v2_stage.py pilot --dry-run
python scripts/run_v2_stage.py main --dry-run
python scripts/run_v2_stage.py robustness --dry-run
python scripts/run_v2_stage.py exploratory --dry-run
```

For a live run, use phased or generation-only execution and preserve the timestamped result for `--resume-from`; the wrapper checks API keys before making calls.

Before the first paid pilot, verify every model/parameter combination with one tiny request and save the resolved model IDs:

```bash
python scripts/preflight_api_models.py \
  --config configs/experiment_v2_api_full.json \
  --output results/api_model_preflight.json
```

The larger nine-target matrix is `configs/experiment_v2_api_full.json`. The six-target, two-tier paraphrase matrix is `configs/experiment_v2_api_robustness.json`. Always dry-run either matrix and estimate cost before removing scenario filters.

Run a small local smoke test:

```bash
bash scripts/runpod_5090_smoke.sh
```

Manual equivalent:

```bash
python main.py \
  --config configs/local_5090_config.json \
  --categories aliens_ufo \
  --types single_turn \
  --max-per-category 1 \
  --output local_calibration.json
```

Run a full interaction-frame sweep with local models:

```bash
python main.py \
  --config configs/local_5090_config.json \
  --context-set main_v2 \
  --output local_seed24_contexts.json
```

Use phased execution to generate all target conversations first, then run judges over cached conversations:

```bash
python main.py \
  --config configs/local_5090_config.json \
  --execution-mode phased \
  --contexts none brainstorming critical_review \
  --output local_seed24_contexts.json
```

Resume a phased or standard run from previous successful rows:

```bash
python main.py \
  --config configs/local_5090_config.json \
  --execution-mode phased \
  --contexts none brainstorming critical_review \
  --resume-from results/<timestamp>/local_seed24_contexts.json \
  --output local_seed24_contexts_resumed.json
```

Export analysis-ready tables:

```bash
python analysis/export_results.py results/<timestamp>/local_seed24_contexts.json
```

This writes CSV files such as `model_summary.csv`, `scenario_results.csv`, `judge_scores.csv`, and `interesting_cases.csv` under the run directory.

## Common CLI Options

- `--config`: path to an evaluation config.
- `--dataset`: raw dataset JSON or an expansion catalog such as `configs/scenario_expansion_v2.json`.
- `--categories`: filter domains, e.g. `aliens_ufo modern_health`.
- `--types`: filter scenario structures: `single_turn`, `multi_turn`, `resistance`.
- `--context`: run one interaction frame.
- `--contexts`: run multiple interaction frames in one result file.
- `--context-set`: run a versioned set from `configs/context_variants.json`.
- `--execution-mode`: `standard`, `phased`, `generation-only`, or `judge-only`.

For the reviewer-driven design, model matrix, rubric, and annotation release sequence, see [docs/experiment_v2_spec.md](docs/experiment_v2_spec.md), [docs/annotation_protocol_v2.md](docs/annotation_protocol_v2.md), and [docs/completion_audit_v2.md](docs/completion_audit_v2.md). The current expansion ledger is [docs/scenario_expansion_candidates.md](docs/scenario_expansion_candidates.md).

Validate or dry-run the 108-scenario source-reviewed v2 draft:

```bash
python scripts/validate_dataset.py configs/scenario_expansion_v2.json --strict-metadata
python analysis/audit_scenario_catalog.py configs/scenario_expansion_v2.json
python main.py --config configs/experiment_v2_api_pilot.json \
  --dataset configs/scenario_expansion_v2.json \
  --context-set main_v2 --dry-run --strict-dataset-metadata
```

The catalog materializes the frozen legacy 24 plus 63 new conspiratorial scenarios and 21 matched non-conspiratorial controls at load time. Its new items are marked `source_reviewed` and `ai_author_reviewed` through a digest-bound ledger. This is sufficient for experiment execution and later conversation annotation, but it is not evidence of independent human scenario review. After a run, `analysis/control_pair_validity.py` measures whether rubric scores discriminate each conspiratorial item from its paired control.

An independent scenario-review package remains available as an optional strengthening study. If completed, its ledger can replace the `review_approval` path in the catalog and will change expansion items to `verified`/`approved` without hand-editing scenario metadata:

```bash
python analysis/scenario_review.py export \
  --catalog configs/scenario_expansion_v2.json \
  --output-dir review/scenarios
python analysis/scenario_review.py import \
  --catalog configs/scenario_expansion_v2.json \
  --review review/scenarios/returned_reviewer_01.csv \
  --output configs/scenario_review_approval_v2.json
```

Validate the preregistration draft and reproduce the already selected robustness subset:

```bash
python analysis/validate_analysis_plan.py configs/analysis_plan_v2.json
python analysis/select_scenario_subset.py configs/scenario_expansion_v2.json \
  --count 32 --seed 24 --output /tmp/robustness_subset_v2_seed24.json
diff configs/robustness_subset_v2_seed24.json /tmp/robustness_subset_v2_seed24.json
```
- `--resume-from`: skip successful matching rows from an existing result JSON.
- `--dry-run`: validate setup and print planned model, scenario, and judge counts.
- `--validate-only`: validate config, dataset, and local model paths.
- `--status-file`: write per-scenario status rows to a TSV file.
- `--output`: output JSON filename within the timestamped results directory.

Run `python main.py --help` for the full option list.

## Outputs

Each run writes to `results/<timestamp>/`. Typical files include:

- `<output>.json`: final result file.
- `temp_<output>.json`: incrementally updated temporary result file.
- `status.tsv`: per-scenario execution status and runtime.
- `analysis_tables/` or exported CSVs: summary tables for analysis.

Rows may contain partial judge failures when one judge returns unparseable output. The runner aggregates over valid judge outputs and records the failed judge in the row.

## Paper

The paper source is in [ConspireBench_paper/latex](ConspireBench_paper/latex). To compile:

```bash
cd ConspireBench_paper/latex
latexmk -pdf -interaction=nonstopmode acl_latex.tex
```

The current compiled draft is [ConspireBench_paper/latex/acl_latex.pdf](ConspireBench_paper/latex/acl_latex.pdf).

## Scope and Safety

Conspire-Bench contains sensitive prompts involving health misinformation, surveillance fears, religious apocalyptic reasoning, AI consciousness claims, and related belief-laden themes. The prompts are synthetic and intended for evaluating assistant behavior, not for persuading users or studying user psychology.

Use the benchmark as a behavioral characterization tool. Human validation and downstream user-outcome studies are required before making claims about clinical safety or effects on vulnerable users.
