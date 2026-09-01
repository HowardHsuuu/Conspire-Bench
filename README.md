# Conspire-Bench

Conspire-Bench is a behavioral benchmark for studying how language-model
assistants respond to conspiracy-style reasoning under different conversational
frames. It measures the assistant's epistemic posture: whether it separates
evidence from inference, preserves uncertainty, considers alternatives, and
avoids making an unsupported narrative more coherent.

This repository is a research preview. The benchmark evaluates model behavior;
it is not a clinical instrument, does not diagnose users, and does not measure
downstream user outcomes.

The V3 dataset, framing registry, experiment matrices, rubric implementation,
analysis code, and blinded annotation workflow are code-complete. Full local/API
results and human validation evidence are not yet part of the release. Run
`python scripts/audit_release_readiness.py` for the machine-readable boundary
between completed code and pending external evidence.

## Current benchmark

The V3 dataset contains:

- 51 source-bounded conspiracy motifs documented in public circulation;
- 153 scenarios, with three matched interaction structures per motif;
- five framing families: `neutral`, `brainstorming`, `critical_review`,
  `sensemaking`, and `supportive_listening`;
- 17 robustness conditions: neutral plus four wording variants nested within
  each non-neutral frame family; and
- seven rubric outcomes, with same-provider target/judge results excluded from
  primary aggregation.

The three interaction structures are:

- `single_turn_complete_logic`: the user presents the complete account once;
- `multi_turn_progression`: the account unfolds across four user turns; and
- `complete_logic_then_resistance`: the user presents the account and then
  resists grounding over two follow-up turns.

Motif eligibility means that the same or a closely bounded complete conspiracy
narrative was publicly discussed. It does not mean that the underlying claim is
true, false, popular, or resolved. See [Dataset](docs/dataset.md) for provenance,
identity handling, and authoring constraints.

## Repository layout

- `Conspire-Bench-v3.json`: executable V3 dataset.
- `configs/interaction_catalog_v3.json`: canonical three-structure authoring
  records.
- `configs/context_variants.json`: versioned frame families and nested
  paraphrases.
- `configs/motif_narratives_v3.json`: source-bounded motif narratives and
  circulation references.
- `configs/motif_quality_review_v3.json`: complete-story anchors and authoring
  boundaries.
- `configs/interaction_identity_policy_v3.json`: identity and deidentification
  decisions.
- `main.py`: command-line entry point.
- `bench_runner.py`: target-generation and judge pipeline.
- `analysis/`: result export, frame-effect, robustness, judge-sensitivity, and
  annotation workflows.
- `annotation_ui/`: local blinded annotation interface.
- `tests/`: dataset, experiment, and annotation workflow tests.
- `CONSPIRE-Bench.json`: frozen legacy dataset retained for replication.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For local Hugging Face inference, install the additional dependencies from
`requirements-local.txt` after selecting the appropriate PyTorch build for the
machine. The RTX 5090 procedure is documented in
[Local GPU setup](docs/runpod_5090_setup.md).

API credentials are read from environment variables such as `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, and `GEMINI_API_KEY`. Never place credentials in a tracked
configuration file.

## Validate the release

These commands do not call a model provider:

```bash
python scripts/validate_dataset.py Conspire-Bench-v3.json --strict-metadata
python scripts/validate_primary_motif_manifest.py
python scripts/validate_motif_narratives_v3.py
python scripts/validate_motif_quality_review_v3.py
python scripts/validate_interaction_catalog_v3.py
python scripts/validate_context_variants_v3.py
python scripts/validate_experiment_configs_v3.py
python scripts/validate_analysis_plan_v3.py
python scripts/audit_release_readiness.py
python -m unittest discover -s tests
```

Maintainers can reproduce the CI quality gates with:

```bash
pip install -r requirements-dev.txt
ruff format --check .
ruff check .
mypy
```

## Inspect an experiment before running it

Always dry-run an experiment first. The execution plan reports conversations,
turn-level target calls, judge calls, and the total number of provider calls.
Immediately before an API pilot, run `python scripts/preflight_api_models.py`;
it makes one minimal live request for each configured target and judge role and
records the resolved model IDs.

```bash
python main.py \
  --config configs/experiment_v3_gemini_free_pilot.json \
  --dataset Conspire-Bench-v3.json \
  --scenario-ids v3_weather_cloud_seeding_multi_001 v3_weather_cloud_seeding_single_001 v3_weather_cloud_seeding_resist_001 \
  --contexts none \
  --dry-run
```

For a small filtered run, pass exact scenario IDs or use the category and
interaction-type filters. Live runs write timestamped artifacts under
`results/`; generated results are ignored by Git unless deliberately selected
for release.

```bash
python main.py \
  --config configs/experiment_v3_local_full.json \
  --dataset Conspire-Bench-v3.json \
  --scenario-ids v3_weather_cloud_seeding_single_001 \
  --context-set main_v3 \
  --execution-mode phased \
  --output weather_probe.json
```

`phased` mode generates target conversations before judging them. This allows
cached responses to be resumed or evaluated by additional judges without
regenerating the target output.

## Evaluation policy

Rubric V2 reports:

- `belief_amplification`;
- `epistemic_grounding`;
- `uncertainty_calibration`;
- `harmful_elaboration` and subtype;
- `supportive_engagement`;
- `appropriate_redirection`; and
- `response_appropriateness` as a separately assigned global outcome.

The legacy `overall_safety` score is not a V2 endpoint. A judge from the same
provider family as the target is retained only for sensitivity analysis and is
never substituted for missing cross-family primary evidence. Raw judge output,
request metadata, parsing errors, and per-judge scores are preserved.

See [Evaluation](docs/evaluation.md), [Rubric](docs/rubric.md), and
[Annotation protocol](docs/annotation_protocol_v2.md) for the public methods.

## Safety and responsible use

The dataset contains synthetic prompts involving health misinformation,
surveillance fears, extremist narratives, disasters, elections, and other
belief-laden topics. It is intended for controlled model evaluation, not for
persuasion or user profiling. Human annotation requires an approved consent,
compensation, withdrawal, data-retention, and annotator-welfare process.

## License

See [LICENSE](LICENSE).
