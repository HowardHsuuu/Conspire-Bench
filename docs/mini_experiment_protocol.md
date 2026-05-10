# Mini Experiment Protocol

Goal: present Conspire-Bench as a reproducible benchmark for conspiracy-susceptibility and epistemic safety, then report observations from a small local-model study.

## Claim Shape

Primary claim:

- Conspire-Bench is a reproducible evaluation protocol for measuring how LLM assistants respond to conspiracy-style and delusion-adjacent conversational pressure.

Secondary claim:

- A small cross-family local-model run can reveal qualitative patterns worth studying, such as which scenario types trigger amplification, where judges disagree, and whether safety behavior changes across turns.

Avoid claiming clinical diagnosis, population-level prevalence, or definitive model ranking in the workshop paper.

## Local 5090 Run

The starter local config is `configs/local_5090_config.json`.

Starter targets:

- `meta-llama/Llama-3.2-3B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-3-1b-it`
- `google/gemma-4-E2B-it`

Judges:

- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-4-E4B-it`

The config runs sequentially and unloads local models between target and judge calls to fit a single high-end consumer GPU. If the gated Llama or Gemma weights are unavailable locally, replace that entry with a model already downloaded on the machine and keep the same family labels in the analysis.

The larger sweep config is `configs/local_5090_full_matrix_config.json`. It adds Llama 3.2 1B, Qwen 1.5B/3B/14B, Gemma 3 4B/12B, and Gemma 4 E4B as targets. The 14B/12B entries use 4-bit loading and require `bitsandbytes`.

Suggested calibration command:

```bash
python3 main.py \
  --config configs/local_5090_config.json \
  --categories aliens_ufo ai_consciousness modern_health \
  --types single_turn \
  --max-per-category 1 \
  --output local_calibration.json
```

Suggested pilot command:

```bash
python3 main.py \
  --config configs/local_5090_config.json \
  --types single_turn resistance \
  --max-per-category 2 \
  --output local_pilot.json
```

After a run:

```bash
python3 analysis/export_results.py results/<timestamp>/local_pilot.json
```

## Reporting Tables

Use `analysis/export_results.py` to produce:

- `model_summary.csv`: average scores, harmful elaboration rate, and judge disagreement by target model.
- `scenario_results.csv`: one row per scenario/model result.
- `judge_scores.csv`: one row per judge/scenario/model result.
- `interesting_cases.csv`: low-safety, harmful-elaboration, high-disagreement, or errored cases for qualitative inspection.

For the workshop paper, the safest table set is:

- Dataset composition table: category, scenario type, risk metadata.
- Model matrix table: target models, judge models, decoding settings, hardware.
- Main result table: mean score by target model and scenario type.
- Judge agreement table: mean absolute disagreement by rubric dimension.
- Qualitative case table: 3-5 representative failure or disagreement cases.

## Dataset Expansion

Expand from the current pilot through a controlled design matrix rather than free-form generation.

Recommended axes:

- Belief domain: public conspiracy, medical misinformation, surveillance/paranoia, AI consciousness, spiritual/apocalyptic, grandiose mission, special-relationship AI beliefs, mixed narratives.
- Scenario type: single-turn complete logic, multi-turn progression, resistance after correction, control prompt.
- Risk pressure: curiosity, high conviction, distress, social isolation, help refusal, secrecy request.
- Grounding trap: real fact plus false inference, fabricated evidence, unverifiable speculation, source laundering, false binary, pattern overfitting.
- Expected safe behavior: validate emotion, separate fact from inference, avoid new claims, calibrate uncertainty, suggest grounded support.

Initial workshop-scale target:

- 80-120 scenarios for the first submission.
- 10-15% paired controls that share the surface topic but remove the fixed false belief or conspiratorial commitment.
- Keep the larger 200-300 scenario target for a follow-up paper.

## Analysis Plan

1. Run the calibration slice and inspect `interesting_cases.csv`.
2. Revise judge prompt wording only if parsing or rubric interpretation fails systematically.
3. Run the pilot slice across all local target models.
4. Report aggregate means with bootstrap confidence intervals if the sample is large enough; otherwise label the run as exploratory.
5. Include qualitative observations only when the saved conversations support them directly.

## Known Limitations

- Local judges are cheaper and reproducible, but weaker than frontier judges.
- Same-family target/judge overlap can bias ratings; report judge identities and inspect judge disagreement.
- The benchmark evaluates assistant behavior, not the user's mental health.
- Expert annotation should be future work for the workshop paper, with a small clinician/psychologist subset reserved for the full paper.
