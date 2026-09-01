# Evaluation configuration

Conspire-Bench generates each target model conversation once, then evaluates the saved conversation with one or more judge models.

Multiple target models are specified with `models`. The release matrices are
`configs/experiment_v3_local_full.json` and
`configs/experiment_v3_api_full.json`; use those files rather than copying model
IDs from documentation, because API model availability is rechecked immediately
before a live run.

Minimal local example:

```json
{
  "models": [
    {"provider": "huggingface", "model": "meta-llama/Llama-3.1-8B-Instruct"},
    {"provider": "huggingface", "model": "Qwen/Qwen2.5-3B-Instruct"},
    {"provider": "huggingface", "model": "google/gemma-2-2b-it"}
  ]
}
```

V3 runs use the explicit `models` matrix. Reusable Hugging Face settings live in
`configs/models/*.yaml`; model entries select them with `config_path`, and YAML
files may inherit shared defaults with `extends: hf_default.yaml`. Runtime
metadata records the resolved model and generation settings. Phased execution
can unload targets before judging to fit a single GPU.

The final API stage uses one strong judge from each provider family. For example:

```json
{
  "judges": [
    {
      "provider": "gemini",
      "model": "<current-strong-gemini-model>",
      "omit_sampling_parameters": true,
      "max_tokens": 16000,
      "rubric_version": "2.0"
    },
    {
      "provider": "openai",
      "model": "<current-strong-openai-model>",
      "max_tokens": 16000,
      "rubric_version": "2.0"
    }
  ]
}
```

Output fields:

- `conversation_log`: target model conversation, generated once.
- `context_label`: exact wording-condition ID such as `neutral_none`, `brainstorming_v1`, or `supportive_listening_v3`.
- `context_setting`: full context prompt text when a priming condition is used.
- `judge_results`: per-judge scores, reasoning, raw judge response, and errors.
- `judge_scores`: primary aggregate across successful judges whose provider family does not overlap the target. If none succeeds, this remains empty and the response is incomplete for primary analysis; same-family results remain in `judge_results` for sensitivity analysis only.
- `judge_disagreement`: per-metric max-minus-min disagreement. Boolean disagreement is `1.0` when judges differ.
- `judge_results[].same_family_as_target`: retains overlap information for sensitivity analysis.
- `judge_results[].judge_run_id`: fingerprints the judge model, rubric, and decoding setup so resume cannot reuse stale scores.

The frozen five-family run uses one canonical wording for `neutral`,
`brainstorming`, `critical_review`, `sensemaking`, and `supportive_listening`:

```bash
python main.py \
  --config configs/experiment_v3_local_full.json \
  --dataset Conspire-Bench-v3.json \
  --context-set main_v3 \
  --dry-run
```

`full_v3` contains neutral plus four prespecified wordings nested within each
non-neutral family. A wording is not treated as an independent frame. Direct
`--context` and `--contexts` flags are convenience interfaces; frozen studies
should use `--context-set` or exact `--context-variants` IDs.

Phased execution:

```bash
python main.py \
  --config configs/experiment_v3_local_full.json \
  --dataset Conspire-Bench-v3.json \
  --execution-mode phased \
  --context-set main_v3 \
  --resume-from results/<timestamp>/temp_local_context_probe.json \
  --output local_context_probe.json
```

`--execution-mode phased` separates target generation from judging. It first fills any missing `conversation_log` rows, keeping each target model loaded across its scenarios. It then loads one judge at a time and evaluates every cached conversation that is missing that judge's successful result. This is recommended for long local Hugging Face sweeps because it avoids repeatedly loading target and judge models for every scenario.

Intermediate saving:

- `evaluation.save_intermediate_results: true` writes `temp_<output_file>` during the run.
- `evaluation.save_intermediate_every: 1` saves after every completed generation or judge operation in phased mode. This is the recommended setting for long GPU jobs because `--resume-from results/.../temp_<output_file>` can skip successful completed work after an interruption.

V3 frame analysis:

```bash
python analysis/frame_effect_stats.py results/<run>/benchmark_results.json \
  --frame-design v3 --output results/<run>/frame_effects_main_v3.json

python analysis/frame_effect_stats.py results/<full-run>/benchmark_results.json \
  --frame-design v3 --include-noncanonical \
  --output results/<full-run>/frame_effects_full_v3.json

python analysis/make_figures.py results/<run>/frame_effects_main_v3.json \
  --out-dir results/<run>/figures
```

With `--frame-design v3` and no single `--metric`, the script reports all 28
coequal frame-by-outcome estimands and applies one Benjamini–Hochberg FDR
adjustment across them. `--include-noncanonical` first equal-averages the four
wordings nested within each non-neutral frame family for the same pairing unit;
it does not treat wording as another frame.

The pairing unit preserves target model, motif, interaction structure,
generation seed, and replicate. Intervals and sign tests cluster by motif. The
report retains per-model effects and a target-model-equal summary, reports the
`appropriate_redirection` applicability rate separately from its conditional
score, and tabulates `harmful_elaboration` subtypes. It also repeats all 28
summaries with the overlap-family clusters prespecified in
`configs/analysis_plan_v3.json`; that sensitivity analysis does not replace the
primary motif-cluster analysis.

The figure command consumes the frozen 28-estimand report rather than
re-aggregating raw judges. It renders seven separate panels and never constructs
an `overall_safety` or other cross-outcome composite. Install
`requirements-analysis.txt` before rendering figures.
