# Evaluation Configuration

Conspire-Bench generates each target model conversation once, then evaluates the saved conversation with one or more judge models.

Multiple target models can be specified with `models`:

```json
{
  "models": [
    {"provider": "huggingface", "model": "meta-llama/Llama-3.1-8B-Instruct"},
    {"provider": "huggingface", "model": "Qwen/Qwen2.5-3B-Instruct"},
    {"provider": "huggingface", "model": "google/gemma-2-2b-it"}
  ]
}
```

The legacy single-target `model` field is still supported.

Single-judge configuration:

```json
{
  "judge": {
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "temperature": 0.1,
    "max_tokens": 8000
  }
}
```

Multi-judge configuration:

```json
{
  "judges": [
    {
      "provider": "gemini",
      "model": "gemini-2.5-flash",
      "temperature": 0.1,
      "max_tokens": 8000,
      "rubric_version": "2.0"
    },
    {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.1,
      "max_tokens": 8000
    }
  ]
}
```

Output fields:

- `conversation_log`: target model conversation, generated once.
- `context_label`: context condition label such as `none`, `brainstorming`, or `critical_review`.
- `context_setting`: full context prompt text when a priming condition is used.
- `judge_results`: per-judge scores, reasoning, raw judge response, and errors.
- `judge_scores`: primary aggregate across successful judges whose provider family does not overlap the target. If none succeeds, this remains empty and the response is incomplete for primary analysis; same-family results remain in `judge_results` for sensitivity analysis only.
- `judge_disagreement`: per-metric max-minus-min disagreement. Boolean disagreement is `1.0` when judges differ.
- `total_safety_score`: aggregate `overall_safety` for legacy rubric v1 only; it is null for rubric v2.
- `judge_results[].same_family_as_target`: retains overlap information for sensitivity analysis.
- `judge_results[].judge_run_id`: fingerprints the judge model, rubric, and decoding setup so resume cannot reuse stale scores.

Rubric v1 results retain the historical `overall_safety` aliases. Rubric v2 does not create or synthesize that score.

Context sweeps:

```bash
python main.py \
  --config configs/local_5090_config.json \
  --contexts none brainstorming critical_review \
  --output local_context_probe.json
```

`--context` runs one condition. `--contexts` runs multiple standard-evaluation conditions in one result file and should not be combined with `--context` or `--custom-context`.

Phased execution:

```bash
python main.py \
  --config configs/local_5090_config.json \
  --execution-mode phased \
  --contexts none brainstorming critical_review \
  --resume-from results/<timestamp>/temp_local_context_probe.json \
  --output local_context_probe.json
```

`--execution-mode phased` separates target generation from judging. It first fills any missing `conversation_log` rows, keeping each target model loaded across its scenarios. It then loads one judge at a time and evaluates every cached conversation that is missing that judge's successful result. This is recommended for long local Hugging Face sweeps because it avoids repeatedly loading target and judge models for every scenario.

Intermediate saving:

- `evaluation.save_intermediate_results: true` writes `temp_<output_file>` during the run.
- `evaluation.save_intermediate_every: 1` saves after every completed result. This is the recommended setting for long RunPod jobs because `--resume-from results/.../temp_<output_file>` can skip successful completed rows after an interruption.
