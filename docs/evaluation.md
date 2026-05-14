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
      "max_tokens": 8000
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
- `context_label`: context condition label such as `none`, `brainstorming`, or `devil_advocate`.
- `context_setting`: full context prompt text when a priming condition is used.
- `judge_results`: per-judge scores, reasoning, raw judge response, and errors.
- `judge_scores`: aggregate scores across successful judges.
- `judge_disagreement`: per-metric max-minus-min disagreement. Boolean disagreement is `1.0` when judges differ.
- `total_safety_score`: aggregate `overall_safety`, retained for compatibility with existing summaries.

Inside each `judge_results` item, `overall_safety` and `total_safety_score` are aliases for the same per-judge score. The top-level `total_safety_score` is the aggregate across successful judges.

Context sweeps:

```bash
python main.py \
  --config configs/local_5090_config.json \
  --contexts none brainstorming devil_advocate \
  --output local_context_probe.json
```

`--context` runs one condition. `--contexts` runs multiple standard-evaluation conditions in one result file and should not be combined with `--context` or `--custom-context`.

Intermediate saving:

- `evaluation.save_intermediate_results: true` writes `temp_<output_file>` during the run.
- `evaluation.save_intermediate_every: 1` saves after every completed result. This is the recommended setting for long RunPod jobs because `--resume-from results/.../temp_<output_file>` can skip successful completed rows after an interruption.
