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
- `judge_results`: per-judge scores, reasoning, and errors.
- `judge_scores`: aggregate scores across successful judges.
- `judge_disagreement`: per-metric max-minus-min disagreement. Boolean disagreement is `1.0` when judges differ.
- `total_safety_score`: aggregate `overall_safety`, retained for compatibility with existing summaries.
