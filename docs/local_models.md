# Local HuggingFace Model Configuration

Conspire-Bench can use local HuggingFace models as either the target model or the judge. Keep reusable model settings in `configs/models/*.yaml`, then reference those files from the main benchmark config.

Example:

```json
{
  "models": [
    {
      "provider": "huggingface",
      "model": "Qwen/Qwen2.5-3B-Instruct",
      "config_path": "configs/models/qwen25_3b_instruct.yaml"
    }
  ],
  "judges": [
    {
      "provider": "huggingface",
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "config_path": "configs/models/qwen25_7b_instruct.yaml",
      "temperature": 0.1,
      "max_tokens": 4000
    }
  ]
}
```

Model config files support YAML inheritance with `extends`:

```yaml
extends: hf_default.yaml
model:
  name: Qwen/Qwen2.5-3B-Instruct
  tokenizer: Qwen/Qwen2.5-3B-Instruct
  dtype: float16
  device_map: auto
  max_seq_length: 4096
  trust_remote_code: true
  low_cpu_mem_usage: true
  use_chat_template: true
  local_files_only: false

generation:
  max_new_tokens: 2000
  temperature: 0.7
  do_sample: true
  top_p: 0.95
```

Runtime behavior:

- `LocalModelManager` loads each unique model/tokenizer pair once and reuses it across target and judge calls when possible.
- Instruct models use `tokenizer.apply_chat_template(..., add_generation_prompt=True)` when the tokenizer provides a chat template.
- Gemma 3/4 multimodal checkpoints can set `model_class: image_text_to_text`; Conspire-Bench still sends text-only prompts, using `AutoProcessor` and `AutoModelForImageTextToText`.
- `max_seq_length`, `dtype`, `device_map`, quantization flags, and generation settings are recorded in result metadata.
- Set `local_files_only: true` for fully offline runs against an already downloaded model or a filesystem model path.
- Set `evaluation.unload_target_before_judge: true` to release the target model before local judge scoring.
- Set `evaluation.unload_after_judge: true` to release each local judge after it scores a saved conversation.
- Set `evaluation.unload_after_model: true` if target models should be released from memory after each model run.

For a single-GPU local run, start from `configs/local_5090_config.json`. It uses sequential scenarios and unloads models between target and judge calls to reduce peak VRAM. Use `configs/local_5090_full_matrix_config.json` only after the starter run succeeds.
