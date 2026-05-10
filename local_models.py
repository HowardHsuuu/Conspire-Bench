from __future__ import annotations

import gc
import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional


MODEL_SECTION_KEYS = {
    "provider",
    "model",
    "config_path",
    "model_config",
    "temperature",
    "max_tokens",
    "generation",
}

IMAGE_TEXT_MODEL_CLASSES = {"image_text_to_text", "auto_model_for_image_text_to_text"}


def deep_update(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_local_model_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Local model config not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        raise ValueError(f"Unsupported model config extension: {path.suffix}")

    parent = data.get("extends")
    if parent:
        parent_path = (path.parent / parent).resolve()
        data = deep_update(load_local_model_config(parent_path), data)

    data["_config_path"] = str(path.resolve())
    return data


def _torch_dtype(dtype: str):
    import torch

    if dtype == "auto":
        return "auto"
    if not hasattr(torch, dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return getattr(torch, dtype)


def _cache_key(model_config: Dict[str, Any]) -> str:
    key_parts = [
        model_config["name"],
        model_config.get("tokenizer", model_config["name"]),
        model_config.get("processor", ""),
        str(model_config.get("model_class", "causal_lm")),
        str(model_config.get("revision", "")),
        str(model_config.get("dtype", "")),
        str(model_config.get("device_map", "")),
        str(model_config.get("load_in_8bit", False)),
        str(model_config.get("load_in_4bit", False)),
    ]
    return "||".join(key_parts).replace(os.sep, "_")


class LocalModelManager:
    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        *,
        root_dir: str | Path = ".",
    ):
        self.config = config
        self.logger = logger
        self.root_dir = Path(root_dir)
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
        self.resolved_configs: Dict[str, Dict[str, Any]] = {}

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def resolve_config(
        self,
        role: str,
        model_id: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        role_config_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        global_hf = deepcopy(self.config.get("huggingface", {}))
        role_cfg = deepcopy(role_config_override if role_config_override is not None else self.config.get(role, {}))

        file_cfg = {}
        config_path = role_cfg.get("config_path") or role_cfg.get("model_config")
        if config_path:
            file_cfg = load_local_model_config(self.root_dir / config_path)

        file_model = file_cfg.get("model", file_cfg)
        file_generation = file_cfg.get("generation", {})
        role_generation = role_cfg.get("generation", {})

        inline_model = {
            key: value
            for key, value in role_cfg.items()
            if key not in MODEL_SECTION_KEYS
        }

        model_config = {
            "name": model_id,
            "tokenizer": model_id,
            "dtype": global_hf.get("dtype", "float16"),
            "device_map": global_hf.get("device_map", "auto"),
            "max_seq_length": global_hf.get("max_seq_length", 2048),
            "trust_remote_code": global_hf.get("trust_remote_code", True),
            "low_cpu_mem_usage": global_hf.get("low_cpu_mem_usage", True),
            "use_chat_template": global_hf.get("use_chat_template", True),
            "model_class": global_hf.get("model_class", "causal_lm"),
        }
        for optional_key in (
            "revision",
            "token",
            "attn_implementation",
            "load_in_8bit",
            "load_in_4bit",
            "local_files_only",
            "processor",
            "message_format",
            "padding_side",
        ):
            if optional_key in global_hf:
                model_config[optional_key] = global_hf[optional_key]

        model_config = deep_update(model_config, file_model)
        model_config = deep_update(model_config, inline_model)
        model_config["name"] = role_cfg.get("model", model_config.get("name", model_id))
        model_config["tokenizer"] = model_config.get("tokenizer") or model_config["name"]

        generation = {
            "max_new_tokens": global_hf.get("max_new_tokens", 2000),
            "temperature": global_hf.get("temperature", 0.7),
            "do_sample": global_hf.get("do_sample", True),
            "top_k": global_hf.get("top_k", 40),
        }
        for optional_key in ("top_p", "repetition_penalty", "cache_implementation"):
            if optional_key in global_hf:
                generation[optional_key] = global_hf[optional_key]
        generation = deep_update(generation, file_generation)
        generation = deep_update(generation, role_generation)

        if max_new_tokens is not None:
            generation["max_new_tokens"] = max_new_tokens
        elif role_cfg.get("max_tokens") is not None:
            generation["max_new_tokens"] = role_cfg["max_tokens"]

        if temperature is not None:
            generation["temperature"] = temperature
        elif role_cfg.get("temperature") is not None:
            generation["temperature"] = role_cfg["temperature"]

        model_config["generation"] = generation
        model_config["role"] = role
        model_config["config_path"] = file_cfg.get("_config_path")

        cache_key = _cache_key(model_config)
        self.resolved_configs[cache_key] = self._metadata(model_config)
        return model_config

    def _metadata(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        keys = (
            "name",
            "tokenizer",
            "dtype",
            "device_map",
            "max_seq_length",
            "revision",
            "config_path",
            "use_chat_template",
            "load_in_8bit",
            "load_in_4bit",
            "trust_remote_code",
            "low_cpu_mem_usage",
            "attn_implementation",
            "local_files_only",
            "model_class",
            "processor",
            "message_format",
            "padding_side",
            "role",
        )
        return {
            key: model_config.get(key)
            for key in keys
            if model_config.get(key) is not None
        } | {"generation": deepcopy(model_config.get("generation", {}))}

    def describe(
        self,
        role: str,
        model_id: str,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        role_config_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        model_config = self.resolve_config(
            role,
            model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            role_config_override=role_config_override,
        )
        return self._metadata(model_config)

    def get_model_and_tokenizer(self, model_config: Dict[str, Any]):
        cache_key = _cache_key(model_config)
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key], self.loaded_tokenizers[cache_key]

        self.logger.info("Loading local HuggingFace model: %s", model_config["name"])
        common_kwargs = self._common_load_kwargs(model_config)
        model_class = model_config.get("model_class", "causal_lm")
        tokenizer = self._load_tokenizer_or_processor(model_config, common_kwargs)
        self._ensure_padding_token(tokenizer)

        model_kwargs = {
            **common_kwargs,
            "torch_dtype": _torch_dtype(model_config.get("dtype", "float16")),
            "device_map": model_config.get("device_map", "auto"),
        }
        for key in (
            "low_cpu_mem_usage",
            "attn_implementation",
            "load_in_8bit",
            "load_in_4bit",
        ):
            if key in model_config:
                model_kwargs[key] = model_config[key]

        if model_class in IMAGE_TEXT_MODEL_CLASSES:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(model_config["name"], **model_kwargs)
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(model_config["name"], **model_kwargs)
        model.eval()

        self.loaded_models[cache_key] = model
        self.loaded_tokenizers[cache_key] = tokenizer
        return model, tokenizer

    def _load_tokenizer_or_processor(self, model_config: Dict[str, Any], common_kwargs: Dict[str, Any]):
        model_class = model_config.get("model_class", "causal_lm")
        load_kwargs = dict(common_kwargs)
        if model_config.get("padding_side"):
            load_kwargs["padding_side"] = model_config["padding_side"]

        if model_class in IMAGE_TEXT_MODEL_CLASSES:
            from transformers import AutoProcessor

            return AutoProcessor.from_pretrained(
                model_config.get("processor", model_config.get("tokenizer", model_config["name"])),
                **load_kwargs,
            )

        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_config.get("tokenizer", model_config["name"]),
            **load_kwargs,
        )

    def _text_tokenizer(self, tokenizer_or_processor):
        return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)

    def _ensure_padding_token(self, tokenizer_or_processor):
        tokenizer = self._text_tokenizer(tokenizer_or_processor)
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

    def _common_load_kwargs(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        for key in ("revision", "token", "trust_remote_code", "local_files_only"):
            if key in model_config:
                kwargs[key] = model_config[key]
        return kwargs

    async def generate(
        self,
        role: str,
        model_id: str,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        role_config_override: Optional[Dict[str, Any]] = None,
    ) -> str:
        import asyncio

        model_config = self.resolve_config(
            role,
            model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            role_config_override=role_config_override,
        )
        model, tokenizer = self.get_model_and_tokenizer(model_config)
        return await asyncio.to_thread(
            self._generate_sync,
            model,
            tokenizer,
            messages,
            model_config,
        )

    def _format_messages(self, tokenizer, messages: List[Dict[str, str]], model_config: Dict[str, Any]) -> str:
        template_source = tokenizer if hasattr(tokenizer, "apply_chat_template") else self._text_tokenizer(tokenizer)
        if model_config.get("use_chat_template", True) and hasattr(template_source, "apply_chat_template"):
            return template_source.apply_chat_template(
                self._format_template_messages(messages, model_config),
                tokenize=False,
                add_generation_prompt=True,
            )

        prompt_parts = []
        for msg in messages:
            role = msg["role"].title()
            prompt_parts.append(f"{role}: {msg['content']}")
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"

    def _format_template_messages(self, messages: List[Dict[str, str]], model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        if model_config.get("message_format") != "multimodal":
            return messages

        return [
            {
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            }
            for msg in messages
        ]

    def _encode_prompt(self, tokenizer, prompt: str, max_seq_length: int):
        try:
            return tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )
        except TypeError:
            return tokenizer(
                text=prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            )

    def _prepare_generation_inputs(
        self,
        tokenizer,
        messages: List[Dict[str, str]],
        model_config: Dict[str, Any],
        max_seq_length: int,
    ):
        if (
            model_config.get("model_class") in IMAGE_TEXT_MODEL_CLASSES
            and model_config.get("use_chat_template", True)
            and hasattr(tokenizer, "apply_chat_template")
        ):
            try:
                inputs = tokenizer.apply_chat_template(
                    self._format_template_messages(messages, model_config),
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                prompt_width = inputs["input_ids"].shape[1]
                return inputs, prompt_width
            except TypeError:
                self.logger.debug(
                    "Falling back to text-tokenizer path for %s",
                    model_config["name"],
                )

        prompt = self._format_messages(tokenizer, messages, model_config)
        inputs = self._encode_prompt(tokenizer, prompt, max_seq_length)
        prompt_width = inputs["input_ids"].shape[1]
        return inputs, prompt_width

    def _generate_sync(self, model, tokenizer, messages: List[Dict[str, str]], model_config: Dict[str, Any]) -> str:
        import torch

        generation = model_config.get("generation", {})
        max_seq_length = int(model_config.get("max_seq_length", 2048))
        max_new_tokens = int(generation.get("max_new_tokens", 2000))

        inputs, prompt_width = self._prepare_generation_inputs(
            tokenizer,
            messages,
            model_config,
            max_seq_length,
        )
        if prompt_width >= max_seq_length:
            self.logger.warning(
                "Prompt for %s was truncated to max_seq_length=%s",
                model_config["name"],
                max_seq_length,
            )

        device = next(model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        text_tokenizer = self._text_tokenizer(tokenizer)
        pad_token_id = (
            getattr(text_tokenizer, "pad_token_id", None)
            or getattr(text_tokenizer, "eos_token_id", None)
        )
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": float(generation.get("temperature", 0.7)),
            "do_sample": bool(generation.get("do_sample", True)),
        }
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id

        for key in ("top_k", "top_p", "repetition_penalty", "cache_implementation"):
            if generation.get(key) is not None:
                generation_kwargs[key] = generation[key]

        if generation_kwargs["temperature"] <= 0 or not generation_kwargs["do_sample"]:
            generation_kwargs["do_sample"] = False
            generation_kwargs.pop("temperature", None)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        generated_tokens = outputs[0][prompt_width:]
        decoder = text_tokenizer if hasattr(text_tokenizer, "decode") else tokenizer
        response = decoder.decode(generated_tokens, skip_special_tokens=True).strip()
        if not response:
            raise Exception("HuggingFace model returned empty response")
        return response

    def unload(self, model_id: Optional[str] = None):
        if model_id is None:
            self.loaded_models.clear()
            self.loaded_tokenizers.clear()
        else:
            matching = [
                key
                for key, metadata in self.resolved_configs.items()
                if metadata.get("name") == model_id
            ]
            for key in matching:
                self.loaded_models.pop(key, None)
                self.loaded_tokenizers.pop(key, None)

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
