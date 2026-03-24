from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch
from huggingface_hub.errors import GatedRepoError
from jinja2.exceptions import TemplateError
from requests.exceptions import RequestException
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GenerationSettings, ModelSettings


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded locally."""


def fold_system_messages_into_user_prompt(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    system_parts: list[str] = []
    rewritten: list[dict[str, str]] = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system":
            stripped = content.strip()
            if stripped:
                system_parts.append(stripped)
            continue
        rewritten.append({"role": role, "content": content})

    if not system_parts:
        return rewritten

    system_block = "System instructions:\n" + "\n\n".join(system_parts)
    if rewritten and rewritten[0].get("role") == "user":
        user_content = rewritten[0].get("content", "").strip()
        combined = system_block
        if user_content:
            combined = f"{system_block}\n\nUser request:\n{user_content}"
        rewritten[0] = {"role": "user", "content": combined}
        return rewritten

    rewritten.insert(0, {"role": "user", "content": system_block})
    return rewritten


def select_device(device_preference: list[str]) -> torch.device:
    normalized = [item.lower() for item in device_preference]
    for candidate in normalized:
        if candidate == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if candidate == "cpu":
            return torch.device("cpu")
        if candidate == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized == "auto":
        return torch.float16 if device.type in {"mps", "cuda"} else torch.float32

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        resolved = mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype '{dtype_name}'") from exc

    if device.type == "mps" and resolved == torch.bfloat16:
        return torch.float16
    return resolved


def is_network_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "ssl",
            "certificate verify failed",
            "httpsconnectionpool",
            "maxretryerror",
            "temporary failure in name resolution",
            "connection error",
            "failed to establish a new connection",
        )
    )


class TransformersChatModel:
    def __init__(
        self,
        settings: ModelSettings,
        device: torch.device,
        dtype: torch.dtype,
        tokenizer,
        model,
    ) -> None:
        self.settings = settings
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.model = model
        self.supports_system_role: Optional[bool] = None

    @classmethod
    def load(cls, settings: ModelSettings) -> "TransformersChatModel":
        device = select_device(settings.device_preference)
        dtype = resolve_dtype(settings.torch_dtype, device)

        def load_artifacts(local_files_only: bool):
            tokenizer = AutoTokenizer.from_pretrained(
                settings.model_id,
                local_files_only=local_files_only,
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                settings.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            )
            return tokenizer, model

        try:
            tokenizer, model = load_artifacts(settings.local_files_only)
        except GatedRepoError as exc:
            raise ModelLoadError(
                "This model is gated on Hugging Face. Accept the model license in your "
                "browser and authenticate with `huggingface-cli login`, then retry."
            ) from exc
        except (OSError, RequestException) as exc:
            if not settings.local_files_only and is_network_error(exc):
                try:
                    tokenizer, model = load_artifacts(local_files_only=True)
                except (OSError, RequestException) as cached_exc:
                    raise ModelLoadError(
                        "Network access failed, and the model could not be loaded fully "
                        "from the local Hugging Face cache. If you already downloaded "
                        "this model, rerun with `--local-files-only` or set "
                        "`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`."
                    ) from cached_exc
            else:
                raise ModelLoadError(
                    "Failed to load the model or tokenizer. Make sure you have network "
                    "access for the first download, or run with a model that already "
                    "exists in your local Hugging Face cache. If the model is already "
                    "downloaded, rerun with `--local-files-only`."
                ) from exc

        model.to(device)
        model.eval()
        return cls(settings=settings, device=device, dtype=dtype, tokenizer=tokenizer, model=model)

    def describe_runtime(self) -> str:
        return f"{self.settings.label} ({self.settings.model_id}) on {self.device.type} using {self.dtype}"

    def _apply_chat_template(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, torch.Tensor]:
        prepared_messages = (
            fold_system_messages_into_user_prompt(messages)
            if self.supports_system_role is False
            else messages
        )

        try:
            inputs = self.tokenizer.apply_chat_template(
                prepared_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TemplateError as exc:
            if "System role not supported" not in str(exc):
                raise RuntimeError(f"Failed to render chat template: {exc}") from exc

            self.supports_system_role = False
            fallback_messages = fold_system_messages_into_user_prompt(messages)
            inputs = self.tokenizer.apply_chat_template(
                fallback_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            if self.supports_system_role is None:
                self.supports_system_role = True

        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    def generate_response(
        self,
        messages: list[dict[str, str]],
        overrides: Optional[GenerationSettings] = None,
    ) -> str:
        generation = overrides or self.settings.generation
        inputs = self._apply_chat_template(messages)

        generation_kwargs = {
            "max_new_tokens": generation.max_new_tokens,
            "do_sample": generation.do_sample,
            "repetition_penalty": generation.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if generation.do_sample:
            generation_kwargs["temperature"] = generation.temperature
            generation_kwargs["top_p"] = generation.top_p

        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output[0][prompt_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response or "[The model returned an empty response.]"

    def with_generation_overrides(
        self,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> GenerationSettings:
        generation = self.settings.generation
        return replace(
            generation,
            max_new_tokens=(
                max_new_tokens
                if max_new_tokens is not None
                else generation.max_new_tokens
            ),
            temperature=temperature if temperature is not None else generation.temperature,
            top_p=top_p if top_p is not None else generation.top_p,
        )
