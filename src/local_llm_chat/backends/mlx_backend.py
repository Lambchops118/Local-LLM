from __future__ import annotations

import platform
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError
from jinja2.exceptions import TemplateError
from requests.exceptions import RequestException

from ..config import GenerationSettings, ModelSettings
from .base import (
    BaseChatModel,
    ModelLoadError,
    clone_messages,
    fold_system_messages_into_user_prompt,
    is_network_error,
    iter_word_chunks,
    should_sample,
)


class MlxChatModel(BaseChatModel):
    def __init__(
        self,
        settings: ModelSettings,
        *,
        model,
        tokenizer,
        model_config: dict[str, Any],
        make_prompt_cache,
    ) -> None:
        super().__init__(settings)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self._make_prompt_cache = make_prompt_cache
        self._backend_options = dict(settings.backend_options)
        self._max_kv_size = self._optional_int("max_kv_size")
        self._prefill_step_size = int(self._backend_options.get("prefill_step_size", 2048))
        self._kv_bits = self._optional_int("kv_bits")
        self._kv_group_size = int(self._backend_options.get("kv_group_size", 64))
        self._quantized_kv_start = int(
            self._backend_options.get("quantized_kv_start", 4096)
        )
        self._repetition_context_size = int(
            self._backend_options.get("repetition_context_size", 20)
        )
        self.supports_system_role: Optional[bool] = None
        self._prompt_cache = self._make_prompt_cache(
            self.model,
            self._max_kv_size,
        )
        self._cached_messages: list[dict[str, str]] = []

    @classmethod
    def load(cls, settings: ModelSettings) -> "MlxChatModel":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ModelLoadError(
                "The MLX backend requires Apple Silicon on macOS."
            )

        try:
            from mlx_lm import load as mlx_load
            from mlx_lm.models.cache import make_prompt_cache
        except ModuleNotFoundError as exc:
            raise ModelLoadError(
                "The MLX backend requires `mlx` and `mlx-lm`. "
                "Install the project dependencies again and retry."
            ) from exc

        backend_options = dict(settings.backend_options)
        revision = backend_options.get("revision")
        lazy = bool(backend_options.get("lazy", False))
        tokenizer_config = {}
        if backend_options.get("trust_remote_code"):
            tokenizer_config["trust_remote_code"] = True

        def load_artifacts(local_files_only: bool):
            model_ref: str | Path = settings.model_id
            load_kwargs: dict[str, Any] = {
                "tokenizer_config": tokenizer_config or None,
                "lazy": lazy,
                "return_config": True,
            }
            if local_files_only:
                model_ref = Path(
                    snapshot_download(
                        settings.model_id,
                        revision=revision,
                        local_files_only=True,
                    )
                )
            elif revision is not None:
                load_kwargs["revision"] = str(revision)

            return mlx_load(str(model_ref), **load_kwargs)

        try:
            model, tokenizer, model_config = load_artifacts(settings.local_files_only)
        except GatedRepoError as exc:
            raise ModelLoadError(
                "This MLX model is gated on Hugging Face. Accept the model license in "
                "your browser and authenticate with `huggingface-cli login`, then retry."
            ) from exc
        except (LocalEntryNotFoundError, OSError, RequestException) as exc:
            if settings.local_files_only:
                raise ModelLoadError(
                    "The MLX model could not be found in the local Hugging Face cache. "
                    "Download it once without `--local-files-only`, then retry offline."
                ) from exc

            if is_network_error(exc):
                try:
                    model, tokenizer, model_config = load_artifacts(local_files_only=True)
                except (LocalEntryNotFoundError, OSError, RequestException) as cached_exc:
                    raise ModelLoadError(
                        "Network access failed, and the MLX model is not fully available "
                        "in the local Hugging Face cache. If you already downloaded it, "
                        "rerun with `--local-files-only` or set `HF_HUB_OFFLINE=1`."
                    ) from cached_exc
            else:
                raise ModelLoadError(
                    "Failed to load the MLX model or tokenizer. Make sure you have "
                    "network access for the first download, or use a model that already "
                    "exists in your local Hugging Face cache."
                ) from exc

        return cls(
            settings=settings,
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            make_prompt_cache=make_prompt_cache,
        )

    def describe_runtime(self) -> str:
        quantization = self.model_config.get("quantization", {})
        bits = quantization.get("bits")
        weight_mode = f"{bits}-bit quantized weights" if bits else "MLX weights"
        return (
            f"{self.settings.label} ({self.settings.model_id}) "
            f"on apple-silicon via mlx using {weight_mode}"
        )

    def reset_session(self) -> None:
        self._prompt_cache = self._make_prompt_cache(
            self.model,
            self._max_kv_size,
        )
        self._cached_messages = []
        self.supports_system_role = None

    def stream_response(
        self,
        messages: list[dict[str, str]],
        overrides: Optional[GenerationSettings] = None,
    ) -> Iterator[str]:
        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_logits_processors, make_sampler
        except ModuleNotFoundError as exc:
            raise ModelLoadError(
                "The MLX backend is configured, but `mlx-lm` is not importable."
            ) from exc

        generation = overrides or self.settings.generation
        do_sample = should_sample(generation)
        prompt_messages = self._messages_for_current_turn(messages)
        prompt = self._render_prompt(prompt_messages)
        repetition_penalty = self._normalize_repetition_penalty(
            generation.repetition_penalty
        )
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=self._repetition_context_size,
        )
        sampler = make_sampler(
            temp=generation.temperature if do_sample else 0.0,
            top_p=generation.top_p if do_sample else 1.0,
        )

        try:
            response_fragments: list[str] = []

            def raw_fragments() -> Iterator[str]:
                for chunk in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=generation.max_new_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors or None,
                    prompt_cache=self._prompt_cache,
                    prefill_step_size=self._prefill_step_size,
                    max_kv_size=self._max_kv_size,
                    kv_bits=self._kv_bits,
                    kv_group_size=self._kv_group_size,
                    quantized_kv_start=self._quantized_kv_start,
                ):
                    text = str(getattr(chunk, "text", ""))
                    if text:
                        response_fragments.append(text)
                        yield text

            for chunk in iter_word_chunks(raw_fragments()):
                yield chunk
        except BaseException:
            # Generation mutates the KV cache in place, so failures need a hard reset.
            self.reset_session()
            raise

        response = "".join(response_fragments).strip()
        if not response:
            response = "[The model returned an empty response.]"
            yield response

        self._cached_messages = clone_messages(messages)
        self._cached_messages.append({"role": "assistant", "content": response})

    def _messages_for_current_turn(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        current_messages = clone_messages(messages)
        if not self._cached_messages:
            return current_messages

        prefix_length = len(self._cached_messages)
        if (
            len(current_messages) < prefix_length
            or current_messages[:prefix_length] != self._cached_messages
        ):
            self.reset_session()
            return current_messages

        delta_messages = current_messages[prefix_length:]
        if not delta_messages:
            self.reset_session()
            return current_messages
        return delta_messages

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        prepared_messages = (
            fold_system_messages_into_user_prompt(messages)
            if self.supports_system_role is False
            else messages
        )

        try:
            prompt = self.tokenizer.apply_chat_template(
                prepared_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TemplateError as exc:
            if "System role not supported" not in str(exc):
                raise RuntimeError(f"Failed to render chat template: {exc}") from exc

            self.supports_system_role = False
            fallback_messages = fold_system_messages_into_user_prompt(messages)
            prompt = self.tokenizer.apply_chat_template(
                fallback_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            if self.supports_system_role is None:
                self.supports_system_role = True

        return str(prompt)

    def _optional_int(self, key: str) -> Optional[int]:
        value = self._backend_options.get(key)
        if value is None:
            return None
        return int(value)

    def _normalize_repetition_penalty(self, penalty: float) -> Optional[float]:
        if penalty <= 0:
            return None
        if abs(penalty - 1.0) < 1e-9:
            return None
        return penalty
