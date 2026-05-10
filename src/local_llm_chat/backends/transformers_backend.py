from __future__ import annotations

from collections.abc import Iterator
from threading import Event, Thread
import sys
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError
from jinja2.exceptions import TemplateError
from requests.exceptions import RequestException
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from ..config import GenerationSettings, ModelSettings
from .base import (
    apply_chat_template_with_options,
    BaseChatModel,
    ModelLoadError,
    fold_system_messages_into_user_prompt,
    is_network_error,
    iter_word_chunks,
    should_sample,
)


def load_tokenizer_with_compat_fallback(model_ref, *, local_files_only: bool):
    try:
        return AutoTokenizer.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
        )
    except AttributeError as exc:
        # Some newer multimodal Gemma tokenizer configs ship
        # `extra_special_tokens` as a list, while the fast tokenizer path
        # expects a mapping. For this text-only CLI, dropping that field is
        # enough to keep text chat working.
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        return AutoTokenizer.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
            extra_special_tokens={},
        )


def is_unknown_model_architecture_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "does not recognize this architecture" in message
        or "unrecognized configuration class" in message
        or "model type `gemma4`" in message
    )


def is_gemma4_model_id(model_id: str) -> bool:
    normalized = model_id.lower()
    return "/gemma-4" in normalized or normalized.startswith("gemma-4")


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


class TransformersChatModel(BaseChatModel):
    def __init__(
        self,
        settings: ModelSettings,
        device: torch.device,
        dtype: torch.dtype,
        tokenizer,
        model,
    ) -> None:
        super().__init__(settings)
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer
        self.model = model
        self.supports_system_role: Optional[bool] = None

    @classmethod
    def load(cls, settings: ModelSettings) -> "TransformersChatModel":
        device = select_device(settings.device_preference)
        dtype = resolve_dtype(settings.torch_dtype, device)
        if is_gemma4_model_id(settings.model_id) and sys.version_info < (3, 10):
            raise ModelLoadError(
                "Gemma 4 profiles require Python 3.10+ because the supported "
                "`transformers` builds for the `gemma4` architecture do not publish "
                "Python 3.9 wheels. Create a Python 3.10+ virtual environment, "
                "install the project dependencies there, then retry."
            )

        def load_artifacts(local_files_only: bool):
            model_ref = settings.model_id
            if local_files_only:
                model_ref = snapshot_download(
                    settings.model_id,
                    local_files_only=True,
                )

            tokenizer = load_tokenizer_with_compat_fallback(
                model_ref,
                local_files_only=local_files_only,
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_ref,
                dtype=dtype,
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
        except (LocalEntryNotFoundError, OSError, RequestException) as exc:
            if settings.local_files_only:
                raise ModelLoadError(
                    "The model could not be found in the local Hugging Face cache. "
                    "Download it once without `--local-files-only`, then retry offline."
                ) from exc
            if not settings.local_files_only and is_network_error(exc):
                try:
                    tokenizer, model = load_artifacts(local_files_only=True)
                except (LocalEntryNotFoundError, OSError, RequestException) as cached_exc:
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
        except ValueError as exc:
            if is_unknown_model_architecture_error(exc):
                raise ModelLoadError(
                    "This installed `transformers` version is too old for this model "
                    "architecture. Gemma 4 models require a newer Transformers build. "
                    "If this environment is on Python 3.9, first recreate it with "
                    "Python 3.10+; then upgrade with "
                    "`pip install --upgrade 'transformers>=5.5.0'` and retry."
                ) from exc
            raise

        model.to(device)
        model.eval()
        return cls(
            settings=settings,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
            model=model,
        )

    def describe_runtime(self) -> str:
        return (
            f"{self.settings.label} ({self.settings.model_id}) "
            f"on {self.device.type} via transformers using {self.dtype}"
        )

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
            inputs = apply_chat_template_with_options(
                self.tokenizer,
                prepared_messages,
                self.settings,
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
            inputs = apply_chat_template_with_options(
                self.tokenizer,
                fallback_messages,
                self.settings,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            if self.supports_system_role is None:
                self.supports_system_role = True

        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    def stream_response(
        self,
        messages: list[dict[str, str]],
        overrides: Optional[GenerationSettings] = None,
    ) -> Iterator[str]:
        generation = overrides or self.settings.generation
        inputs = self._apply_chat_template(messages)
        do_sample = should_sample(generation)

        generation_kwargs = {
            "max_new_tokens": generation.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": generation.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = generation.temperature
            generation_kwargs["top_p"] = generation.top_p

        cancel_event = Event()

        class CancelOnInterrupt(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return cancel_event.is_set()

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs["streamer"] = streamer
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [CancelOnInterrupt()]
        )
        generation_error: list[BaseException] = []

        def run_generation() -> None:
            try:
                with torch.inference_mode():
                    self.model.generate(**inputs, **generation_kwargs)
            except BaseException as exc:
                generation_error.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        worker = Thread(target=run_generation, daemon=True)
        worker.start()

        fragments: list[str] = []
        try:
            for chunk in iter_word_chunks(streamer):
                fragments.append(chunk)
                yield chunk
        except BaseException:
            cancel_event.set()
            streamer.on_finalized_text("", stream_end=True)
            raise
        finally:
            worker.join()

        if generation_error:
            exc = generation_error[0]
            if isinstance(exc, RuntimeError):
                raise exc
            raise RuntimeError(f"Generation failed: {exc}") from exc

        if not "".join(fragments).strip():
            yield "[The model returned an empty response.]"
