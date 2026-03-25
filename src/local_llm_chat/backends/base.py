from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import replace
import re
from typing import Optional

from ..config import GenerationSettings, ModelSettings


class ModelLoadError(RuntimeError):
    """Raised when a model backend cannot be initialized locally."""


def clone_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "role": message.get("role", ""),
            "content": message.get("content", ""),
        }
        for message in messages
    ]


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


def should_sample(generation: GenerationSettings) -> bool:
    return generation.do_sample and generation.temperature > 0


def iter_word_chunks(fragments: Iterable[str]) -> Iterator[str]:
    buffer = ""
    for fragment in fragments:
        if not fragment:
            continue
        buffer += fragment

        cutoff = 0
        for match in re.finditer(r"\s*\S+\s*", buffer):
            if match.end() == len(buffer) and not buffer[-1].isspace():
                break
            cutoff = match.end()

        if cutoff:
            yield buffer[:cutoff]
            buffer = buffer[cutoff:]

    if buffer:
        yield buffer


class BaseChatModel(ABC):
    def __init__(self, settings: ModelSettings) -> None:
        self.settings = settings

    @classmethod
    @abstractmethod
    def load(cls, settings: ModelSettings) -> "BaseChatModel":
        raise NotImplementedError

    @abstractmethod
    def describe_runtime(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream_response(
        self,
        messages: list[dict[str, str]],
        overrides: Optional[GenerationSettings] = None,
    ) -> Iterator[str]:
        raise NotImplementedError

    def generate_response(
        self,
        messages: list[dict[str, str]],
        overrides: Optional[GenerationSettings] = None,
    ) -> str:
        response = "".join(self.stream_response(messages, overrides=overrides)).strip()
        return response or "[The model returned an empty response.]"

    def reset_session(self) -> None:
        """Clear any backend-side prompt cache or runtime session state."""

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
