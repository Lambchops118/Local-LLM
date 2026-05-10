from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.json"


@dataclass
class GenerationSettings:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.05

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationSettings":
        return cls(
            max_new_tokens=int(payload.get("max_new_tokens", 512)),
            temperature=float(payload.get("temperature", 0.7)),
            top_p=float(payload.get("top_p", 0.95)),
            do_sample=bool(payload.get("do_sample", True)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.05)),
        )


@dataclass
class ModelSettings:
    name: str
    label: str
    model_id: str
    backend: str
    system_prompt: Optional[str]
    torch_dtype: str
    device_preference: list[str]
    local_files_only: bool
    max_context_messages: Optional[int]
    generation: GenerationSettings
    chat_template_options: dict[str, Any] = field(default_factory=dict)
    backend_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        name: str,
        payload: dict[str, Any],
    ) -> "ModelSettings":
        return cls(
            name=name,
            label=str(payload.get("label", name)),
            model_id=str(payload["model_id"]),
            backend=str(payload.get("backend", "transformers")),
            system_prompt=payload.get("system_prompt"),
            torch_dtype=str(payload.get("torch_dtype", "auto")),
            device_preference=list(payload.get("device_preference", ["mps", "cpu"])),
            local_files_only=bool(payload.get("local_files_only", False)),
            max_context_messages=payload.get("max_context_messages"),
            chat_template_options=dict(payload.get("chat_template_options", {})),
            backend_options=dict(payload.get("backend_options", {})),
            generation=GenerationSettings.from_dict(payload.get("generation", {})),
        )


@dataclass
class AppConfig:
    active_profile: str
    profiles: dict[str, ModelSettings]

    @classmethod
    def load(cls, config_path: Union[Path, str] = DEFAULT_CONFIG_PATH) -> "AppConfig":
        path = Path(config_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        profiles_raw = payload.get("profiles", {})
        if not profiles_raw:
            raise ValueError(f"No model profiles found in {path}")

        profiles = {
            name: ModelSettings.from_dict(name, data)
            for name, data in profiles_raw.items()
        }
        active_profile = str(payload.get("active_profile", next(iter(profiles))))
        if active_profile not in profiles:
            raise ValueError(
                f"Active profile '{active_profile}' is missing from {path}"
            )
        return cls(active_profile=active_profile, profiles=profiles)

    def get_profile(self, name: Optional[str] = None) -> ModelSettings:
        profile_name = name or self.active_profile
        try:
            return self.profiles[profile_name]
        except KeyError as exc:
            available = ", ".join(sorted(self.profiles))
            raise KeyError(
                f"Unknown profile '{profile_name}'. Available profiles: {available}"
            ) from exc
