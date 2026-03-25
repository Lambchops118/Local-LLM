from __future__ import annotations

from .backends import BaseChatModel, MlxChatModel, ModelLoadError, TransformersChatModel
from .config import ModelSettings


BACKEND_REGISTRY = {
    "mlx": MlxChatModel,
    "transformers": TransformersChatModel,
}


def load_chat_model(settings: ModelSettings) -> BaseChatModel:
    backend_name = settings.backend.lower()
    try:
        backend_cls = BACKEND_REGISTRY[backend_name]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(
            f"Unsupported backend '{settings.backend}'. Supported backends: {supported}"
        ) from exc
    return backend_cls.load(settings)


ChatModel = BaseChatModel

__all__ = [
    "ChatModel",
    "ModelLoadError",
    "TransformersChatModel",
    "MlxChatModel",
    "load_chat_model",
]
