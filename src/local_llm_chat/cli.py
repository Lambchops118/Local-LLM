from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .config import AppConfig, DEFAULT_CONFIG_PATH
from .model_cache import (
    delete_cached_model,
    describe_cached_model,
    format_bytes,
    list_cached_model_repos,
    load_deleted_model_ids,
    mark_model_deleted,
    resolve_hf_cache_root,
    unmark_model_deleted,
)
from .session import ChatSession

if TYPE_CHECKING:
    from .modeling import ChatModel


COMMANDS = {
    "/help": "Show available commands.",
    "/reset": "Clear the current conversation history.",
    "/history": "Show how many user turns are currently in session memory.",
    "/exit": "Quit the program.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local-only multi-turn chat CLI for Hugging Face models."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON config file containing model profiles.",
    )
    parser.add_argument(
        "--model-profile",
        help="Override the active profile from the config file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override the configured max_new_tokens generation setting.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Override the configured sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Override the configured top_p sampling setting.",
    )
    parser.add_argument(
        "--system-prompt",
        help="Override the configured system prompt for this session only.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load model files from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force fully offline mode using only the local Hugging Face cache.",
    )
    parser.add_argument(
        "--list-models",
        dest="list_downloaded_models",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--list-downloaded-models",
        action="store_true",
        help="List configured models and show whether each is downloaded, deleted, or not downloaded.",
    )
    parser.add_argument(
        "--delete-downloaded-model",
        metavar="PROFILE_OR_MODEL_ID",
        help="Delete one cached model by profile name or exact Hugging Face model id.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts for destructive cache actions.",
    )
    return parser


def print_banner(model: "ChatModel") -> None:
    print("Local LLM Chat")
    print(model.describe_runtime())
    print("Commands: " + ", ".join(COMMANDS))
    print("Enter a message to begin.\n")


def handle_command(raw_text: str, session: ChatSession, model: "ChatModel") -> bool:
    command = raw_text.strip().lower()
    if command == "/help":
        for name, description in COMMANDS.items():
            print(f"{name:<9} {description}")
        return True
    if command == "/reset":
        session.reset()
        model.reset_session()
        print("Session history cleared.")
        return True
    if command == "/history":
        print(f"Session contains {session.turn_count()} user turn(s).")
        return True
    if command == "/exit":
        raise SystemExit(0)
    print(f"Unknown command: {raw_text}. Type /help for options.")
    return True


def chat_loop(
    model: "ChatModel",
    session: ChatSession,
    *,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> None:
    generation = model.with_generation_overrides(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    print_banner(model)

    while True:
        try:
            user_text = input("You> ").strip()
        except EOFError:
            print("\nExiting.")
            return
        except KeyboardInterrupt:
            print("\nInterrupted. Type /exit to quit.")
            continue

        if not user_text:
            continue
        if user_text.startswith("/"):
            handle_command(user_text, session, model)
            continue

        session.add_user_message(user_text)
        started_at = time.perf_counter()
        first_chunk_ms: Optional[float] = None
        response_parts: list[str] = []
        print("Assistant> ", end="", flush=True)
        try:
            for chunk in model.stream_response(
                session.prompt_messages(model.settings.max_context_messages),
                overrides=generation,
            ):
                if first_chunk_ms is None:
                    first_chunk_ms = (time.perf_counter() - started_at) * 1000
                response_parts.append(chunk)
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            session.remove_last_message()
            print("\nGeneration cancelled.\n")
            continue
        except RuntimeError as exc:
            session.remove_last_message()
            print()
            if "out of memory" in str(exc).lower():
                print(
                    "Generation failed due to memory pressure. Try `/reset`, reduce "
                    "`--max-new-tokens`, or switch to a smaller model profile."
                )
            else:
                print(f"Generation failed: {exc}")
            continue

        response = "".join(response_parts).strip()
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if first_chunk_ms is None:
            first_chunk_ms = elapsed_ms
        session.add_assistant_message(response)
        print("\n")
        print(f"Time to first token: {first_chunk_ms:.2f} ms")
        print(f"Time to completion: {elapsed_ms:.2f} ms\n")


def deleted_models_state_path(config_path: Path) -> Path:
    return config_path.expanduser().resolve().parent / ".deleted-models.json"


def print_downloaded_models(config: AppConfig, *, deleted_state_path: Path) -> None:
    cache_root = resolve_hf_cache_root()
    cached_repos = {repo.repo_id: repo for repo in list_cached_model_repos(cache_root)}
    deleted_model_ids = load_deleted_model_ids(deleted_state_path)

    print(f"Hugging Face cache: {cache_root}")
    print()
    print("Configured profiles:")
    for profile_name, settings in config.profiles.items():
        cached = describe_cached_model(settings.model_id, cache_root=cache_root)
        if cached:
            status = "downloaded"
            if settings.model_id in deleted_model_ids:
                unmark_model_deleted(settings.model_id, deleted_state_path)
                deleted_model_ids.discard(settings.model_id)
        elif settings.model_id in deleted_model_ids:
            status = "deleted"
        else:
            status = "not downloaded"
        size = format_bytes(cached.size_bytes) if cached else "-"
        active_marker = "*" if profile_name == config.active_profile else " "
        print(
            f"{active_marker} {profile_name:<26} {status:<14} "
            f"{size:<8} {settings.model_id}"
        )

    configured_ids = {settings.model_id for settings in config.profiles.values()}
    extra_repos = [repo for repo_id, repo in cached_repos.items() if repo_id not in configured_ids]
    if not extra_repos:
        return

    print()
    print("Other cached model repos:")
    for repo in extra_repos:
        print(
            f"  {repo.repo_id:<42} "
            f"{format_bytes(repo.size_bytes):<8} {repo.cache_dir}"
        )


def resolve_delete_target(
    target: str,
    config: AppConfig,
) -> tuple[str, list[str]]:
    if target in config.profiles:
        model_id = config.profiles[target].model_id
    else:
        model_id = target

    referencing_profiles = [
        name for name, settings in config.profiles.items() if settings.model_id == model_id
    ]
    return model_id, referencing_profiles


def confirm_delete(
    model_id: str,
    cache_path: Path,
    referencing_profiles: list[str],
) -> bool:
    print(f"About to delete cached model: {model_id}")
    if referencing_profiles:
        profiles = ", ".join(referencing_profiles)
        print(f"Referenced by profile(s): {profiles}")
    print(f"Cache path: {cache_path}")
    response = input("Continue? [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def handle_cache_command(args: argparse.Namespace) -> bool:
    if not args.list_downloaded_models and not args.delete_downloaded_model:
        return False

    try:
        config = AppConfig.load(args.config)
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.list_downloaded_models:
        print_downloaded_models(
            config,
            deleted_state_path=deleted_models_state_path(args.config),
        )
        return True

    target = str(args.delete_downloaded_model)
    cache_root = resolve_hf_cache_root()
    deleted_state_path = deleted_models_state_path(args.config)
    model_id, referencing_profiles = resolve_delete_target(target, config)
    cached = describe_cached_model(model_id, cache_root=cache_root)
    if cached is None:
        print(
            f"Error: no downloaded cache entry was found for '{target}' in {cache_root}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not args.yes and not confirm_delete(
        model_id=model_id,
        cache_path=cached.cache_dir,
        referencing_profiles=referencing_profiles,
    ):
        print("Cancelled.")
        return True

    deleted_path = delete_cached_model(model_id, cache_root=cache_root)
    if deleted_path is None:
        print(
            f"Error: cache path disappeared before deletion: {cached.cache_dir}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    mark_model_deleted(model_id, deleted_state_path)
    print(f"Deleted cached model '{model_id}' from {deleted_path}")
    return True


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if handle_cache_command(args):
        return

    try:
        import torch

        from .modeling import ModelLoadError, load_chat_model
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "an unknown package"
        print(
            "Error: missing runtime dependency "
            f"`{missing_name}`. Reinstall the project dependencies with "
            "`pip install -r requirements.txt` and try again.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    torch.set_grad_enabled(False)

    try:
        config = AppConfig.load(args.config)
        settings = config.get_profile(args.model_profile)
        if args.local_files_only or args.offline:
            settings.local_files_only = True
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if args.system_prompt is not None:
            settings.system_prompt = args.system_prompt
        model = load_chat_model(settings)
    except (ModelLoadError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    session = ChatSession(system_prompt=settings.system_prompt)
    try:
        chat_loop(
            model,
            session,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    except SystemExit:
        print("Goodbye.")
        raise


if __name__ == "__main__":
    main()
