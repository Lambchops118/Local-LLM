from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

try:
    from huggingface_hub.constants import HF_HUB_CACHE as HUB_CACHE_ROOT
except ImportError:
    HUB_CACHE_ROOT = None


@dataclass(frozen=True)
class CachedModelRepo:
    repo_id: str
    cache_dir: Path
    size_bytes: int
    snapshot_count: int


def resolve_hf_cache_root() -> Path:
    if HUB_CACHE_ROOT:
        return Path(HUB_CACHE_ROOT).expanduser()

    explicit_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if explicit_cache:
        return Path(explicit_cache).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "huggingface" / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def repo_id_from_dir_name(dir_name: str) -> str | None:
    if not dir_name.startswith("models--"):
        return None
    return dir_name[len("models--") :].replace("--", "/")


def cache_dir_for_model(model_id: str, cache_root: Path | None = None) -> Path:
    root = cache_root or resolve_hf_cache_root()
    return root / repo_dir_name(model_id)


def has_downloaded_snapshot(cache_dir: Path) -> bool:
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return False
    return any(path.is_file() for path in snapshots_dir.rglob("*"))


def snapshot_count(cache_dir: Path) -> int:
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return 0
    return sum(1 for path in snapshots_dir.iterdir() if path.is_dir())


def disk_usage_bytes(cache_dir: Path) -> int:
    total = 0
    for path in cache_dir.rglob("*"):
        try:
            if path.is_symlink() or not path.is_file():
                continue
            total += path.stat().st_size
        except FileNotFoundError:
            continue
    return total


def describe_cached_model(
    model_id: str,
    cache_root: Path | None = None,
) -> CachedModelRepo | None:
    cache_dir = cache_dir_for_model(model_id, cache_root=cache_root)
    if not cache_dir.is_dir() or not has_downloaded_snapshot(cache_dir):
        return None
    return CachedModelRepo(
        repo_id=model_id,
        cache_dir=cache_dir,
        size_bytes=disk_usage_bytes(cache_dir),
        snapshot_count=snapshot_count(cache_dir),
    )


def list_cached_model_repos(cache_root: Path | None = None) -> list[CachedModelRepo]:
    root = cache_root or resolve_hf_cache_root()
    if not root.is_dir():
        return []

    repos: list[CachedModelRepo] = []
    for path in sorted(root.glob("models--*")):
        if not path.is_dir() or not has_downloaded_snapshot(path):
            continue
        repo_id = repo_id_from_dir_name(path.name)
        if repo_id is None:
            continue
        repos.append(
            CachedModelRepo(
                repo_id=repo_id,
                cache_dir=path,
                size_bytes=disk_usage_bytes(path),
                snapshot_count=snapshot_count(path),
            )
        )
    return repos


def delete_cached_model(model_id: str, cache_root: Path | None = None) -> Path | None:
    cache_dir = cache_dir_for_model(model_id, cache_root=cache_root)
    if not cache_dir.is_dir():
        return None
    shutil.rmtree(cache_dir)
    return cache_dir


def format_bytes(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"
