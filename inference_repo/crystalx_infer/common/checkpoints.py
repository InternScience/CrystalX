"""Checkpoint naming and Hugging Face download helpers for CrystalX."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


HF_REPO_ENV = "CRYSTALX_HF_REPO_ID"
HF_TOKEN_ENV = "HF_TOKEN"


@dataclass(frozen=True)
class CheckpointSpec:
    kind: str
    filename: str
    legacy_filenames: tuple[str, ...]


HEAVY_CHECKPOINT = CheckpointSpec(
    kind="heavy",
    filename="crystalx-heavy.pth",
    legacy_filenames=("final_main_model_add_no_noise_fold_3.pth",),
)
HYDRO_CHECKPOINT = CheckpointSpec(
    kind="hydro",
    filename="crystalx-hydro.pth",
    legacy_filenames=("final_hydro_model_add_no_noise_fold_3.pth",),
)

CHECKPOINT_SPECS = {
    HEAVY_CHECKPOINT.kind: HEAVY_CHECKPOINT,
    HYDRO_CHECKPOINT.kind: HYDRO_CHECKPOINT,
}


def get_weights_dir() -> Path:
    """Return the canonical local weights directory for inference_repo."""

    return Path(__file__).resolve().parents[2] / "weights"


def get_checkpoint_spec(kind: str) -> CheckpointSpec:
    try:
        return CHECKPOINT_SPECS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown checkpoint kind: {kind}") from exc


def get_default_repo_id(repo_id: str | None = None) -> str | None:
    if repo_id and repo_id.strip():
        return repo_id.strip()
    env_value = os.environ.get(HF_REPO_ENV, "").strip()
    return env_value or None


def _iter_candidate_paths(requested_path: str | None, spec: CheckpointSpec):
    weights_dir = get_weights_dir()
    requested = (requested_path or spec.filename).strip()
    path_obj = Path(requested)

    yielded: set[Path] = set()

    def _yield(path: Path):
        normalized = path.resolve() if path.exists() else path
        if normalized not in yielded:
            yielded.add(normalized)
            yield path

    if path_obj.is_absolute() or path_obj.parent != Path("."):
        yield from _yield(path_obj)
        return

    yield from _yield(Path.cwd() / path_obj.name)
    yield from _yield(weights_dir / path_obj.name)

    official_and_legacy = (spec.filename,) + spec.legacy_filenames
    if path_obj.name in official_and_legacy:
        for name in official_and_legacy:
            yield from _yield(weights_dir / name)


def download_checkpoint(
    kind: str,
    repo_id: str | None = None,
    force_download: bool = False,
) -> Path:
    """Download a published CrystalX checkpoint into the local weights directory."""

    resolved_repo_id = get_default_repo_id(repo_id)
    if not resolved_repo_id:
        raise ValueError(
            f"No Hugging Face repo id provided. Pass --repo-id or set {HF_REPO_ENV}."
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic checkpoint download. "
            "Install it with `pip install -U huggingface_hub`."
        ) from exc

    spec = get_checkpoint_spec(kind)
    weights_dir = get_weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=resolved_repo_id,
        repo_type="model",
        filename=spec.filename,
        local_dir=str(weights_dir),
        force_download=force_download,
        token=os.environ.get(HF_TOKEN_ENV),
    )
    return Path(downloaded)


def resolve_checkpoint_path(
    requested_path: str | None,
    kind: str,
    repo_id: str | None = None,
    auto_download: bool = True,
) -> str:
    """Resolve a checkpoint path from local weights, legacy aliases, or Hugging Face."""

    spec = get_checkpoint_spec(kind)
    checked: list[str] = []
    for candidate in _iter_candidate_paths(requested_path, spec):
        checked.append(str(candidate))
        if candidate.exists():
            return str(candidate.resolve())

    if auto_download and get_default_repo_id(repo_id):
        return str(download_checkpoint(kind=kind, repo_id=repo_id).resolve())

    checked_lines = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        f"Cannot find the CrystalX {kind} checkpoint.\n"
        f"Requested: {requested_path or spec.filename}\n"
        f"Checked:\n{checked_lines}\n"
        f"Download it with:\n"
        f"  python -m crystalx_infer.tools.download_weights --repo-id <hf_repo_id>\n"
        f"or set {HF_REPO_ENV}=<hf_repo_id> and rerun."
    )


def download_official_weights(
    repo_id: str | None = None,
    force_download: bool = False,
    kinds: tuple[str, ...] = ("heavy", "hydro"),
) -> dict[str, Path]:
    """Download one or more official checkpoints and return their local paths."""

    return {
        kind: download_checkpoint(
            kind=kind,
            repo_id=repo_id,
            force_download=force_download,
        )
        for kind in kinds
    }
