"""Download official CrystalX checkpoints from Hugging Face."""

from __future__ import annotations

import argparse

from crystalx_infer.common.checkpoints import (
    HF_REPO_ENV,
    download_official_weights,
    get_default_repo_id,
    get_weights_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download official CrystalX checkpoints from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="",
        help=f"Hugging Face model repo id. Defaults to ${HF_REPO_ENV} when set.",
    )
    parser.add_argument(
        "--only",
        type=str,
        choices=["heavy", "hydro", "both"],
        default="both",
        help="Download only one checkpoint or both.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh download even if the file already exists locally.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_id = get_default_repo_id(args.repo_id)
    if not repo_id:
        parser.error(f"missing repo id; pass --repo-id or set {HF_REPO_ENV}")

    if args.only == "both":
        kinds = ("heavy", "hydro")
    else:
        kinds = (args.only,)

    paths = download_official_weights(
        repo_id=repo_id,
        force_download=args.force,
        kinds=kinds,
    )

    print(f"[INFO] Downloaded CrystalX checkpoints into: {get_weights_dir()}")
    for kind, path in paths.items():
        print(f"[INFO] {kind}: {path}")


if __name__ == "__main__":
    main()
