from pathlib import Path
import argparse
import shutil

from tqdm import tqdm


def read_ids(txt_path: Path):
    return [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Copy selected sample directories to a new location.")
    parser.add_argument("--root_dir", required=True, help="Source root containing many cod_id subdirectories.")
    parser.add_argument("--test_ids", required=True, help="Text file with one cod_id per line.")
    parser.add_argument("--out_dir", required=True, help="Destination root directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing destination subdirectory before copying.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    test_ids_txt = Path(args.test_ids)

    out_dir.mkdir(parents=True, exist_ok=True)

    ids = read_ids(test_ids_txt)
    print("test ids:", len(ids))

    missing = 0
    copied = 0
    skipped_exist = 0

    for cod_id in tqdm(ids, desc="Copying test dirs"):
        src = root_dir / cod_id
        dst = out_dir / cod_id

        if not src.is_dir():
            missing += 1
            continue

        if dst.exists():
            if args.overwrite:
                shutil.rmtree(dst)
            else:
                skipped_exist += 1
                continue

        shutil.copytree(src, dst)
        copied += 1

    print("Done.")
    print("copied:", copied)
    print("missing_dir:", missing)
    print("skipped_exist:", skipped_exist)
    print("out_dir:", str(out_dir))


if __name__ == "__main__":
    main()
