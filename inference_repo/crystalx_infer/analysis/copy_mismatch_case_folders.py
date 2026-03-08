import argparse
import csv
import os
import shutil
from typing import List, Set


def load_case_names(mismatch_txt: str) -> List[str]:
    cases: List[str] = []
    seen: Set[str] = set()

    with open(mismatch_txt, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)

        if "case_name" in sample and "\t" in sample:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                case_name = (row.get("case_name") or "").strip()
                if not case_name or case_name in seen:
                    continue
                seen.add(case_name)
                cases.append(case_name)
            return cases

        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("case_name"):
                continue
            case_name = s.split()[0]
            if case_name in seen:
                continue
            seen.add(case_name)
            cases.append(case_name)

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy mismatch case folders listed in pred_h_mismatch_cases.txt."
    )
    parser.add_argument(
        "--mismatch_txt",
        type=str,
        default="pred_h_mismatch_cases.txt",
        help="Path to mismatch txt produced by compare_aihydroweight_ratio.py",
    )
    parser.add_argument(
        "--src_root",
        type=str,
        default="",
        help="Root dir containing case folders. Default: parent directory of mismatch_txt",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="",
        help="Destination dir. Default: <src_root>/mismatch_case_folders",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination case folder when already exists",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned copy actions",
    )
    args = parser.parse_args()

    mismatch_txt = os.path.abspath(args.mismatch_txt)
    if not os.path.isfile(mismatch_txt):
        raise FileNotFoundError(f"mismatch txt not found: {mismatch_txt}")

    src_root = os.path.abspath(args.src_root) if args.src_root else os.path.dirname(mismatch_txt)
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"src_root not found: {src_root}")

    dst_dir = os.path.abspath(args.dst_dir) if args.dst_dir else os.path.join(src_root, "mismatch_case_folders")
    os.makedirs(dst_dir, exist_ok=True)

    case_names = load_case_names(mismatch_txt)
    if not case_names:
        print(f"No mismatch cases found in: {mismatch_txt}")
        print(f"Destination directory: {dst_dir}")
        return

    copied = 0
    skipped_exists = 0
    missing = 0

    for case_name in case_names:
        src_case = os.path.join(src_root, case_name)
        dst_case = os.path.join(dst_dir, case_name)

        if not os.path.isdir(src_case):
            print(f"[MISS ] {case_name} -> {src_case}")
            missing += 1
            continue

        if os.path.exists(dst_case):
            if not args.overwrite:
                print(f"[SKIP ] {case_name} (exists: {dst_case})")
                skipped_exists += 1
                continue
            if not args.dry_run:
                shutil.rmtree(dst_case)

        if args.dry_run:
            print(f"[PLAN ] {case_name}: {src_case} -> {dst_case}")
            copied += 1
            continue

        shutil.copytree(src_case, dst_case)
        print(f"[COPY ] {case_name}: {dst_case}")
        copied += 1

    print()
    print("Copy summary:")
    print(f"  mismatch_txt : {mismatch_txt}")
    print(f"  src_root     : {src_root}")
    print(f"  dst_dir      : {dst_dir}")
    print(f"  planned/copy : {copied}")
    print(f"  skipped_exist: {skipped_exists}")
    print(f"  missing_src  : {missing}")


if __name__ == "__main__":
    main()
