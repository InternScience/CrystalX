import argparse
import glob
import os
import re
from typing import List, Tuple


# Match token anywhere in line, including forms like:
#   434_ALERT_2_B Short Inter ...
PAT_A = re.compile(r"ALERT_2_A")
PAT_B = re.compile(r"ALERT_2_B")


def _collect_chk_files(root_dir: str) -> Tuple[List[str], List[str], bool]:
    id_dirs = [
        os.path.join(root_dir, d)
        for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    # New layout: root/<id>/*.chk
    if id_dirs:
        files = []
        used_dirs = []
        for d in id_dirs:
            chk_in_dir = sorted(glob.glob(os.path.join(d, "*.chk")))
            if chk_in_dir:
                used_dirs.append(d)
                files.extend(chk_in_dir)
        if files:
            return files, used_dirs, True

    # Fallback: old layout or mixed tree
    files = []
    for base, _, names in os.walk(root_dir):
        for name in names:
            if name.lower().endswith(".chk"):
                files.append(os.path.join(base, name))
    files.sort()
    return files, [], False


def _scan_file(path: str) -> Tuple[bool, bool, int, int]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    count_a = len(PAT_A.findall(txt))
    count_b = len(PAT_B.findall(txt))
    return count_a > 0, count_b > 0, count_a, count_b


def _chk_bucket(path: str) -> str:
    """
    Bucket rules under root/<id>/*.chk:
      - <id>.chk                  -> id.chk
      - <id>_AIhydroWeight.chk or *_AIhydroWeight.chk -> AIhydroWeight.chk
    """
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    if ext.lower() != ".chk":
        return "other"

    parent_id = os.path.basename(os.path.dirname(path))
    stem_l = stem.lower()
    parent_l = parent_id.lower()

    if stem_l == parent_l:
        return "id.chk"
    if stem_l == f"{parent_l}_aihydroweight" or stem_l.endswith("_aihydroweight"):
        return "AIhydroWeight.chk"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count .chk files containing ALERT_2_A or ALERT_2_B and print file paths."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Directory to scan recursively for .chk files.",
    )
    args = parser.parse_args()

    root_dir = os.path.normpath(args.root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    chk_files, used_id_dirs, by_id_layout = _collect_chk_files(root_dir)
    if not chk_files:
        print(f"No .chk files found under: {root_dir}")
        return

    hit_files = []
    hit_id_dirs = set()
    total_a_occ = 0
    total_b_occ = 0
    per_id_dir = {}
    per_target = {
        "id.chk": {"total_files": 0, "hit_files": 0, "total_a_occ": 0, "total_b_occ": 0, "hit_paths": []},
        "AIhydroWeight.chk": {
            "total_files": 0,
            "hit_files": 0,
            "total_a_occ": 0,
            "total_b_occ": 0,
            "hit_paths": [],
        },
    }

    cwd = os.path.abspath(os.getcwd())

    for p in chk_files:
        has_a, has_b, count_a, count_b = _scan_file(p)
        total_a_occ += count_a
        total_b_occ += count_b

        rel_root = os.path.relpath(p, root_dir)
        top = rel_root.split(os.sep)[0] if os.sep in rel_root else "."
        if top not in per_id_dir:
            per_id_dir[top] = {
                "id_file_cnt": 0,
                "id_has_alert": False,
                "ai_file_cnt": 0,
                "ai_has_alert": False,
            }

        bucket = _chk_bucket(p)
        if bucket in per_target:
            per_target[bucket]["total_files"] += 1
            per_target[bucket]["total_a_occ"] += count_a
            per_target[bucket]["total_b_occ"] += count_b
        if bucket == "id.chk":
            per_id_dir[top]["id_file_cnt"] += 1
            if has_a or has_b:
                per_id_dir[top]["id_has_alert"] = True
        elif bucket == "AIhydroWeight.chk":
            per_id_dir[top]["ai_file_cnt"] += 1
            if has_a or has_b:
                per_id_dir[top]["ai_has_alert"] = True

        if has_a or has_b:
            hit_id_dirs.add(top)
            rel_full = os.path.relpath(os.path.abspath(p), cwd)
            hit_files.append((rel_full, count_a, count_b))
            if bucket in per_target:
                per_target[bucket]["hit_files"] += 1
                per_target[bucket]["hit_paths"].append((rel_full, count_a, count_b))

    ai_alert_id_no_alert_strict = 0
    ai_alert_id_missing_or_no_alert = 0
    for st in per_id_dir.values():
        if st["ai_has_alert"] and not st["id_has_alert"]:
            ai_alert_id_missing_or_no_alert += 1
            if st["id_file_cnt"] > 0:
                ai_alert_id_no_alert_strict += 1

    print(f"Root dir: {root_dir}")
    if by_id_layout:
        print(f"ID subdirs scanned: {len(used_id_dirs)}")
        print(f"ID subdirs with ALERT_2_A/B: {len(hit_id_dirs)}")
    print(f"Total .chk files: {len(chk_files)}")
    print(f"Files with ALERT_2_A or ALERT_2_B: {len(hit_files)}")
    print(f"Total ALERT_2_A occurrences: {total_a_occ}")
    print(f"Total ALERT_2_B occurrences: {total_b_occ}")
    print(
        "ID dirs where AIhydroWeight.chk has alert but id.chk has no alert "
        f"(id.chk exists): {ai_alert_id_no_alert_strict}"
    )
    print(
        "ID dirs where AIhydroWeight.chk has alert and id.chk has no alert "
        f"(including missing id.chk): {ai_alert_id_missing_or_no_alert}"
    )
    print()
    print("Per-target summary:")
    for key in ("id.chk", "AIhydroWeight.chk"):
        st = per_target[key]
        print(
            f"  {key}: files={st['total_files']}, hit_files={st['hit_files']}, "
            f"ALERT_2_A={st['total_a_occ']}, ALERT_2_B={st['total_b_occ']}"
        )
    print()
    print("Matched file paths:")
    if not hit_files:
        print("  (none)")
        return

    for rel, count_a, count_b in hit_files:
        print(f"  {rel}  | ALERT_2_A={count_a}, ALERT_2_B={count_b}")

    for key in ("id.chk", "AIhydroWeight.chk"):
        print()
        print(f"Matched {key} paths:")
        if not per_target[key]["hit_paths"]:
            print("  (none)")
            continue
        for rel, count_a, count_b in per_target[key]["hit_paths"]:
            print(f"  {rel}  | ALERT_2_A={count_a}, ALERT_2_B={count_b}")


if __name__ == "__main__":
    main()
