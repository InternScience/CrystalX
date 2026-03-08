import argparse
import datetime as dt
import os
import re
import shutil
from typing import List, Tuple

from crystalx_infer.analysis.compare_aihydroweight_ratio import (
    ResultFileNotFoundError as RatioResultFileNotFoundError,
    analyze_case,
)

class CifFileNotFoundError(FileNotFoundError):
    pass


def _parse_timestamp_dir_name(dir_name: str, case_name: str):
    pattern = r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_(.+)$"
    m = re.match(pattern, dir_name)
    if not m:
        return None
    date_part, time_part, tail = m.group(1), m.group(2), m.group(3)
    if tail != case_name:
        return None
    ts = f"{date_part}_{time_part}"
    try:
        return dt.datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return None


def _find_timestamp_run_dirs(case_dir: str, case_name: str) -> List[Tuple[dt.datetime, str]]:
    run_dirs = []
    for d in os.listdir(case_dir):
        rd = os.path.join(case_dir, d)
        if not os.path.isdir(rd):
            continue
        ts = _parse_timestamp_dir_name(d, case_name)
        if ts is None:
            continue
        run_dirs.append((ts, rd))
    run_dirs.sort(key=lambda x: x[0], reverse=True)
    return run_dirs


def _pick_cif_files_in_run_dir(run_dir: str, case_name: str) -> List[str]:
    exact = os.path.join(run_dir, f"{case_name}_AIhydroWeight.cif")
    if os.path.isfile(exact):
        return [exact]

    cands = []
    for fn in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, fn)
        if not os.path.isfile(p):
            continue
        if fn.endswith("AIhydroWeight.cif"):
            cands.append(p)
    return cands


def _pick_result_file_in_run_dir(run_dir: str, case_name: str) -> str:
    cands = [
        os.path.join(run_dir, f"{case_name}_AIhydroWeight.ins"),
        os.path.join(run_dir, f"{case_name}_AIhydroWeight.res"),
        os.path.join(run_dir, f"{case_name}_AIhydro.res"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return ""


def find_case_cif_files(case_dir: str, all_timestamps: bool = False) -> List[Tuple[str, str]]:
    case_dir = os.path.abspath(case_dir)
    case_name = os.path.basename(case_dir.rstrip("\\/"))

    run_dirs = _find_timestamp_run_dirs(case_dir, case_name)
    if not run_dirs:
        raise FileNotFoundError(
            f"No timestamp run subdirectory found under {case_dir} "
            f"(expected: YYYY-MM-DD_HH-MM-SS_{case_name})"
        )

    if all_timestamps:
        items = []
        for _, rd in run_dirs:
            for cif in _pick_cif_files_in_run_dir(rd, case_name):
                items.append((rd, cif))
        if not items:
            raise CifFileNotFoundError(
                f"No *AIhydroWeight.cif found in any timestamp run dir under: {case_dir}"
            )
        return items

    latest_run_dir = run_dirs[0][1]
    cands = _pick_cif_files_in_run_dir(latest_run_dir, case_name)
    if not cands:
        raise CifFileNotFoundError(
            f"No *AIhydroWeight.cif found in latest run dir: {latest_run_dir}"
        )
    return [(latest_run_dir, cands[0])]


def _build_dst_name(case_name: str, run_dir: str, src_file: str, all_timestamps: bool) -> str:
    base = os.path.basename(src_file)
    if all_timestamps:
        run_name = os.path.basename(run_dir.rstrip("\\/"))
        return f"{case_name}__{run_name}__{base}"

    expected = f"{case_name}_AIhydroWeight.cif"
    if base == expected:
        return base
    return f"{case_name}__{base}"


def _find_origin_cif(origin_cif_root: str, case_name: str) -> str:
    p = os.path.join(origin_cif_root, f"{case_name}.cif")
    if os.path.isfile(p):
        return p
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect *AIhydroWeight.cif from timestamp run subdirectories and copy "
            "them into one destination directory."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--case_dir",
        type=str,
        help="Single case directory, e.g. inference_code/2021244",
    )
    group.add_argument(
        "--root_dir",
        type=str,
        help="Batch root directory: iterate all immediate subdirectories as cases",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="",
        help="Destination dir. Default: <case_dir or root_dir>/collected_aihydroweight_cif",
    )
    parser.add_argument(
        "--origin_cif_root",
        type=str,
        default="all_cif",
        help="Directory containing original CIF files named <case_id>.cif (default: all_cif).",
    )
    parser.add_argument(
        "--all_timestamps",
        action="store_true",
        help="Copy matching CIF files from all timestamp run dirs instead of only latest one.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination file when already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned copy actions.",
    )
    parser.add_argument(
        "--allow_ratio_tol",
        type=float,
        default=0.0,
        help="with_h consistency tolerance (same as compare_aihydroweight_ratio.py).",
    )
    args = parser.parse_args()

    src_base = os.path.abspath(args.case_dir if args.case_dir else args.root_dir)
    if not os.path.isdir(src_base):
        raise FileNotFoundError(f"source dir not found: {src_base}")

    origin_cif_root = os.path.normpath(args.origin_cif_root)
    if not os.path.isdir(origin_cif_root):
        raise FileNotFoundError(f"origin_cif_root not found: {origin_cif_root}")

    dst_dir = os.path.abspath(args.dst_dir) if args.dst_dir else os.path.join(src_base, "collected_aihydroweight_cif")
    os.makedirs(dst_dir, exist_ok=True)

    if args.case_dir:
        case_dirs = [os.path.abspath(args.case_dir)]
    else:
        root_dir = os.path.abspath(args.root_dir)
        case_dirs = [
            os.path.join(root_dir, d)
            for d in sorted(os.listdir(root_dir))
            if os.path.isdir(os.path.join(root_dir, d))
        ]

    total_cases = 0
    copied = 0
    skipped_exists = 0
    skipped_noncase = 0
    missing_cif = 0
    missing_origin = 0
    missing_result = 0
    inconsistent_with_h = 0
    errors = 0

    print(f"Source base: {src_base}")
    print(f"Origin CIF : {origin_cif_root}")
    print(f"Destination: {dst_dir}")
    print()

    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir.rstrip("\\/"))
        total_cases += 1

        if os.path.abspath(case_dir) == os.path.abspath(dst_dir):
            print(f"[SKIP ] {case_name} (destination directory)")
            skipped_noncase += 1
            continue

        if args.root_dir:
            case_ins = os.path.join(case_dir, f"{case_name}.ins")
            has_any_ins = any(
                fn.lower().endswith(".ins")
                for fn in os.listdir(case_dir)
                if os.path.isfile(os.path.join(case_dir, fn))
            )
            if (not os.path.isfile(case_ins)) and (not has_any_ins):
                print(f"[SKIP ] {case_name} (no initial .ins in case root)")
                skipped_noncase += 1
                continue

        try:
            items = find_case_cif_files(case_dir=case_dir, all_timestamps=args.all_timestamps)
            for run_dir, src_file in items:
                ai_file = _pick_result_file_in_run_dir(run_dir, case_name)
                if not ai_file:
                    print(
                        f"[MISS ] {case_name}: no AIhydroWeight .ins/.res or AIhydro.res in {run_dir}"
                    )
                    missing_result += 1
                    continue

                ret = analyze_case(
                    case_dir=case_dir,
                    ai_file=ai_file,
                    allow_ratio_tol=args.allow_ratio_tol,
                    any_timestamp_pass=False,
                )
                if not ret["with_h"]["is_match"]:
                    print(
                        f"[SKIP ] {case_name}: with_h=INCONSISTENT "
                        f"(max_diff={ret['with_h']['max_diff']:.6f}) | ai={ai_file}"
                    )
                    inconsistent_with_h += 1
                    continue

                origin_cif = _find_origin_cif(origin_cif_root, case_name)
                if not origin_cif:
                    print(
                        f"[MISS ] {case_name}: original cif not found -> "
                        f"{os.path.join(origin_cif_root, f'{case_name}.cif')}"
                    )
                    missing_origin += 1
                    continue

                case_dst_dir = os.path.join(dst_dir, case_name)
                os.makedirs(case_dst_dir, exist_ok=True)

                ai_dst_name = _build_dst_name(case_name, run_dir, src_file, args.all_timestamps)
                ai_dst_file = os.path.join(case_dst_dir, ai_dst_name)
                origin_dst_file = os.path.join(case_dst_dir, f"{case_name}.cif")

                if (os.path.exists(ai_dst_file) or os.path.exists(origin_dst_file)) and (not args.overwrite):
                    print(
                        f"[SKIP ] {case_name}: exists -> {case_dst_dir} "
                        "(use --overwrite to replace)"
                    )
                    skipped_exists += 1
                    continue

                if args.dry_run:
                    print(f"[PLAN ] {case_name}: {src_file} -> {ai_dst_file}")
                    print(f"[PLAN ] {case_name}: {origin_cif} -> {origin_dst_file}")
                    copied += 2
                    continue

                shutil.copy2(src_file, ai_dst_file)
                shutil.copy2(origin_cif, origin_dst_file)
                print(f"[COPY ] {case_name}: {ai_dst_file}")
                print(f"[COPY ] {case_name}: {origin_dst_file}")
                copied += 2

        except CifFileNotFoundError as e:
            print(f"[MISS ] {case_name} ({e})")
            missing_cif += 1
        except RatioResultFileNotFoundError as e:
            print(f"[MISS ] {case_name} ({e})")
            missing_result += 1
        except FileNotFoundError as e:
            print(f"[SKIP ] {case_name} ({e})")
            skipped_noncase += 1
        except Exception as e:
            print(f"[ERR  ] {case_name} ({e})")
            errors += 1

    print()
    print("Copy summary:")
    print(f"  total_cases     : {total_cases}")
    print(f"  copied/planned  : {copied}")
    print(f"  skipped_exists  : {skipped_exists}")
    print(f"  skipped_non_case: {skipped_noncase}")
    print(f"  missing_cif     : {missing_cif}")
    print(f"  missing_origin  : {missing_origin}")
    print(f"  missing_result  : {missing_result}")
    print(f"  with_h_inconsist: {inconsistent_with_h}")
    print(f"  errors          : {errors}")


if __name__ == "__main__":
    main()
