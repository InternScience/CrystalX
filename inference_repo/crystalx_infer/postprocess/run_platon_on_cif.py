import argparse
import glob
import os
import shutil
import subprocess
from typing import List


def _resolve_bin(platon_bin: str, script_dir: str) -> str:
    if platon_bin:
        p = os.path.abspath(platon_bin)
        if os.path.isfile(p):
            return p
        found = shutil.which(platon_bin)
        if found:
            return found
        return ""

    local_name = os.path.join(script_dir, "platon")
    if os.path.isfile(local_name):
        return local_name

    found = shutil.which("platon")
    if found:
        return found
    return ""


def _collect_cif_files(cif_dir: str, pattern: str, recursive: bool) -> List[str]:
    if recursive:
        search_pat = os.path.join(cif_dir, "**", pattern)
        files = glob.glob(search_pat, recursive=True)
    else:
        # Default mode supports current layout: root/<id>/*.cif
        # and also keeps compatibility with root/*.cif.
        files = []
        files.extend(glob.glob(os.path.join(cif_dir, pattern)))
        files.extend(glob.glob(os.path.join(cif_dir, "*", pattern)))

    files = [
        os.path.normpath(os.path.relpath(p, cif_dir))
        for p in files
        if os.path.isfile(p)
    ]
    files.sort()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 'platon -u' for each CIF file in a directory."
    )
    parser.add_argument(
        "--cif_dir",
        type=str,
        required=True,
        help="Directory containing CIF files (e.g. collected_aihydroweight_cif).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.cif",
        help="File glob pattern under cif_dir (default: *.cif).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search deeper subdirectories (default already includes one level id subdirs).",
    )
    parser.add_argument(
        "--platon_bin",
        type=str,
        default="",
        help="Path/name for platon executable. Default: ./platon then PATH.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately when one file fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned commands.",
    )
    args = parser.parse_args()

    cif_dir = os.path.normpath(args.cif_dir)
    if not os.path.isdir(cif_dir):
        raise FileNotFoundError(f"cif_dir not found: {cif_dir}")

    cif_files = _collect_cif_files(
        cif_dir=cif_dir,
        pattern=args.pattern,
        recursive=args.recursive,
    )
    if not cif_files:
        print(f"No CIF files matched: dir={cif_dir}, pattern={args.pattern}")
        return

    if args.dry_run:
        platon_bin = args.platon_bin.strip() if args.platon_bin.strip() else "./platon"
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        platon_bin = _resolve_bin(args.platon_bin.strip(), script_dir)
        if not platon_bin:
            raise FileNotFoundError(
                "Cannot find platon executable. Use --platon_bin or put platon in script dir/PATH."
            )

    print(f"CIF dir   : {cif_dir}")
    print(f"platon    : {platon_bin}")
    print(f"pattern   : {args.pattern}")
    print(f"recursive : {args.recursive}")
    print(f"files     : {len(cif_files)}")
    print()

    ok = 0
    fail = 0

    for cif_path in cif_files:
        rel_dir = os.path.dirname(cif_path)
        cif_name = os.path.basename(cif_path)
        run_cwd = os.path.join(cif_dir, rel_dir) if rel_dir else cif_dir
        cmd = [platon_bin, "-u", cif_name]
        print(f"[RUN ] cwd={run_cwd} :: " + " ".join(cmd))

        if args.dry_run:
            ok += 1
            continue

        res = subprocess.run(cmd, cwd=run_cwd)
        if res.returncode == 0:
            ok += 1
            continue

        fail += 1
        print(f"[FAIL] returncode={res.returncode} file={cif_path} cwd={run_cwd}")
        if args.stop_on_error:
            break

    print()
    print("Platon summary:")
    print(f"  total : {len(cif_files)}")
    print(f"  ok    : {ok}")
    print(f"  fail  : {fail}")


if __name__ == "__main__":
    main()
