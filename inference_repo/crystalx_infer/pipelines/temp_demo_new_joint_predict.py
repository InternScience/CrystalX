import argparse
import os
import subprocess
import sys
import shutil


def _run(cmd, env=None):
    print("[RUN ]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _resolve_bin(name, script_dir):
    local_path = os.path.join(script_dir, name)
    if os.path.isfile(local_path) and os.access(local_path, os.X_OK):
        return local_path
    found = shutil.which(name)
    if found:
        return found
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Run heavy then hydro prediction in one script (logic unchanged)."
    )
    parser.add_argument("--fname", type=str, required=True, help="base file prefix, e.g. 2021244")
    parser.add_argument("--work_dir", type=str, required=True, help="working directory")
    parser.add_argument(
        "--main_model_path",
        type=str,
        default="final_main_model_add_no_noise_fold_3.pth",
        help="path to heavy model",
    )
    parser.add_argument(
        "--hydro_model_path",
        type=str,
        default="final_hydro_model_add_no_noise_fold_3.pth",
        help="path to hydro model",
    )
    parser.add_argument(
        "--shelxl_bin",
        type=str,
        default="",
        help="optional shelxl executable path (default: resolve from script dir or PATH)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(script_dir)
    repo_dir = os.path.dirname(package_dir)
    py = sys.executable
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = repo_dir + os.pathsep + child_env.get("PYTHONPATH", "")

    main_model_path = args.main_model_path
    if not os.path.isabs(main_model_path):
        main_model_path = os.path.join(repo_dir, "weights", main_model_path)

    hydro_model_path = args.hydro_model_path
    if not os.path.isabs(hydro_model_path):
        hydro_model_path = os.path.join(repo_dir, "weights", hydro_model_path)

    heavy_script = os.path.join(script_dir, "temp_demo_new_main.py")
    hydro_script = os.path.join(script_dir, "temp_demo_new_hydro.py")
    bond_script = os.path.join(package_dir, "postprocess", "temp_demo_new_bond_cal.py")

    shelxl_bin = args.shelxl_bin.strip()
    if not shelxl_bin:
        shelxl_bin = _resolve_bin("shelxl", script_dir)
    if not shelxl_bin:
        raise FileNotFoundError("Cannot find shelxl (set --shelxl_bin or put shelxl in PATH).")

    # Step 1: heavy prediction -> writes {fname}_AI.ins/{fname}_AI.hkl
    _run(
        [
            py,
            heavy_script,
            f"--fname={args.fname}",
            f"--work_dir={args.work_dir}",
            f"--model_path={main_model_path}",
        ],
        env=child_env,
    )

    # Keep behavior consistent with existing shell pipeline.
    err_path = os.path.join(args.work_dir, "ERROR.json")
    if os.path.exists(err_path):
        print(f"[WARN] Found {err_path}, skip hydro stage.")
        return

    # Step 2: keep AIBond stage to generate .lst for hydro graph.
    heavy_prefix = f"{args.fname}_AI"
    ai_ins = os.path.join(args.work_dir, f"{heavy_prefix}.ins")
    ai_res = os.path.join(args.work_dir, f"{heavy_prefix}.res")
    if (not os.path.exists(ai_res)) and os.path.exists(ai_ins):
        # temp_demo_new_bond_cal.py expects {fname}.res; copy AI.ins as fallback template.
        shutil.copyfile(ai_ins, ai_res)
        print(f"[INFO] Missing {ai_res}, copied from {ai_ins}")

    _run(
        [
            py,
            bond_script,
            f"--fname={heavy_prefix}",
            f"--work_dir={args.work_dir}",
        ],
        env=child_env,
    )
    _run([shelxl_bin, os.path.join(args.work_dir, f"{args.fname}_AIBond")])

    # Step 3: hydro prediction based on heavy result -> writes {fname}_AIhydro.ins
    _run(
        [
            py,
            hydro_script,
            f"--fname={heavy_prefix}",
            f"--work_dir={args.work_dir}",
            f"--model_path={hydro_model_path}",
        ],
        env=child_env,
    )

    final_ins = os.path.join(args.work_dir, f"{heavy_prefix}hydro.ins")
    if os.path.exists(final_ins):
        print(f"[ OK ] Joint output: {final_ins}")
    else:
        print(f"[WARN] Expected output not found: {final_ins}")


if __name__ == "__main__":
    main()
