from pathlib import Path
import subprocess

from tqdm import tqdm


root_dir = Path("all_shelx_file")
platon = Path("./platon").resolve()
subdirs = sorted(path for path in root_dir.iterdir() if path.is_dir())

skipped = []
failed = []

for sample_dir in tqdm(subdirs, desc="Processing same-name HKL", unit="dir"):
    hkl_path = sample_dir / f"{sample_dir.name}.hkl"
    out_hkl_path = sample_dir / f"{sample_dir.name}_sx.hkl"

    if out_hkl_path.exists() and out_hkl_path.stat().st_size > 0:
        skipped.append(f"{sample_dir} (exists non-empty: {out_hkl_path.name})")
        continue

    if not hkl_path.exists():
        skipped.append(str(sample_dir))
        continue

    try:
        subprocess.run([str(platon), "-H", hkl_path.name], cwd=str(sample_dir), check=True)
    except subprocess.CalledProcessError as exc:
        failed.append((str(hkl_path), f"returncode={exc.returncode}"))

print(f"Done. Total dirs: {len(subdirs)}")
print(f"Skipped (no same-name .hkl): {len(skipped)}")
print(f"Failed: {len(failed)}")

if skipped:
    print("Skipped (first 10):")
    for item in skipped[:10]:
        print(" -", item)

if failed:
    print("Failed (first 10):")
    for item in failed[:10]:
        print(" -", item)
