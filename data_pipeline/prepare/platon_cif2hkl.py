from pathlib import Path
import subprocess

from tqdm import tqdm


root_dir = Path("all_shelx_file")
platon = Path("./platon").resolve()

subdirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
failed = []

for sample_dir in tqdm(subdirs, desc="Running PLATON", unit="dir"):
    cif_files = sorted(sample_dir.glob("*.cif"))
    if not cif_files:
        failed.append((str(sample_dir), "no .cif found"))
        continue

    cif_path = next((path for path in cif_files if path.stem == sample_dir.name), cif_files[0])

    try:
        subprocess.run([str(platon), "-H", cif_path.name], cwd=str(sample_dir), check=True)
    except subprocess.CalledProcessError as exc:
        failed.append((str(cif_path), f"returncode={exc.returncode}"))

print(f"Done. Total dirs: {len(subdirs)}, Failed: {len(failed)}")
if failed:
    print("Failed (first 10):")
    for item in failed[:10]:
        print(" -", item)
