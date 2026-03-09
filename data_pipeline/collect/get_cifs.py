from pathlib import Path
import shutil

from tqdm import tqdm


root_dir = Path("all_test_shelx")
cif_root = Path("all_cif")
out_dir = Path("all_test_cifs")

out_dir.mkdir(parents=True, exist_ok=True)

cod_ids = [path.name for path in root_dir.iterdir() if path.is_dir()]

index = {}
for cif_path in cif_root.rglob("*.cif"):
    index.setdefault(cif_path.stem, []).append(cif_path)

missing = []
copied = 0

for cod_id in tqdm(cod_ids, desc="Copying CIFs"):
    matches = index.get(cod_id, [])
    if not matches:
        missing.append(cod_id)
        continue

    for i, src in enumerate(matches, 1):
        if len(matches) == 1:
            dst = out_dir / f"{cod_id}.cif"
            if dst.exists():
                dst = out_dir / f"{cod_id}__{i}.cif"
        else:
            dst = out_dir / f"{cod_id}__{i}.cif"

        suffix_index = i
        while dst.exists():
            suffix_index += 1
            dst = out_dir / f"{cod_id}__{suffix_index}.cif"

        shutil.copy2(src, dst)
        copied += 1

print(f"cod_id count: {len(cod_ids)}")
print(f"Copied CIF files: {copied}")
print(f"Missing CIF ids: {len(missing)}")
if missing:
    print("Examples:", missing[:10])
