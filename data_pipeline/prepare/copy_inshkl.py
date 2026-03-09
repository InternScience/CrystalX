from pathlib import Path
import shutil

from tqdm import tqdm


src_root = Path("all_shelx_file")
dst_root = Path("all_shelx_file_new")
OVERWRITE = False

subdirs = sorted(path for path in src_root.iterdir() if path.is_dir())
dst_root.mkdir(parents=True, exist_ok=True)

copied_ins = 0
copied_hkl = 0
skipped_exist = 0
missing_ins_dirs = 0
missing_sx_hkl_dirs = 0
errors = []

for sample_dir in tqdm(subdirs, desc="Copy .ins and *_sx.hkl", unit="dir"):
    try:
        rel = sample_dir.relative_to(src_root)
        out_dir = dst_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        ins_files = sorted(sample_dir.glob("*.ins"))
        sx_hkls = sorted(sample_dir.glob("*_sx.hkl"))

        if not ins_files:
            missing_ins_dirs += 1
        if not sx_hkls:
            missing_sx_hkl_dirs += 1

        for src in ins_files:
            dst = out_dir / src.name
            if dst.exists() and not OVERWRITE:
                skipped_exist += 1
                continue
            shutil.copy2(src, dst)
            copied_ins += 1

        for src in sx_hkls:
            new_name = src.name[: -len("_sx.hkl")] + ".hkl"
            dst = out_dir / new_name
            if dst.exists() and not OVERWRITE:
                skipped_exist += 1
                continue
            shutil.copy2(src, dst)
            copied_hkl += 1
    except Exception as exc:
        errors.append((str(sample_dir), repr(exc)))

print("\n===== Summary =====")
print(f"Source dirs: {len(subdirs)}")
print(f"Copied .ins files: {copied_ins}")
print(f"Copied *_sx.hkl -> .hkl files: {copied_hkl}")
print(f"Dirs missing any .ins: {missing_ins_dirs}")
print(f"Dirs missing any *_sx.hkl: {missing_sx_hkl_dirs}")
print(f"Skipped (dst exists, OVERWRITE=False): {skipped_exist}")
print(f"Errors: {len(errors)}")

if errors:
    print("\n--- Errors (first 10) ---")
    for item, err in errors[:10]:
        print(" -", item, err)
