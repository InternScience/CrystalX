from pathlib import Path

from tqdm import tqdm


root_dir = Path("all_shelx_file")
subdirs = sorted(path for path in root_dir.iterdir() if path.is_dir())

dirs_no_sx = []
dirs_empty_sx = []
empty_files = []
dirs_need_fix = []
errors = []

for sample_dir in tqdm(subdirs, desc="Checking missing/empty *_sx.hkl", unit="dir"):
    try:
        matches = sorted(sample_dir.glob("*_sx.hkl"))
        if not matches:
            dirs_no_sx.append(str(sample_dir))
            dirs_need_fix.append(str(sample_dir))
            continue

        empties = [path for path in matches if path.is_file() and path.stat().st_size == 0]
        if empties:
            dirs_empty_sx.append(str(sample_dir))
            empty_files.extend(str(path) for path in empties)
            dirs_need_fix.append(str(sample_dir))
    except Exception as exc:
        errors.append((str(sample_dir), repr(exc)))

print("\n===== Summary =====")
print(f"Total dirs: {len(subdirs)}")
print(f"Dirs with NO *_sx.hkl: {len(dirs_no_sx)}")
print(f"Dirs with EMPTY *_sx.hkl: {len(dirs_empty_sx)}")
print(f"Total empty *_sx.hkl files: {len(empty_files)}")
print(f"Dirs need fix (missing OR empty): {len(dirs_need_fix)}")
print(f"Errors: {len(errors)}")

if dirs_need_fix:
    print("\n--- Dirs need fix (first 20) ---")
    for item in dirs_need_fix[:20]:
        print(item)

if dirs_no_sx:
    print("\n--- Dirs with NO *_sx.hkl (first 20) ---")
    for item in dirs_no_sx[:20]:
        print(item)

if empty_files:
    print("\n--- Empty *_sx.hkl files (first 20) ---")
    for item in empty_files[:20]:
        print(item)

if errors:
    print("\n--- Errors (first 10) ---")
    for item, err in errors[:10]:
        print(item, err)
