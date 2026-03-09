from pathlib import Path
import random
import re
import time
import urllib.error
import urllib.request

from tqdm import tqdm


root_dir = Path("all_shelx_file")
url_tpl = "https://www.crystallography.net/cod/{cod_id}.hkl"


def is_effectively_empty(path: Path) -> bool:
    """Treat zero-byte or whitespace-only files as empty."""
    try:
        if not path.exists():
            return True
        if path.stat().st_size == 0:
            return True
        with path.open("r", errors="ignore") as file_obj:
            for line in file_obj:
                if line.strip():
                    return False
        return True
    except Exception:
        return True


def extract_cod_id(sample_dir: Path) -> str | None:
    """Extract a COD id from CIF filenames or the directory name."""
    cif_files = sorted(sample_dir.glob("*.cif"))
    for cif_path in cif_files:
        if cif_path.stem.isdigit():
            return cif_path.stem
    for cif_path in cif_files:
        match = re.search(r"(\d{5,})", cif_path.stem)
        if match:
            return match.group(1)

    if sample_dir.name.isdigit():
        return sample_dir.name
    match = re.search(r"(\d{5,})", sample_dir.name)
    if match:
        return match.group(1)

    return None


def download_hkl(cod_id: str, dst_path: Path) -> bool:
    """Download a COD HKL file to dst_path."""
    url = url_tpl.format(cod_id=cod_id)
    try:
        time.sleep(random.uniform(0.05, 0.2))
        tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(dst_path)
        return True
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


SEED = 20240211
random.seed(SEED)

subdirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
random.shuffle(subdirs)

dirs_need_download = []
download_ok = []
download_fail = []
no_cod_id = []
errors = []

for sample_dir in tqdm(subdirs, desc="Download if *_sx.hkl missing/empty", unit="dir"):
    try:
        sx_files = sorted(sample_dir.glob("*_sx.hkl"))
        need_download = (not sx_files) or any(is_effectively_empty(path) for path in sx_files)
        if not need_download:
            continue

        dirs_need_download.append(str(sample_dir))

        cod_id = extract_cod_id(sample_dir)
        if not cod_id:
            no_cod_id.append(str(sample_dir))
            continue

        target = sample_dir / f"{cod_id}.hkl"
        if target.exists() and not is_effectively_empty(target):
            continue

        ok = download_hkl(cod_id, target)
        if ok and not is_effectively_empty(target):
            download_ok.append((str(target), cod_id))
            print(str(target))
        else:
            download_fail.append((str(target), cod_id))
    except Exception as exc:
        errors.append((str(sample_dir), repr(exc)))

print("\n===== Summary =====")
print(f"Total dirs: {len(subdirs)}")
print(f"Dirs need download (missing OR empty *_sx.hkl): {len(dirs_need_download)}")
print(f"Download OK: {len(download_ok)}")
print(f"Download FAIL: {len(download_fail)}")
print(f"No COD id found: {len(no_cod_id)}")
print(f"Errors: {len(errors)}")

if download_fail:
    print("\n--- Download FAIL (first 20) ---")
    for path, cod_id in download_fail[:20]:
        print(" -", path, "COD:", cod_id)

if no_cod_id:
    print("\n--- No COD id (first 20) ---")
    for path in no_cod_id[:20]:
        print(" -", path)

if errors:
    print("\n--- Errors (first 10) ---")
    for path, err in errors[:10]:
        print(" -", path, err)
