import os
import argparse
from pathlib import Path


def parse_years(years: str):
    years = years.strip()
    if not years:
        return set()
    return set(item.strip() for item in years.split(",") if item.strip())


def iter_pt_ids(pt_dir: str, pt_prefix: str, pt_suffix: str):
    """Yield sample ids from pt filenames after removing the prefix and suffix."""
    for file_name in os.listdir(pt_dir):
        if not file_name.endswith(pt_suffix):
            continue
        if pt_prefix and not file_name.startswith(pt_prefix):
            continue
        stem = os.path.splitext(file_name)[0]
        sample_id = stem[len(pt_prefix) :] if stem.startswith(pt_prefix) else stem
        if sample_id:
            yield sample_id


def main():
    parser = argparse.ArgumentParser(
        description="Split sample ids into train/test groups by year and save them."
    )
    parser.add_argument("--txt", required=True, help="Input txt file with lines like: year ... cifname")
    parser.add_argument("--pt_dir", required=True, help="Directory containing pt files.")
    parser.add_argument(
        "--test_years",
        required=True,
        help="Comma-separated test years, for example: 2020,2021,2022",
    )
    parser.add_argument("--pt_prefix", default="equiv_", help="PT filename prefix.")
    parser.add_argument("--pt_suffix", default=".pt", help="PT filename suffix.")
    parser.add_argument("--out_dir", default="splits", help="Output directory.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Only use ids that appear in the txt file.",
    )
    args = parser.parse_args()

    test_years = parse_years(args.test_years)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, test_ids, missing_ids = set(), set(), set()
    seen_ids = set()

    with open(args.txt, "r", encoding="utf-8") as file_obj:
        for line_num, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] malformed line {line_num}: {line}")
                continue

            year = parts[0].strip()
            cif_name = parts[-1].strip()
            cif_stem = os.path.splitext(os.path.basename(cif_name))[0]

            if cif_stem in seen_ids:
                continue
            seen_ids.add(cif_stem)

            pt_path = os.path.join(args.pt_dir, f"{args.pt_prefix}{cif_stem}{args.pt_suffix}")
            if not os.path.exists(pt_path):
                missing_ids.add(cif_stem)
                continue

            if year in test_years:
                test_ids.add(cif_stem)
            else:
                train_ids.add(cif_stem)

    if not args.strict:
        for sample_id in iter_pt_ids(args.pt_dir, args.pt_prefix, args.pt_suffix):
            if sample_id not in train_ids and sample_id not in test_ids:
                train_ids.add(sample_id)

    years_tag = "-".join(sorted(test_years)) if test_years else "none"
    train_path = out_dir / f"train_ids_test-{years_tag}.txt"
    test_path = out_dir / f"test_ids_test-{years_tag}.txt"
    missing_path = out_dir / f"missing_ids_test-{years_tag}.txt"

    train_path.write_text("\n".join(sorted(train_ids)) + "\n", encoding="utf-8")
    test_path.write_text("\n".join(sorted(test_ids)) + "\n", encoding="utf-8")
    missing_path.write_text("\n".join(sorted(missing_ids)) + "\n", encoding="utf-8")

    print(f"Test years: {sorted(test_years)}")
    print(f"Train IDs: {len(train_ids)} -> {train_path}")
    print(f"Test IDs: {len(test_ids)} -> {test_path}")
    print(f"Missing IDs: {len(missing_ids)} -> {missing_path}")


if __name__ == "__main__":
    main()
