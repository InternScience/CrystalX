import argparse
import os

from crystalx_infer.common.utils import copy_file, update_shelxt_weight


def main(args):
    hkl_file_path = f"{args.work_dir}/{args.fname}.hkl"
    res_file_path = f"{args.work_dir}/{args.fname}.res"
    new_hkl_file_path = f"{args.work_dir}/{args.fname}Weight.hkl"
    new_res_file_path = f"{args.work_dir}/{args.fname}Weight.ins"

    if os.path.getsize(res_file_path) == 0:
        res_file_path = f"{args.work_dir}/{args.fname[:-5]}.res"

    copy_file(hkl_file_path, new_hkl_file_path)
    update_shelxt_weight(
        res_file_path,
        new_res_file_path,
        is_weight=True,
        is_acta=True,
        re_afix=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust SHELXL weight refinement inputs")
    parser.add_argument("--fname", type=str, default="sample2_AIhydro", help="file name")
    parser.add_argument("--work_dir", type=str, default="work_dir", help="work directory")
    main(parser.parse_args())
