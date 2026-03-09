import argparse

from crystalx_infer.common.utils import copy_file


def main(args):
    hkl_file_path = f"{args.work_dir}/{args.fname}.hkl"
    res_file_path = f"{args.work_dir}/{args.fname}.res"
    new_hkl_file_path = f"{args.work_dir}/{args.fname}Bond.hkl"
    new_res_file_path = f"{args.work_dir}/{args.fname}Bond.ins"

    copy_file(hkl_file_path, new_hkl_file_path)
    copy_file(res_file_path, new_res_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SHELXL bond-calculation inputs")
    parser.add_argument("--fname", type=str, default="sample2_AIhydro", help="file name")
    parser.add_argument("--work_dir", type=str, default="work_dir", help="work directory")
    main(parser.parse_args())
