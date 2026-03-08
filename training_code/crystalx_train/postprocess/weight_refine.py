import os
from crystalx_train.common.utils import *
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='weight refine')
    parser.add_argument('--path', type=str, default='final_all_error', help='path')
    args = parser.parse_args()

    refined_dir = args.path
    file_list = os.listdir(refined_dir)
    for fname in tqdm(file_list):
        hkl_file_path = f'{refined_dir}/{fname}/{fname}_AIhydro.hkl'
        res_file_path = f'{refined_dir}/{fname}/{fname}_AIhydro.res'

        if not os.path.exists(hkl_file_path) or not os.path.exists(res_file_path):
            print(fname)
            continue

        new_hkl_file_path = f'{refined_dir}/{fname}/{fname}_AIhydroWeight.hkl'
        new_res_file_path = f'{refined_dir}/{fname}/{fname}_AIhydroWeight.ins'

        copy_file(hkl_file_path, new_hkl_file_path)
        try:
            update_shelxt_weight(res_file_path, new_res_file_path, is_weight = True, is_acta=True, re_afix=False)
        except Exception as e:
            print(res_file_path)



