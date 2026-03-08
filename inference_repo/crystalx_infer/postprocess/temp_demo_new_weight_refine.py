import os
from crystalx_infer.common.utils import *
from tqdm import tqdm
import argparse




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='adjust ShelxL weight')
    parser.add_argument('--fname', type=str, default='sample2_AIhydro', help='file name')
    parser.add_argument('--work_dir', type=str, default='work_dir', help='work directory')
    args = parser.parse_args()

    fname = args.fname
    work_dir = args.work_dir

    hkl_file_path = f'{work_dir}/{fname}.hkl'
    res_file_path = f'{work_dir}/{fname}.res'
    new_hkl_file_path = f'{work_dir}/{fname}Weight.hkl'
    new_res_file_path = f'{work_dir}/{fname}Weight.ins'

    if os.path.getsize(res_file_path) == 0:
        res_file_path = f'{work_dir}/{fname[:-5]}.res'
    copy_file(hkl_file_path, new_hkl_file_path)
    update_shelxt_weight(res_file_path, new_res_file_path, is_weight = True, is_acta = True, re_afix = False)




