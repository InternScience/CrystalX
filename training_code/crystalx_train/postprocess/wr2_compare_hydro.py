from iotbx.data_manager import DataManager    # Load in the DataManager
import iotbx.cif
import os
from crystalx_train.common.utils import *
from tqdm import tqdm
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compare wr2')
    parser.add_argument('--path', type=str, default='final_hydro_error', help='path')
    args = parser.parse_args()
    refined_dir = args.path
    file_list = os.listdir(refined_dir)
    cnt = 0
    find_error_cnt = 0
    large_hydro_cnt = 0
    disorder_cnt = 0
    special_cnt = 0
    mis_file = []
    main_dir = 'final_main_correct'
    for fname in tqdm(file_list):
        res_file_path = f'{refined_dir}/{fname}/{fname}_AIhydroWeight.res'
        gt_res_file_path = f'{refined_dir}/{fname}/{fname}_HumanhydroWeight.res'
        try:
            wr2, r1, goof, q1 = get_R2(res_file_path)
            wr2_gt, r1_gt, goof_gt, q1_gt = get_R2(gt_res_file_path)
        except Exception as e:
            continue
        if r1 < r1_gt - 0.0002:
            print(fname)
            cnt += 1
    print(cnt)





