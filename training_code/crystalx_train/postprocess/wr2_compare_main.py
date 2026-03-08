import os
from crystalx_train.common.utils import *
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='compare wr2')
    parser.add_argument('--path', type=str, default='final_main_error_1', help='path')
    args = parser.parse_args()
    refined_dir = args.path
    file_list = os.listdir(refined_dir)
    cnt = 0
    find_error_cnt = 0
    mis_file = []
    for fname in tqdm(file_list):
        res_file_path = f'{refined_dir}/{fname}/{fname}_AI.res'
        gt_res_file_path = f'{refined_dir}/{fname}/{fname}_Human.res'
        try:
            wr2, r1, goof, q1 = get_R2(res_file_path)
            wr2_gt, r1_gt, goof_gt, q1_gt = get_R2(gt_res_file_path)
        except Exception as e:
            continue
        if r1 < r1_gt - 0.0002:
            print(r1)
            print(r1_gt)
            print(res_file_path)
            print(gt_res_file_path)
            mis_file.append(res_file_path)
            cnt += 1
    print(cnt)
    print(cnt / len(file_list))
            





