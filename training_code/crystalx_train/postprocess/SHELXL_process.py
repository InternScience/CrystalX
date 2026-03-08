import os
import subprocess
from tqdm import tqdm
import random
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='shelxl process')
    parser.add_argument('--path', type=str, default='final_all_error', help='path')
    parser.add_argument('--subname', type=str, default='HumanhydroWeight', help='subname')
    args = parser.parse_args()

    path = args.path
    subname = args.subname
    fpath_list = []
    for fname in tqdm(os.listdir(path)):
        # if os.path.getsize(f'{path}/{fname}/{fname}_AI.res') == 0:
        if not os.path.exists(f'{path}/{fname}/{fname}_{subname}.res'):
            fpath_list.append(f'{path}/{fname}/{fname}_{subname}')
    print(len(fpath_list))
    executable_path = f'./shelxl'
    for fpath in fpath_list:
        subprocess.run([executable_path, fpath])


