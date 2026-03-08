import os
import subprocess
from tqdm import tqdm
import random
path = 'all_real_data_final'
jlist = os.listdir(path)
jlist = ['Zeitschrift']
fpath_list = []
for journal_name in tqdm(jlist):
    flist = os.listdir(f'{path}/{journal_name}')
    for fname in flist:
        if os.path.isdir(f'{path}/{journal_name}/{fname}'):
            if not os.path.isfile(f'{path}/{journal_name}/{fname}/{fname}_a.res'):
                fpath_list.append(f'{path}/{journal_name}/{fname}/{fname}')
print(len(fpath_list))
random.seed(52)
random.shuffle(fpath_list)
executable_path = './shelxt'
for fpath in fpath_list:
    subprocess.run([executable_path, fpath])


