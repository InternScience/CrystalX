import os
import random
import shutil
import json
from tqdm import tqdm
import os
import iotbx.cif
from iotbx.data_manager import DataManager    # Load in the DataManager
from cctbx.development.create_models_or_maps import generate_map_coefficients   # import the tool
from cctbx.development.create_models_or_maps import get_map_from_map_coeffs   #  import the tool

# 源文件夹和目标文件夹的路径
source_folder = 'Zeitschrift_cif_files'
save_folder = 'Zeitschrift_pdb_data'
os.makedirs(save_folder, exist_ok=True)
all_files = sorted(os.listdir(source_folder))

for file_prefix in tqdm(all_files):
    try:
        file_prefix = file_prefix[:-4]
        source_path = f'{source_folder}/{file_prefix}.cif'
        structure = iotbx.cif.reader(file_path=source_path).build_crystal_structures()[file_prefix]
        pdb_name = f'{file_prefix}.pdb'
        save_path = os.path.join(save_folder, pdb_name)
        with open(save_path, 'w') as file:
            file.write(structure.as_pdb_file())
    except Exception as e:
        continue

