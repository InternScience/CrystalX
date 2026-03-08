import os
import os
import subprocess
from tqdm import tqdm
import random

# jacs
# dalton
# organ
# organletter
# organmetal
# Chemical_Communications_cif_files
# CrystEngComm_cif_files
# American_Mineralogist_cif_files
# Inorganic_Chemistry_cif_files

# Crystal_Growth_Design_cif_files
# Organic_biomolecular_chemistry_cif_files
# Dalton_Transactions_cif_files
# Chemistry_of_materials_cif_files
# Journal_of_the_Chemical_Society_Dalton_Transactions_cif_files
# New_Journal_of_Chemistry_cif_files
# Physics_and_Chemistry_of_Minerals_cif_files
# Zeitschrift
# Ae_remain


path = 'Physics_and_Chemistry_of_Minerals_cif_files'
fpath_list = []
flist = os.listdir(path)
for fname in tqdm(flist):
    fpath_list.append(os.path.join(path,fname))
print(len(fpath_list))
executable_path = f'./platon'
for fpath in fpath_list:
    subprocess.run([executable_path, '-H', fpath])


