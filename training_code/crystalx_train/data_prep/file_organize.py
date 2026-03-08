import os
import shutil
from tqdm import tqdm

def organize_files(source_folder, destination_folder):
    files = os.listdir(source_folder)
    for file in tqdm(files):
        source_path = os.path.join(source_folder, file)
        if os.path.isfile(source_path):
            prefix = file.split('.')[0]
            destination_path = os.path.join(destination_folder, prefix)
            os.makedirs(destination_path, exist_ok=True)
            shutil.move(source_path, os.path.join(destination_path, file))

source_folder_1 = "Acta_Crystallographica_Section_B_cif_files"
source_folder_2 = "Acta_Crystallographica_Section_B_hkl_files"
destination_folder = "Acta_Crystallographica_Section_B"
os.makedirs(destination_folder, exist_ok=True)
organize_files(source_folder_1, destination_folder)
organize_files(source_folder_2, destination_folder)
