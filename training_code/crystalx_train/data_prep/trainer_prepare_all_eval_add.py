from iotbx.data_manager import DataManager    # Load in the DataManager
from iotbx.shelx import crystal_symmetry_from_ins
import iotbx.cif
import os
import math
import json
import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import OrderedDict
import re
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import argparse
from joblib import dump, load
from crystalx_train.common.utils import *

def get_all_possible_pos(point):
    x = [point[0], -point[0], point[0]+0.5, -point[0]+0.5]
    y = [point[1], -point[1], point[1]+0.5, -point[1]+0.5]
    z = [point[2], -point[2], point[2]+0.5, -point[2]+0.5]
    equiv_nodes = []
    for _x in x:
        for _y in y:
            for _z in z:
                equiv_nodes.append([_x, _y, _z])
    return equiv_nodes


def new_label_align(real_frac, ideal_frac, all_label, hydro_num):
    gt = []
    hydro_gt = []
    gt_idx = []
    noise_list = []
    min_dist_list = []
    for i in range(real_frac.shape[0]):
        equiv_nodes = get_all_possible_pos(real_frac[i]+np.random.normal(0, 1e-8, 3))
        equiv_nodes = np.array(equiv_nodes)% 1
        min_dist = 1000
        min_id = -1
        equiv_min_id = -1
        for j in range(ideal_frac.shape[0]):
            if j in gt_idx:
                continue
            distances = np.mean(np.abs(equiv_nodes - ideal_frac[j]%1), axis=1)
            _min_dist = np.min(distances)
            _min_equiv = np.argmin(distances)
            if _min_dist < min_dist:
                min_dist = _min_dist
                min_id = j
                equiv_min_id = _min_equiv
        min_dist_list.append(min_dist)
        gt.append(all_label[min_id])
        hydro_gt.append(hydro_num[min_id])
        gt_idx.append(min_id)
        noise = ideal_frac[min_id]%1 - equiv_nodes[equiv_min_id]
        noise_list.append(noise.tolist())
    return gt, hydro_gt, noise_list, gt_idx, min_dist_list

def check_file_list(path):
    flist = os.listdir(path)
    fpath_list = []
    for fname in tqdm(flist):
        fpath = f'{path}/{fname}'
        if os.path.isdir(fpath):
            q = os.listdir(fpath)
            if f'{fname}_a.res' in q:
                fpath_list.append(fname)
    print(len(fpath_list))
    return fpath_list

def get_meta_info(structure):
    wavelength = structure.wavelength
    sfac_dict = structure.unit_cell_content()
    cell_params = structure.crystal_symmetry().as_cif_block()
    a = cell_params['_cell.length_a']
    b = cell_params['_cell.length_b']
    c = cell_params['_cell.length_c']
    alpha = cell_params['_cell.angle_alpha']
    beta = cell_params['_cell.angle_beta']
    gamma = cell_params['_cell.angle_gamma']
    spag = cell_params['_space_group.name_H-M_alt']
    bravis = spag[0]
    is_centric = structure.crystal_symmetry().space_group().is_centric()
    meta_info = {}
    meta_info['cell_params'] = [a,b,c,alpha,beta,gamma]
    meta_info['bravis'] = bravis
    meta_info['is_centric'] = is_centric
    meta_info['wavelength'] = wavelength
    meta_info['sfac'] = sfac_dict
    return meta_info


def main():
    from cctbx.xray.structure import structure
    path = 'all_real_data_final'

    jlist = [
    'jacs',
    'dalton',
    'organ',
    'Organic_Letters',
    'Organometallics',
    'Chemical_Communications',
    'CrystEngComm',
    'American_Mineralogist',
    'Inorganic_Chemistry',
    'Crystal_Growth_Design',
    'Organic_biomolecular_chemistry',
    'Dalton_Transactions',
    'Chemistry_of_materials',
    'Journal_of_the_Chemical_Society_Dalton_Transactions',
    'New_Journal_of_Chemistry',
    'Physics_and_Chemistry_of_Minerals',
    'Zeitschrift',
    'Ae_remain'
    ]

    pdb_list = []
    cif_list = []
    res_list = []
    for journal_name in tqdm(jlist):
        flist = os.listdir(f'{path}/{journal_name}')
        for fname in flist:
            if os.path.isdir(f'{path}/{journal_name}/{fname}'):
                if os.path.isfile(f'{path}/{journal_name}/{fname}/{fname}_a.res'):
                    pdb_list.append(f'{journal_name}_pdb_data/{fname}.pdb')
                    res_list.append(f'{path}/{journal_name}/{fname}/{fname}_a.res')
                    cif_list.append(f'{path}/{journal_name}/{fname}/{fname}.cif')

    save_dir = 'final_real_mol_add'
    os.makedirs(save_dir, exist_ok=True)

    cnt = 0
    unequal_cnt = 0
    file_error_cnt = 0
    for pdb_name, qpeak_res_name, cif_name in tqdm(zip(pdb_list, res_list, cif_list)):
        fname = os.path.basename(pdb_name)[:-4]

        save_path = f'{save_dir}/equiv_{fname}.pt'
        
        dm = DataManager()                            # Initialize the DataManager and call it dm
        dm.set_overwrite(True)   
        
        try:
            model = dm.get_model(pdb_name)  
            mol_structure = iotbx.cif.reader(file_path=cif_name).build_crystal_structures()[fname]
            ideal_frac, ideal_hydro_frac, all_label, min_dist, min_index, hydro_num = get_all_gt_points(model)
        except Exception as e:
            file_error_cnt += 1
            print(e)
            continue

        ideal_frac = np.array(ideal_frac)
        
        try:
            real_frac, z, _, _ = load_shelxt(qpeak_res_name)
        except Exception as e:
            file_error_cnt += 1
            print(e)
            continue

        if len(real_frac) != len(all_label):
            unequal_cnt += 1
            continue

        real_frac = np.array(real_frac)

        shift_list = []
        for i in np.arange(0.0, 0.46, 0.05):
            for j in np.arange(0.0, 0.46, 0.05):
                for k in np.arange(0.0, 0.46, 0.05):
                    shift_list.append([i,j,k])
        shift_list = np.array(shift_list)

        min_diff = 1000
        min_shift_id = -1
        for i in range(shift_list.shape[0]):
            gt, hydro_gt, noise_list, gt_idx, min_dist_list = new_label_align(real_frac+shift_list[i], ideal_frac, all_label, hydro_num)
            diff = np.array(min_dist_list).mean()
            if diff < min_diff:
                min_diff = diff
                min_shift_id = i
            if min_diff < 0.001:
                break
        gt, hydro_gt, noise_list, gt_idx, _ = new_label_align(real_frac+shift_list[min_shift_id], ideal_frac, all_label, hydro_num)

        real_cart, z = get_equiv_pos2(real_frac, z, mol_structure, radius=3.2)

        equiv_gt = [item for item in gt]
        _, equiv_gt = get_equiv_pos2(real_frac, equiv_gt, mol_structure, radius=3.2)


        mol_info = {}
        mol_info['pos'] =  real_cart
        mol_info['gt'] = gt
        mol_info['equiv_gt'] = equiv_gt
        mol_info['gt_idx'] = gt_idx
        mol_info['hydro_gt'] = hydro_gt
        mol_info['z'] = z
        mol_info['noise_list'] = noise_list

        torch.save(mol_info, save_path)
        cnt += 1
    print(cnt)
if __name__ == "__main__":
    main()
