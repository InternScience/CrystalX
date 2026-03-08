import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
from collections import OrderedDict, Counter
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
import argparse
from joblib import dump, load
from crystalx_train.common.utils import *
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from crystalx_train.models.schnet import SchNet
from crystalx_train.models.dimenet import DimeNet
from crystalx_train.models.spherenet import SphereNet
from crystalx_train.models.comenet import ComENet
from crystalx_train.common.utils import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch_scatter import scatter_mean
from crystalx_train.models.torchmd_et import TorchMD_ET
from crystalx_train.models.noise_output_model import (
    EquivariantScalar,
    EquivariantVectorOutput,
)
from crystalx_train.models.torchmd_net import TorchMD_Net
from rdkit import Chem
from scipy.spatial.distance import cdist
import pandas as pd

device = torch.device('cuda')

def split_test(file_list, test_num = 1000, random_state = 42):
    random.seed(random_state)
    split_index = len(file_list) - test_num
    random.shuffle(file_list)
    train_file  = file_list[:split_index]
    test_file = file_list[split_index:]
    return train_file, test_file

def partial_permute(lst, percentage=20):
    num_elements_to_shuffle = int(len(lst) * percentage / 100)
    indices_to_shuffle = random.sample(range(len(lst)), num_elements_to_shuffle)
    lst = np.array(lst)
    _lst = lst[indices_to_shuffle]
    shuffel_lst = np.random.permutation(_lst)
    lst[indices_to_shuffle] = shuffel_lst
    return lst.tolist()

def build_simple_in_memory_dataset(file_list, is_check_dist = False, is_filter = False):
    dataset = []
    cnt = 0
    mse = 0
    max_h = -1
    equiv_cnt = 0
    dist_error_cnt = 0
    main_list = os.listdir('final_main_correct') + os.listdir('final_main_subcorrect')
    for fname in tqdm(file_list):
        _fname = os.path.basename(fname)
        _fname = os.path.splitext(_fname)[0]
        _fname = _fname[6:]
        if _fname not in main_list:
            continue

        mol_info = torch.load(fname)

        z = mol_info['equiv_gt']
        y = mol_info['gt']
        z = [item.capitalize() for item in z]
        y = [item.capitalize() for item in y]
        try:
            _z = [Chem.Atom(item).GetAtomicNum() for item in z]
            y = [Chem.Atom(item).GetAtomicNum() for item in y]
        except Exception as e:
            continue
        
        
        hydro_num = mol_info['hydro_gt']
        max_h = max(max_h, max(hydro_num))

        if is_filter:
            if max(hydro_num) > 4:
                # print('large hydro')
                continue
            if 6 not in _z or 7 not in _z or 8 not in _z:
                # print('no C N O')
                continue
        
        _real_cart = mol_info['pos']
        real_cart = []
        z = []
        main_z = []
        for i in range(_real_cart.shape[0]):
            if _real_cart[i].tolist() not in real_cart:
                real_cart.append(_real_cart[i].tolist())
                main_z.append(_z[i])
                z.append(_z[i])
        real_cart = np.array(real_cart)

        # real_cart = real_cart[:len(hydro_num)]
        # z = np.array(z)[:len(hydro_num)].tolist()

        mask = np.array([0] * len(z))
        mask[:len(hydro_num)] = 1
        mask = torch.from_numpy(mask).bool()

        # z += hydro_label
        # real_cart = np.concatenate((real_cart, real_hydro_cart), axis=0)


        if is_check_dist:
            distance_matrix = cdist(real_cart, real_cart)
            distance_matrix = distance_matrix + 10*np.eye(distance_matrix.shape[0])
            if np.min(distance_matrix) < 0.2:
                dist_error_cnt += 1
                print('lalala')
                # print(fname)
                continue

        z = torch.tensor(z)
        y = torch.tensor(y)
        main_z = torch.tensor(main_z)
        real_cart = torch.from_numpy(real_cart.astype(np.float32))
        hydro_num = torch.tensor(hydro_num)
        dataset.append(Data(z = z, main_z = main_z, y = hydro_num, pos = real_cart, main_gt = y,
                            fname=fname, mask = mask))
        cnt += 1
    print(cnt)
    print(equiv_cnt)
    print(dist_error_cnt)
    return dataset, max_h

def cross_val(train_file, k=5, test_id = 0):
    part_size = len(train_file) // k + 1
    split_lists = [train_file[i:i+part_size] for i in range(0, len(train_file), part_size)]

    val_ = split_lists[test_id]
    train_k = split_lists[:test_id] + split_lists[(test_id + 1):]
    train_ = []
    for item in train_k:
        train_ += item
    return train_, val_

def get_split_data_list(file_path):
    with open(file_path, 'r') as file:
        string_list = file.readlines()
    string_list = [string.strip() for string in string_list]
    return string_list

def comb_hydro(main_label, hydro_num):
    atom_hydro = []
    for atom, hydro in zip(main_label, hydro_num):
        if hydro == 0:
            atom_hydro.append(atom)
        else:
            atom_hydro.append(atom+str(hydro))
    return atom_hydro

def get_vocab(file_list):
    vocab = []
    for fname in tqdm(file_list):
        mol_info = torch.load(fname)
        y = mol_info['gt']
        vocab += y
    vocab += 'H'
    label_encoder = LabelEncoder()
    label_encoder.fit(vocab)
    return label_encoder

def eval_validate(model, test_loader, dump_test_data = False, atom_analysis = False):
    suspect = 0
    missing_cnt = 0
    model.eval()
    correct_predictions = 0
    correct_mol = 0
    total_atoms = 0
    total_mol = 0
    mse = 0
    water = 0
    all_pred = []
    all_label = []
    all_feat = []
    water_fname = []
    error_lst = [
    '7244616','7117252','7246096','4345059','4087950','4088379','4125905',
    '4123283','1540524','1555760','7131559','1555474','7132272'
    ]
    find_error_cnt = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            try:
                data = data.to(device)
                pred, mid_feat = model(data.z, data.pos, data.batch)
                mid_feat = mid_feat[data.mask]
                pred = pred[data.mask]
                # pred = model(data)
                # loss = F.mse_loss(pred, data.noise)
                pred= F.softmax(pred, dim = -1)
            except Exception as e:
                print(e)
                missing_cnt += 1
                continue
            # mse += loss.item()
            main_correct = (data.main_gt == data.main_z[data.mask]).all()
            if not main_correct:
                # print(os.path.basename(data.fname[0]))
                continue
            pred_prob, predicted = torch.max(pred, dim=1)
            _, sorted_indices = torch.sort(pred, dim=1, descending=True)
            label = data.y
            main_z = data.main_z[data.mask]

            predicted1 = main_z * 10 + predicted
            label1 = main_z * 10 + label
            label1[label1 == 66] = 63
            correct_atom_num = (predicted1 == label1).sum().item()

            correct_predictions += correct_atom_num
            all_atom_num = label.shape[0]
            total_atoms += all_atom_num
            fname = os.path.basename(data.fname[0])
            fname = os.path.splitext(fname)[0]
            fname = fname[6:]

            old_dir = 'final_main_correct'
            lst_file_path = f'{old_dir}/{fname}/{fname}_AI.lst'
            ins_file_path = f'{old_dir}/{fname}/{fname}_AI.ins'
            if not os.path.exists(lst_file_path):
                continue
                old_dir = 'final_main_correct'
                lst_file_path = f'{old_dir}/{fname}/{fname}_AI.lst'
                ins_file_path = f'{old_dir}/{fname}/{fname}_AI.ins'

            # if correct_atom_num >= all_atom_num - 1:
            if correct_atom_num == all_atom_num:
            # if correct_atom_num >= all_atom_num - 1:
                correct_mol += 1
                refined_dir = 'final_all_correct'
            # elif correct_atom_num >= all_atom_num - 1:
            #     sub_predicted = sorted_indices[:, 1]
            #     sub_predicted1 = main_z * 10 + sub_predicted
            #     correct_atom_num += (sub_predicted1 == label1).sum().item()
            #     if correct_atom_num == all_atom_num:
            #         correct_mol += 1
            #         # predicted = label
            #         refined_dir = 'final_hydro_subcorrect'
            #     else:
            #         refined_dir = 'final_hydro_error'
            else:
                # indices = torch.nonzero(predicted != label)
                # if pred_prob[indices].max().item() > 0.99:
                #     if fname[0] != '2':
                #         suspect += 1
                #     if fname in error_lst:
                #         find_error_cnt += 1

                # print(data.main_gt[indices])
                # print(predicted[indices])
                # print(label[indices])
                # print(pred[indices])
                refined_dir = 'final_hydro_error'
            total_mol += 1
            bond_num = get_bond_in_order(lst_file_path)
            mol_graph = get_bond(lst_file_path)
            predicted = predicted.cpu()
            label = label.cpu()
            main_z = main_z.cpu()
            main_z = [Chem.Atom(int(item.item())).GetSymbol() for item in main_z]
            detailed_label = [f'{str(item1)}_{item2}_H{str(item3.item())}' for item1, item2, item3 in zip(bond_num, main_z, label)]
            detailed_predicted = [f'{str(item1)}_{item2}_H{str(item3.item())}' for item1, item2, item3 in zip(bond_num, main_z, predicted)]
            if '1_O_H2' in detailed_predicted:
                # Q_atom = get_oxygen_Q(lst_file_path, ins_file_path, mol_graph, predicted)
                water += 1
                # print(Q_atom)

            if dump_test_data and correct_atom_num == all_atom_num:
                hkl_file_path = f'{old_dir}/{fname}/{fname}_AI.hkl'
                ins_file_path = f'{old_dir}/{fname}/{fname}_AI.ins'
                res_file_path = f'{old_dir}/{fname}/{fname}_AI.res'

                os.makedirs(refined_dir, exist_ok=True)
                os.makedirs(f'{refined_dir}/{fname}', exist_ok=True)

                new_hkl_file_path = f'{refined_dir}/{fname}/{fname}_Humanhydro.hkl'
                new_res_file_path = f'{refined_dir}/{fname}/{fname}_Humanhydro.ins'
                copy_file(hkl_file_path, new_hkl_file_path)
                mol_graph = get_bond(lst_file_path)
                hfix_ins = gen_hfix_ins(ins_file_path, mol_graph, label)
                update_shelxt_hydro(res_file_path, new_res_file_path, hfix_ins)
            if atom_analysis:
                mid_feat = mid_feat.cpu()
                predicted = [f'{str(item1)}_{item2}_H{str(item3.item())}' for item1, item2, item3 in zip(bond_num, main_z, predicted)]
                label = [f'{str(item1)}_{item2}_H{str(item3.item())}' for item1, item2, item3 in zip(bond_num, main_z, label)]

                complex_hydro = ['0_O_H2','1_O_H2','2_O_H2','0_N_H4','0_O_H3','1_C_H6','5_C_H1','4_C_H1','2_O_H1']
                if any(item in detailed_label for item in complex_hydro):
                    water_fname.append(fname)
                    # print(f'{refined_dir}/{fname}/{fname}_AIhydro.ins')


                all_feat.append(mid_feat)
                all_pred.append(predicted)
                all_label.append(label)
    if atom_analysis:
        all_feat = torch.cat(all_feat)
        all_pred = np.concatenate(all_pred)
        all_label = np.concatenate(all_label)

        all_pred = np.array(['zero' if item.endswith('H0') else item for item in all_pred])
        all_label = np.array(['zero' if item.endswith('H0') else item for item in all_label])

        conf_matrix = confusion_matrix(all_label, all_pred)
        f1 = f1_score(all_label, all_pred, average=None)
        # np.save('all_feat_hydro.npy', all_feat)
        # np.save('all_f1_hydro.npy', f1)
        # np.save('conf_matrix_hydro.npy', conf_matrix)
        # np.save('all_label_hydro.npy', all_label)
        # np.save('all_pred_hydro.npy', all_pred)
        print(classification_report(all_label, all_pred))
    print(correct_predictions)
    print(total_atoms)
    atom_accuracy = correct_predictions / total_atoms
    mol_accuracy = correct_mol / total_mol
    # mse = mse / total_mol
    model.train()
    # return mse
    print('water:')
    print(water)
    print('suspect')
    print(suspect)
    print(find_error_cnt)
    # with open('z_complex_hydro_2.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for item in water_fname:
    #         writer.writerow([item])
    return atom_accuracy, mol_accuracy

def main():
    path = '/ailab/group/groups/ai4phys/zhengkaipeng/xrd/all_matched_density'
    all_train_file = [os.path.join(path, fname) for fname in os.listdir(path)]

    random.shuffle(all_train_file)
    all_test_file = all_train_file[-10000:]
    all_train_file = all_train_file[:-10000]
    
    print(f'Training Data: {len(all_train_file)}')
    print(f'Testing Data: {len(all_test_file)}')

    load_model_path = 'final_hydro_model_add_no_noise_fold_3.pth'

    all_test_dataset, _ = build_simple_in_memory_dataset(all_test_file, is_check_dist = True, is_filter=False)
    print(f'Testing Data: {len(all_test_dataset)}')

    num_classes = 8

    test_loader = DataLoader(all_test_dataset, batch_size=1, shuffle=False)
    representation_model = TorchMD_ET(
        attn_activation='silu',
        num_heads=8,
        distance_influence='both',
        )
    output_model = EquivariantScalar(256, num_classes=num_classes)
    model = TorchMD_Net(representation_model=representation_model,
                        output_model=output_model)
    model.to(device)
    model.load_state_dict(torch.load(load_model_path))

    atom_accuracy, mol_accuracy = eval_validate(model, test_loader, dump_test_data=False, atom_analysis = True)
    print(f'Atom Test Accuracy: {atom_accuracy * 100:.2f}%, Mol Test Accuracy: {mol_accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
