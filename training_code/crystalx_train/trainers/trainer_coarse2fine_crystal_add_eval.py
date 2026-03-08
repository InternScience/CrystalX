import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# CUDA_VISIBLE_DEVICES=0
from itertools import permutations
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
from torch_geometric.data import Data, Dataset, download_url
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
import matplotlib.pyplot as plt
import seaborn as sns

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
print(device)

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

def build_simple_in_memory_dataset(file_list, is_eval = False, is_check_dist = True, is_acc = True):
    dataset = []
    all_correct_atom = 0
    total_atom = 0
    correct_mol = 0
    total_mol = 0
    cnt = 0
    dist_error_cnt = 0
    noise_cnt = 0
    element_error_cnt = 0
    mse = 0
    max_h = -1
    bad_label = 0
    all_pred = []
    all_label = []
    for fname in tqdm(file_list):
        mol_info = torch.load(fname)

        noise = mol_info['noise_list']
        if np.max(np.abs(noise)) > 0.1:
            noise_cnt += 1
            continue
        
        z = mol_info['z']
        y = mol_info['gt']
        z = [item.capitalize() for item in z]
        y = [item.capitalize() for item in y]
        try:
            _z = [Chem.Atom(item).GetAtomicNum() for item in z]
            y = [Chem.Atom(item).GetAtomicNum() for item in y]
        except Exception as e:
            print(e)
            element_error_cnt += 1
            continue
        
        # _z = np.array(_z)
        # sort_z = sorted(_z[:len(y)][_z[:len(y)] > 20].tolist(), reverse=True) 
        # _z = sort_z + _z[len(sort_z):].tolist()

        # _z = sorted(y, reverse=True) + _z[len(y):]

        max_h = max(max_h, max(y))

        _real_cart = mol_info['pos']
        real_cart = []
        z = []
        for i in range(_real_cart.shape[0]):
            if _real_cart[i].tolist() not in real_cart:
                real_cart.append(_real_cart[i].tolist())
                z.append(_z[i])
        real_cart = np.array(real_cart)


        mask = np.array([0] * len(z))
        mask[:len(y)] = 1
        mask = torch.from_numpy(mask).bool()

        if is_check_dist:
            distance_matrix = cdist(real_cart, real_cart)
            distance_matrix = distance_matrix + 10*np.eye(distance_matrix.shape[0])
            if np.min(distance_matrix) < 0.2:
                dist_error_cnt += 1
                print('lalala')
                # print(fname)
                continue
        # all_z = one_atom_purturb(y)
        # all_z = random.sample(all_z, int(len(all_z) * 0.1))

        if is_eval:
            all_z = []
            all_z.append(sorted(y, reverse=True))
        else:
            # all_z = one_atom_purturb(y)
            # all_z = random.sample(all_z, int(len(all_z) * 0.5))
            all_z = []
            all_z.append(sorted(y, reverse=True))
            all_z.append(z[:len(y)])
            # all_z += sfac_purturb(y)
        y = torch.tensor(y)

        if is_acc:
            predicted = torch.tensor(z)[mask]
            correct_atom = (predicted == y).sum()
            mol_atom = y.shape[0]
            if correct_atom == mol_atom:
                correct_mol += 1
            total_mol += 1

        real_cart = torch.from_numpy(real_cart.astype(np.float32))
        for pz in all_z:
            pz += z[len(y):]
            if len(pz) != len(real_cart):
                print('lalala')
            pz = torch.tensor(pz)
            d1 = Data(z = pz, y = y, pos = real_cart, fname=fname, mask = mask, noise = noise)
            dataset.append(d1)
        _fname = os.path.basename(fname)
        _fname = os.path.splitext(_fname)[0]
        _fname = _fname[6:]
        save_real_main_dir = 'main_real_mol'
        os.makedirs(save_real_main_dir, exist_ok=True)
        torch.save(d1, f'{save_real_main_dir}/equiv_{_fname}.pt')   
        cnt += 1
    mol_accuracy = -1
    if is_acc:
        mol_accuracy = correct_mol / total_mol
    mse = mse / cnt
    print(cnt)
    print(dist_error_cnt)
    print(noise_cnt)
    print(mse)
    return dataset, max_h, mol_accuracy

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

def validate(model, test_loader, dump_test_data = False, atom_analysis = False):
    missing_cnt = 0
    model.eval()
    all_correct_atom = 0
    correct_mol = 0
    total_atom = 0
    total_mol = 0
    total_mse = 0
    all_pred = []
    all_label = []
    with torch.no_grad():
        for data in tqdm(test_loader):

            data = data.to(device)
            outputs = model(data.z, data.pos, data.batch)
            # outputs = model(data)
            outputs = outputs[data.mask]

            # logits = outputs[:,:-3]
            logits = outputs

            sfac = torch.unique(data.y)
            logits = logits[:,sfac]
            logits = F.softmax(logits, dim = -1)
            # outputs = scatter_mean(outputs, data.equiv_idx, dim=0)

            _, predicted = torch.max(logits, dim=1)
            predicted = sfac[predicted]

            # predicted = predicted.detach()
            # outputs = model(predicted, data.pos, data.batch)
            # outputs = outputs[data.mask]

            # # logits = outputs[:,:-3]
            # # reg = outputs[:,-3:]
            # # mse_loss = F.mse_loss(reg, data.noise)
            # # total_mse += mse_loss.item()

            # logits = outputs
            # sfac = torch.unique(data.y)
            # logits = logits[:,sfac]
            # logits = F.softmax(logits, dim = -1)
            # _, predicted = torch.max(logits, dim=1)
            # predicted = sfac[predicted]

            label = data.y
            correct_atom = (predicted == label).sum().item()
            all_correct_atom += correct_atom
            mol_atom = label.shape[0]
            total_atom += mol_atom
            if correct_atom == mol_atom or correct_atom == mol_atom - 1:
                correct_mol += 1
            total_mol += 1

            
            if dump_test_data:
                predicted = predicted.to('cpu')
                predicted = [Chem.Atom(int(item.item())).GetSymbol() for item in predicted]

                # dir_list = ['all_xrd_dataset_test', 'all_xrd_dataset_test_2', 
                #        'all_xrd_dataset_test_3', 'all_xrd_dataset_test_4']
                fname = os.path.basename(data.fname[0])
                fname = os.path.splitext(fname)[0]
                # for dir in dir_list:
                #     res_file_path = f'{dir}/{fname}/{fname}_a.res'
                #     hkl_file_path = f'{dir}/{fname}/{fname}_a.hkl'
                #     if os.path.exists(res_file_path) and os.path.exists(hkl_file_path):
                #         break

                refined_dir = 'new_bad_main'
                os.makedirs(refined_dir, exist_ok=True)
                # os.makedirs(f'{refined_dir}/{fname}', exist_ok=True)
                # new_hkl_file_path = f'{refined_dir}/{fname}/{fname}_AI.hkl'
                # copy_res_file_path = f'{refined_dir}/{fname}/{fname}_a.res'
                # copy_file(hkl_file_path, new_hkl_file_path)
                # copy_file(res_file_path, copy_res_file_path)
                # new_res_file_path = f'{refined_dir}/{fname}/{fname}_AI.ins'
                # update_shelxt(res_file_path, new_res_file_path, predicted)

                data = data.to('cpu')
                d1 = Data(z = predicted, y = data.y, pos = data.pos, fname=data.fname[0])
                torch.save(d1, f'{refined_dir}/{fname}/mol_{fname}.pt')

            if atom_analysis:
                predicted = predicted.cpu()
                label = label.cpu()
                all_pred.append(predicted)
                all_label.append(label)
                
    if atom_analysis:
        all_pred = np.concatenate(all_pred)
        all_label = np.concatenate(all_label)
        print(classification_report(all_label, all_pred))
    atom_accuracy = all_correct_atom / total_atom
    mol_accuracy = correct_mol / total_mol
    total_mse = total_mse / total_mol
    print(missing_cnt)
    model.train()
    return atom_accuracy, mol_accuracy, total_mse



def main():
    path = '/ailab/group/groups/ai4phys/zhengkaipeng/xrd/all_matched_density'
    all_train_file = [os.path.join(path, fname) for fname in os.listdir(path)]

    random.shuffle(all_train_file)
    all_test_file = all_train_file[-10000:]
    all_train_file = all_train_file[:-10000]
    
    print(f'Training Data: {len(all_train_file)}')
    print(f'Testing Data: {len(all_test_file)}')

    load_model_path = 'final_main_model_add_no_noise_fold_3.pth'

    # build test dataset
    test_dataset, _, init_mol_acc = build_simple_in_memory_dataset(all_test_file, is_eval=True, is_acc=False)

    num_classes = 98
    # num_classes = 95

    print(f'Total class: {num_classes}')
    print(f'Initial Mol Test Accuracy: {init_mol_acc * 100:.2f}%')
    print(init_mol_acc)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    representation_model = TorchMD_ET(
        attn_activation='silu',
        num_heads=8,
        distance_influence='both',
        )
    output_model = EquivariantScalar(256, num_classes=num_classes)
    model = TorchMD_Net(representation_model=representation_model,
                        output_model=output_model)
    model.to(device)


    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))
        # atom_accuracy, mol_accuracy, total_mse = validate(model, test_loader, dump_test_data=False, atom_analysis=False)
        # max_mol_accuracy = mol_accuracy
        # print(f'Atom Test Accuracy: {atom_accuracy * 100:.2f}%, Mol Test Accuracy: {mol_accuracy * 100:.2f}%, Noise MSE: {total_mse:.4f}')
        # print(f'Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%')


    atom_analysis = False
    dump_test_data = True
    missing_cnt = 0
    model.eval()
    all_correct_atom = 0
    correct_mol = 0
    total_atom = 0
    total_mol = 0
    total_mse = 0
    bad_label = 0
    all_feat = []
    all_logits = []
    all_pred = []
    all_label = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)
            outputs, mid_feat = model(data.z, data.pos, data.batch)
            # outputs = model(data)
            # outputs = outputs[data.mask]

            # logits = outputs[:,:-3]
            logits = outputs

            sfac = torch.unique(data.y)
            logits = logits[:,sfac]
            logits = F.softmax(logits, dim = -1)
            # outputs = scatter_mean(outputs, data.equiv_idx, dim=0)

            _, predicted = torch.max(logits, dim=1)
            predicted = sfac[predicted]

            # predicted = predicted.detach()
            outputs, mid_feat = model(predicted, data.pos, data.batch)
            mid_feat = mid_feat[data.mask]
            outputs = outputs[data.mask]
            inputs = data.z[data.mask]

            # logits = outputs[:,:-3]
            # reg = outputs[:,-3:]
            # mse_loss = F.mse_loss(reg, data.noise)
            # total_mse += mse_loss.item()

            logits = outputs
            logits = logits[:,sfac]
            logits = F.softmax(logits, dim = -1)

            _, predicted = torch.max(logits, dim=1)
            _, sorted_indices = torch.sort(logits, dim=1, descending=True)
            predicted = sorted_indices[:, 0]
            predicted = sfac[predicted]

            label = data.y
            correct_atom = (predicted == label).sum().item()

            all_correct_atom += correct_atom
            mol_atom = label.shape[0]
            total_atom += mol_atom
            correct_flag = None
            # if correct_atom == mol_atom or correct_atom == mol_atom - 1:
            if correct_atom == mol_atom:
                correct_mol += 1       
                refined_dir = 'final_main_correct'   
                correct_flag = 'correct'
            # elif correct_atom == mol_atom - 1:
            #     sub_predicted = sorted_indices[:, 1]
            #     sub_predicted = sfac[sub_predicted]
            #     correct_atom += (sub_predicted == label).sum().item()
            #     if correct_atom == mol_atom:
            #         correct_mol += 1  
            #         predicted = label
            #         refined_dir = 'final_main_subcorrect'      
            #         correct_flag = 'sub_correct'
            else:
                # pred_gt_diff = abs(inputs - label)
                # if pred_gt_diff[pred_gt_diff > 20].shape[0] > 1:
                #     print(pred_gt_diff)
                #     print(predicted)
                #     print(label)
                #     bad_label += 1
                indices = torch.nonzero(predicted != label)
                refined_dir = 'final_main_error_1'
                correct_flag = 'false'
                # print(predicted[indices])
                # print(label[indices])
                # print(logits[indices])
            total_mol += 1
            if dump_test_data and correct_flag == 'false':
                predicted = predicted.to('cpu')
                predicted = [Chem.Atom(int(item.item())).GetSymbol() for item in predicted]

                label = label.to('cpu')
                label = [Chem.Atom(int(item.item())).GetSymbol() for item in label]

                # dir_list = ['all_xrd_dataset_test', 'all_xrd_dataset_test_2', 
                #         'all_xrd_dataset_test_3', 'all_xrd_dataset_test_4',]
                # dir_list += ['Acta_Crystallographica_Section_B_test',
                # 'Acta_Crystallographica_Section_C_test']
                # real_path_final = 'all_real_data_final'
                # jlist = os.listdir(real_path_final)
                # dir_list += [os.path.join(real_path_final, jname) for jname in jlist]


                fname = os.path.basename(data.fname[0])
                fname = os.path.splitext(fname)[0]
                fname = fname[6:]

                # if not os.path.getsize(f'{refined_dir}/{fname}/{fname}_AI.res') == 0:
                #     continue

                # for ori_dir in dir_list:
                #     res_file_path = f'{ori_dir}/{fname}/{fname}_a.res'
                #     hkl_file_path = f'{ori_dir}/{fname}/{fname}_a.hkl'
                #     cif_file_path = f'{ori_dir}/{fname}/{fname}.cif'
                #     if os.path.exists(res_file_path) and os.path.exists(hkl_file_path):
                #         break
                
                ori_dir = 'final_main_error_1'
                res_file_path = f'{ori_dir}/{fname}/{fname}_a.res'
                hkl_file_path = f'{ori_dir}/{fname}/{fname}_AI.hkl'
                # cif_file_path = f'{ori_dir}/{fname}/{fname}.cif'

                os.makedirs(refined_dir, exist_ok=True)
                os.makedirs(f'{refined_dir}/{fname}', exist_ok=True)

                new_hkl_file_path = f'{refined_dir}/{fname}/{fname}_Human.hkl'
                # copy_cif_file_path = f'{refined_dir}/{fname}/{fname}.cif'
                # copy_res_file_path = f'{refined_dir}/{fname}/{fname}_a.res'


                copy_file(hkl_file_path, new_hkl_file_path)
                # copy_file(res_file_path, copy_res_file_path)
                # copy_file(cif_file_path, copy_cif_file_path)

                new_res_file_path = f'{refined_dir}/{fname}/{fname}_Human.ins'

                update_shelxt(res_file_path, new_res_file_path, label, refine_round = 10)

                # data = data.to('cpu')
                # d1 = Data(z = predicted, y = data.y, pos = data.pos, fname=data.fname[0], noise = data.noise[0])
                # torch.save(d1, f'{refined_dir}/{fname}/mol_{fname}.pt')   
                
            if atom_analysis:
                predicted = predicted.cpu()
                label = label.cpu()
                mid_feat = mid_feat.cpu()
                logits = logits.cpu()

                all_feat.append(mid_feat)
                all_logits.append(logits)
                all_pred.append(predicted)
                all_label.append(label)
                
    if atom_analysis:
        all_feat = torch.cat(all_feat)
        np.save('all_feat.npy', all_feat)
        # all_logits = torch.cat(all_logits)
        # np.save('all_logits.npy', all_logits)

        all_pred = np.concatenate(all_pred)
        all_label = np.concatenate(all_label)
        print(classification_report(all_label, all_pred))
        conf_matrix = confusion_matrix(all_label, all_pred)
        f1 = f1_score(all_label, all_pred, average=None)
        np.save('all_f1.npy', f1)

        class_counts = np.sum(conf_matrix, axis=1)
        num_most_common_classes = 25
        most_common_classes_indices = np.argsort(class_counts)[::-1][:num_most_common_classes]
        filtered_conf_matrix = conf_matrix[most_common_classes_indices][:, most_common_classes_indices]

        all_label = [Chem.Atom(int(item.item())).GetSymbol() for item in all_label]
        np.save('all_label.npy', all_label)

        print(filtered_conf_matrix)
        np.save('confusion_matrix.npy', conf_matrix)
        
        all_label = np.sort(np.unique(all_label))
        np.save('conf_label.npy', all_label)


    atom_accuracy = all_correct_atom / total_atom
    print(all_correct_atom)
    print(total_atom)
    mol_accuracy = correct_mol / total_mol
    total_mse = total_mse / total_mol
    print(bad_label)
    print(missing_cnt)
    print(f'{load_model_path}: Atom Test Accuracy: {atom_accuracy * 100:.2f}%, Mol Test Accuracy: {mol_accuracy * 100:.2f}%')
        


if __name__ == "__main__":
    main()
