import os
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
from sklearn.metrics import classification_report
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
    for fname in tqdm(file_list):
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

    train_ = []
    for item in split_lists[:test_id]:
        train_ += item
    for item in split_lists[(test_id+1):]:
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

def validate(model, test_loader, dump_test_data = False, atom_analysis = False):
    missing_cnt = 0
    model.eval()
    correct_predictions = 0
    correct_mol = 0
    total_atoms = 0
    total_mol = 0
    mse = 0
    all_pred = []
    all_label = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            try:
                data = data.to(device)
                pred = model(data.z, data.pos, data.batch)
                pred = pred[data.mask]
                # pred = model(data)
                # loss = F.mse_loss(pred, data.noise)
                pred= F.softmax(pred, dim = -1)
            except Exception as e:
                print(e)
                missing_cnt += 1
                continue
            # mse += loss.item()
            _, predicted = torch.max(pred, dim=1)
            label = data.y
            correct_atom_num = (predicted == label).sum().item()
            correct_predictions += correct_atom_num
            all_atom_num = label.shape[0]
            total_atoms += all_atom_num
            if correct_atom_num == all_atom_num and all_atom_num > 0:
                correct_mol += 1
            total_mol += 1
            if dump_test_data:
                refined_dir = 'all_refined_10'
                fname = os.path.basename(data.fname[0])
                hkl_file_path = f'{refined_dir}/{fname}/{fname}_AI.hkl'
                ins_file_path = f'{refined_dir}/{fname}/{fname}_AI.ins'
                lst_file_path = f'{refined_dir}/{fname}/{fname}_AI.lst'
                res_file_path = f'{refined_dir}/{fname}/{fname}_AI.res'
                new_hkl_file_path = f'{refined_dir}/{fname}/{fname}_AIhydro.hkl'
                new_res_file_path = f'{refined_dir}/{fname}/{fname}_AIhydro.ins'
                predicted = predicted.to('cpu')
                copy_file(hkl_file_path, new_hkl_file_path)
                mol_graph = get_bond(lst_file_path)
                hfix_ins = gen_hfix_ins(ins_file_path, mol_graph, predicted)
                update_shelxt_hydro(res_file_path, new_res_file_path, hfix_ins)
            if atom_analysis:
                predicted = predicted.cpu()
                label = label.cpu()
                all_pred.append(predicted)
                all_label.append(label)
    if atom_analysis:
        all_pred = np.concatenate(all_pred)
        all_label = np.concatenate(all_label)
        print(classification_report(all_label, all_pred))
    atom_accuracy = correct_predictions / total_atoms
    mol_accuracy = correct_mol / total_mol
    # mse = mse / total_mol
    model.train()
    # return mse
    return atom_accuracy, mol_accuracy

def eval_validate(model, test_loader, dump_test_data = False, atom_analysis = False):
    missing_cnt = 0
    model.eval()
    correct_predictions = 0
    correct_mol = 0
    total_atoms = 0
    total_mol = 0
    mse = 0
    all_pred = []
    all_label = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            try:
                data = data.to(device)
                pred = model(data.z, data.pos, data.batch)
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
            _, predicted = torch.max(pred, dim=1)
            label = data.y
            correct_atom_num = (predicted == label).sum().item()
            correct_predictions += correct_atom_num
            all_atom_num = label.shape[0]
            total_atoms += all_atom_num
            if correct_atom_num == all_atom_num:
                correct_mol += 1
            total_mol += 1
            if dump_test_data:
                refined_dir = 'all_refined_10'
                fname = os.path.basename(data.fname[0])
                hkl_file_path = f'{refined_dir}/{fname}/{fname}_AI.hkl'
                ins_file_path = f'{refined_dir}/{fname}/{fname}_AI.ins'
                lst_file_path = f'{refined_dir}/{fname}/{fname}_AI.lst'
                res_file_path = f'{refined_dir}/{fname}/{fname}_AI.res'
                new_hkl_file_path = f'{refined_dir}/{fname}/{fname}_AIhydro.hkl'
                new_res_file_path = f'{refined_dir}/{fname}/{fname}_AIhydro.ins'
                predicted = predicted.to('cpu')
                copy_file(hkl_file_path, new_hkl_file_path)
                mol_graph = get_bond(lst_file_path)
                hfix_ins = gen_hfix_ins(ins_file_path, mol_graph, predicted)
                update_shelxt_hydro(res_file_path, new_res_file_path, hfix_ins)
            if atom_analysis:
                predicted = predicted.cpu()
                label = label.cpu()
                all_pred.append(predicted)
                all_label.append(label)
    if atom_analysis:
        all_pred = np.concatenate(all_pred)
        all_label = np.concatenate(all_label)
        print(classification_report(all_label, all_pred))
    atom_accuracy = correct_predictions / total_atoms
    mol_accuracy = correct_mol / total_mol
    # mse = mse / total_mol
    model.train()
    # return mse
    return atom_accuracy, mol_accuracy

def main():
    path = '/ailab/group/groups/ai4phys/zhengkaipeng/xrd/all_matched_density'
    all_train_file = [os.path.join(path, fname) for fname in os.listdir(path)]

    random.shuffle(all_train_file)
    all_test_file = all_train_file[-10000:]
    all_train_file = all_train_file[:-10000]
    
    print(f'Training Data: {len(all_train_file)}')
    print(f'Testing Data: {len(all_test_file)}')

    model_save_name = 'final_hydro_model_add_no_noise_fold'
    load_model_path = None


    # build dataset
    all_train_dataset, num_classes = build_simple_in_memory_dataset(all_train_file, is_check_dist = True, is_filter=False)
    num_classes += 1
    print(num_classes)
    all_test_dataset, _ = build_simple_in_memory_dataset(all_test_file, is_check_dist = True, is_filter=False)
    # num_classes = 5

    # extra_train_dataset, num_classes1 = build_simple_in_memory_dataset(extra_train_file, is_check_dist=True, is_filter=True)
    # num_classes = max(num_classes, num_classes1) + 1
    
    k_fold = 10
    for i in range(k_fold):
        # wrapped in dataloader
        train_dataset, test_dataset = cross_val(all_train_dataset, k = k_fold, test_id = i)

        # train_dataset = []
        # test_dataset = []
        # for data in all_train_dataset:
        #     fname = os.path.basename(data.fname)
        #     fname = os.path.splitext(fname)[0]
        #     if fname in os.listdir('all_refined_10'):
        #         test_dataset.append(data)
        #     else:
        #         train_dataset.append(data)


        # extra_node_feature_dim = train_dataset[0].node_feature.shape[1]
        # print(extra_node_feature_dim)

        # train_dataset += extra_train_dataset

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # build model
        # model = SchNet(num_classes=num_classes, hidden_channels=256)
        # model = DimeNet(num_classes=num_classes, hidden_channels=256, out_channels=256)
        # model = SphereNet(hidden_channels=256, out_channels=num_classes, cutoff=5.0, num_layers=4, 
        #                 use_node_features = True, use_extra_node_feature=False, extra_node_feature_dim=1)
        # model = ComENet(cutoff=5.0, out_channels=num_classes+3)
        representation_model = TorchMD_ET(
            attn_activation='silu',
            num_heads=8,
            distance_influence='both',
            )
        output_model = EquivariantScalar(256, num_classes=num_classes)
        model = TorchMD_Net(representation_model=representation_model,
                            output_model=output_model)
        model.to(device)
        # model.load_state_dict(torch.load('best_hydro_model_heavy.pth'))


        criterion = nn.CrossEntropyLoss()
        learning_rate = 1e-4
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        min_mse = 100
        max_mol_accuracy = -1

        if load_model_path:
            model.load_state_dict(torch.load(load_model_path))

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
        epochs = 80
        # refine_built = [1]
        validation_interval = 1000
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # training
        model.train()
        for epoch in range(epochs):
            for iteration, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(device)
                pred = model(data.z, data.pos, data.batch)
                pred = pred[data.mask]
                # real_atom_num = pred.shape[0]

                # pred = model(data)
                loss = criterion(pred, data.y)
                # loss = F.mse_loss(pred, data.noise)


                if torch.isnan(loss):
                    print("Loss contains NaN!")
                    break
                loss.backward()
                optimizer.step()

                # validate
                if (iteration+1) % validation_interval == 0:
                    atom_accuracy, mol_accuracy = eval_validate(model, test_loader)
                    if mol_accuracy > max_mol_accuracy:
                        max_mol_accuracy = mol_accuracy
                        torch.save(model.state_dict(), f'{model_save_name}_{i}.pth')
                    print(f'Fold {i} Epoch {epoch}, Iteration {iteration+1}: Loss: {loss.item()}, Atom Test Accuracy: {atom_accuracy * 100:.2f}%, Mol Test Accuracy: {mol_accuracy * 100:.2f}%')
                    print(f'Fold {i} Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%')
            scheduler.step()

if __name__ == "__main__":
    main()
