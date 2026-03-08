import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
from sklearn.metrics import classification_report
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

        mean = 0  
        std_dev = 0.0  
        gaussian_noise = np.random.normal(mean, std_dev, real_cart.shape)
        real_cart = real_cart + gaussian_noise

        mask = np.array([0] * len(z))
        mask[:len(y)] = 1
        mask = torch.from_numpy(mask).bool()

        if is_check_dist:
            distance_matrix = cdist(real_cart, real_cart)
            distance_matrix = distance_matrix + 10*np.eye(distance_matrix.shape[0])
            if np.min(distance_matrix) < 0.1:
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
            correct_atom = (torch.tensor(z)[mask] == y).sum()
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
            dataset.append(Data(z = pz, y = y, pos = real_cart, fname=fname, mask = mask))
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

    model_save_name = 'torchmd-net.pth'
    load_model_path = None

    # build dataset
    train_dataset, num_classes,  _ = build_simple_in_memory_dataset(all_train_file, is_eval = False)
    test_dataset, _, init_mol_acc = build_simple_in_memory_dataset(all_test_file, is_eval=True)

    num_classes = 98
    print(f'Total class: {num_classes}')
    print(f'Initial Mol Test Accuracy: {init_mol_acc * 100:.2f}%')

    k_fold = 10
    for i in range(k_fold):
        # wrapped in dataloader

        # train_dataset, test_dataset = cross_val(all_train_dataset, k = k_fold, test_id = i)

        # extra_node_feature_dim = train_dataset[0].node_feature.shape[1]
        # print(extra_node_feature_dim)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # build model
        # model = SchNet(num_classes=num_classes, hidden_channels=256)
        # model = DimeNet(num_classes=num_classes+3, hidden_channels=256, out_channels=256)
        # model = SphereNet(hidden_channels=128, out_channels=num_classes+3, cutoff=5.0, num_layers=4)
        # model = ComENet(cutoff=5.0, out_channels=num_classes+3)

        representation_model = TorchMD_ET(
            attn_activation='silu',
            num_heads=8,
            distance_influence='both',
            )
        # output_model = EquivariantScalar(256, num_classes=num_classes+3)
        output_model = EquivariantScalar(256, num_classes=num_classes)

        model = TorchMD_Net(representation_model=representation_model,
                            output_model=output_model)

        model.to(device)

        criterion = nn.CrossEntropyLoss()
        learning_rate = 1e-4
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        max_mol_accuracy = -1

        if load_model_path:
            model.load_state_dict(torch.load(load_model_path))
            # atom_accuracy, mol_accuracy, total_mse = validate(model, test_loader, dump_test_data=False, atom_analysis=False)
            # max_mol_accuracy = mol_accuracy
            # print(f'Atom Test Accuracy: {atom_accuracy * 100:.2f}%, Mol Test Accuracy: {mol_accuracy * 100:.2f}%, Noise MSE: {total_mse:.4f}')
            # print(f'Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%')

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon)
        epochs = 100
        validation_interval = 2000
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        model.train()
        for epoch in range(epochs):
            for iteration, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(device)

                outputs, _ = model(data.z, data.pos, data.batch)
                # outputs = model(data)

                outputs = outputs[data.mask]

                # outputs = model(data)
                # logits = outputs[:,:-3]
                # reg = outputs[:,-3:]
                logits = outputs

                loss = criterion(logits, data.y)
                # train_losses.append(loss)

                # temperature = 1
                # logits = F.softmax(logits * temperature, dim = -1)
                # onehot_label = F.one_hot(data.y, num_classes=num_classes).float()
                # # logits = torch.sigmoid(logits * temperature)

                # sfac = torch.unique(data.y)
                # _m = torch.zeros(logits.shape[-1]).to(device)
                # _m = _m.scatter_(0, sfac, 1).bool()

                # sfac_mask = torch.zeros_like(logits)
                # sfac_mask[:,_m] = 1

                # loss = F.binary_cross_entropy(logits*sfac_mask, onehot_label*sfac_mask)


                # mse_loss = F.mse_loss(reg, data.noise)
                # loss += mse_loss



                if torch.isnan(loss):
                    print("Loss contains NaN!")
                    break
                loss.backward()
                optimizer.step()

                # validate
                if epoch > -1 and (iteration+1) % validation_interval == 0:
                    atom_analysis = False
                    dump_test_data = False
                    missing_cnt = 0
                    model.eval()
                    all_correct_atom = 0
                    correct_mol = 0
                    total_atom = 0
                    total_mol = 0
                    total_mse = 0
                    all_pred = []
                    all_label = []
                    total_loss = 0
                    with torch.no_grad():
                        for data in tqdm(test_loader):
                            data = data.to(device)
                            outputs, _ = model(data.z, data.pos, data.batch)
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
                            outputs, _ = model(predicted, data.pos, data.batch)
                            outputs = outputs[data.mask]

                            # logits = outputs[:,:-3]
                            # reg = outputs[:,-3:]
                            # mse_loss = F.mse_loss(reg, data.noise)
                            # total_mse += mse_loss.item()

                            logits = outputs

                            val_loss = criterion(logits, data.y)
                            # val_losses.append(val_loss)

                            sfac = torch.unique(data.y)
                            logits = logits[:,sfac]
                            logits = F.softmax(logits, dim = -1)
                            _, predicted = torch.max(logits, dim=1)
                            predicted = sfac[predicted]

                            label = data.y
                            correct_atom = (predicted == label).sum().item()
                            all_correct_atom += correct_atom
                            mol_atom = label.shape[0]
                            total_atom += mol_atom
                            if correct_atom == mol_atom:
                                correct_mol += 1                            
                            else:
                                if dump_test_data:
                                    predicted = predicted.to('cpu')
                                    predicted = [Chem.Atom(int(item.item())).GetSymbol() for item in predicted]

                                    dir_list = ['all_xrd_dataset_test', 'all_xrd_dataset_test_2', 
                                           'all_xrd_dataset_test_3', 'all_xrd_dataset_test_4']
                                    fname = os.path.basename(data.fname[0])
                                    fname = os.path.splitext(fname)[0]
                                    for dir in dir_list:
                                        res_file_path = f'{dir}/{fname}/{fname}_a.res'
                                        hkl_file_path = f'{dir}/{fname}/{fname}_a.hkl'
                                        if os.path.exists(res_file_path) and os.path.exists(hkl_file_path):
                                            break

                                    refined_dir = 'new_bad_main'
                                    os.makedirs(refined_dir, exist_ok=True)
                                    os.makedirs(f'{refined_dir}/{fname}', exist_ok=True)
                                    new_hkl_file_path = f'{refined_dir}/{fname}/{fname}_AI.hkl'
                                    copy_res_file_path = f'{refined_dir}/{fname}/{fname}_a.res'
                                    copy_file(hkl_file_path, new_hkl_file_path)
                                    copy_file(res_file_path, copy_res_file_path)
                                    new_res_file_path = f'{refined_dir}/{fname}/{fname}_AI.ins'
                                    update_shelxt(res_file_path, new_res_file_path, predicted)

                                    data = data.to('cpu')
                                    d1 = Data(z = predicted, y = data.y, pos = data.pos, fname=data.fname[0])
                                    torch.save(d1, f'{refined_dir}/{fname}/mol_{fname}.pt')
                            total_mol += 1
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

                    # atom_accuracy, mol_accuracy, total_mse = validate(model, test_loader)
                    if mol_accuracy > max_mol_accuracy:
                        max_mol_accuracy = mol_accuracy
                        torch.save(model.state_dict(), model_save_name)
                    print(f'Epoch {epoch}, Iteration {iteration+1}: Loss: {loss.item()}, Atom Test Accuracy: {atom_accuracy * 100:.2f}%, Mol Test Accuracy: {mol_accuracy * 100:.2f}%, Noise MSE: {total_mse:.4f}')
                    print(f'Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%')

            scheduler.step()
        # model.load_state_dict(torch.load(model_save_name))
        # atom_accuracy, mol_accuracy, total_mse = validate(model, test_loader, dump_test_data=False, atom_analysis=False)
        
        # np.save('train_losses.npy', train_losses)
        # np.save('val_losses.npy', val_losses)

        atom_analysis = False
        dump_test_data = False
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
                outputs, _ = model(data.z, data.pos, data.batch)
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
                outputs, _ = model(predicted, data.pos, data.batch)
                outputs = outputs[data.mask]

                # logits = outputs[:,:-3]
                # reg = outputs[:,-3:]
                # mse_loss = F.mse_loss(reg, data.noise)
                # total_mse += mse_loss.item()

                logits = outputs
                sfac = torch.unique(data.y)
                logits = logits[:,sfac]
                logits = F.softmax(logits, dim = -1)
                _, predicted = torch.max(logits, dim=1)
                predicted = sfac[predicted]

                label = data.y
                correct_atom = (predicted == label).sum().item()

                all_correct_atom += correct_atom
                mol_atom = label.shape[0]
                total_atom += mol_atom
                # if correct_atom == mol_atom or correct_atom == mol_atom - 1:
                if correct_atom == mol_atom or correct_atom == mol_atom - 1:
                    correct_mol += 1                            
                else:
                    indices = torch.nonzero(predicted != label)
                    # print(predicted[indices])
                    # print(label[indices])
                    # print(logits[indices])
                    if dump_test_data:
                        predicted = predicted.to('cpu')
                        predicted = [Chem.Atom(int(item.item())).GetSymbol() for item in predicted]

                        dir_list = ['all_xrd_dataset_test', 'all_xrd_dataset_test_2', 
                                'all_xrd_dataset_test_3', 'all_xrd_dataset_test_4']
                        fname = os.path.basename(data.fname[0])
                        fname = os.path.splitext(fname)[0]
                        for dir in dir_list:
                            res_file_path = f'{dir}/{fname}/{fname}_a.res'
                            hkl_file_path = f'{dir}/{fname}/{fname}_a.hkl'
                            if os.path.exists(res_file_path) and os.path.exists(hkl_file_path):
                                break

                        refined_dir = 'new_bad_main'
                        os.makedirs(refined_dir, exist_ok=True)
                        os.makedirs(f'{refined_dir}/{fname}', exist_ok=True)
                        new_hkl_file_path = f'{refined_dir}/{fname}/{fname}_AI.hkl'
                        copy_res_file_path = f'{refined_dir}/{fname}/{fname}_a.res'
                        copy_file(hkl_file_path, new_hkl_file_path)
                        copy_file(res_file_path, copy_res_file_path)
                        new_res_file_path = f'{refined_dir}/{fname}/{fname}_AI.ins'
                        update_shelxt(res_file_path, new_res_file_path, predicted)

                        data = data.to('cpu')
                        d1 = Data(z = predicted, y = data.y, pos = data.pos, fname=data.fname[0])
                        torch.save(d1, f'{refined_dir}/{fname}/mol_{fname}.pt')
                total_mol += 1
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

        print(f'Atom Test Accuracy: {atom_accuracy * 100:.2f}%, Mol Test Accuracy: {mol_accuracy * 100:.2f}%, Noise MSE: {total_mse:.4f}')
        print(f'Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
