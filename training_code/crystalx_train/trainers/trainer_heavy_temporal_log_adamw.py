import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from rdkit import Chem
from sklearn.metrics import classification_report

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from crystalx_train.common.utils import *  # cdist / copy_file / update_shelxt
from crystalx_train.models.torchmd_et import TorchMD_ET
from crystalx_train.models.noise_output_model import EquivariantScalar
from crystalx_train.models.torchmd_net import TorchMD_Net


device = torch.device("cuda")
print(device)


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42, deterministic_cudnn: bool = False):
    """固定随机种子（不改变训练逻辑，只让随机过程可复现）"""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_run_timestamp():
    """生成时间戳字符串（优先按 Asia/Singapore；不支持则用本地时间）"""
    try:
        from zoneinfo import ZoneInfo  # py>=3.9
        now = datetime.now(ZoneInfo("Asia/Singapore"))
    except Exception:
        now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


# -------------------------
# Data split
# -------------------------
def split_by_year_txt(
    txt_path: str,
    pt_dir: str,
    test_years=(2022, 2023, 2024),
    pt_prefix="equiv_",
    pt_suffix=".pt",
    strict=False,  # strict=True: 只使用txt里出现过的样本；False: txt没提到的pt也进train
):
    test_years = set(str(y) for y in test_years)

    train_files, test_files = [], []
    missing = []
    seen = set()

    with open(txt_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # txt: year  timestamp  cifname（tab/空格分隔）
            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] txt第{ln}行格式不对：{line}")
                continue

            year = parts[0]
            cif_name = parts[-1]  # 最后一列：7248215.cif
            cif_stem = os.path.splitext(os.path.basename(cif_name))[0]  # 7248215

            pt_path = os.path.join(pt_dir, f"{pt_prefix}{cif_stem}{pt_suffix}")

            if pt_path in seen:
                continue
            seen.add(pt_path)

            if not os.path.exists(pt_path):
                missing.append(pt_path)
                continue

            if year in test_years:
                test_files.append(pt_path)
            else:
                train_files.append(pt_path)

    if not strict:
        # 把目录里所有pt中“txt没列出”的补进train（但不会补进test）
        all_pt = [
            os.path.join(pt_dir, fn)
            for fn in os.listdir(pt_dir)
            if fn.endswith(pt_suffix)
        ]
        test_set = set(test_files)
        listed_set = set(train_files) | test_set
        extra = [p for p in all_pt if p not in listed_set and p not in test_set]
        train_files += extra

    return train_files, test_files, missing


def split_test(file_list, test_num=1000, random_state=42):
    random.seed(random_state)
    split_index = len(file_list) - test_num
    random.shuffle(file_list)
    train_file = file_list[:split_index]
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


# -------------------------
# Dataset building
# -------------------------
def build_simple_in_memory_dataset(file_list, is_eval=False, is_check_dist=True, is_acc=True):
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

        noise = mol_info["noise_list"]
        if np.max(np.abs(noise)) > 0.1:
            noise_cnt += 1
            continue

        z = mol_info["z"]
        y = mol_info["gt"]
        z = [item.capitalize() for item in z]
        y = [item.capitalize() for item in y]
        try:
            _z = [Chem.Atom(item).GetAtomicNum() for item in z]
            y = [Chem.Atom(item).GetAtomicNum() for item in y]
        except Exception as e:
            print(e)
            element_error_cnt += 1
            continue

        max_h = max(max_h, max(y))

        _real_cart = mol_info["pos"]
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
        mask[: len(y)] = 1
        mask = torch.from_numpy(mask).bool()

        if is_check_dist:
            distance_matrix = cdist(real_cart, real_cart)
            distance_matrix = distance_matrix + 10 * np.eye(distance_matrix.shape[0])
            if np.min(distance_matrix) < 0.1:
                dist_error_cnt += 1
                print("lalala")
                continue

        if is_eval:
            all_z = []
            all_z.append(sorted(y, reverse=True))
        else:
            all_z = []
            all_z.append(sorted(y, reverse=True))
            all_z.append(z[: len(y)])

        y = torch.tensor(y)

        if is_acc:
            correct_atom = (torch.tensor(z)[mask] == y).sum()
            mol_atom = y.shape[0]
            if correct_atom == mol_atom:
                correct_mol += 1
            total_mol += 1

        real_cart = torch.from_numpy(real_cart.astype(np.float32))
        for pz in all_z:
            pz += z[len(y) :]
            if len(pz) != len(real_cart):
                print("lalala")
            pz = torch.tensor(pz)
            dataset.append(Data(z=pz, y=y, pos=real_cart, fname=fname, mask=mask))
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


# -------------------------
# Logging helpers
# -------------------------
def write_hparams(log_f, hparams: dict):
    log_f.write("---- Hyperparameters ----\n")
    log_f.write(json.dumps(hparams, indent=2, ensure_ascii=False) + "\n")
    log_f.write("-------------------------\n\n")


@torch.no_grad()
def enforce_coverage_by_prob(prob, sfac, pred):
    """
    prob: [N, K] softmax over sfac-subspace
    sfac: [K] unique(gt) atomic numbers
    pred: [N] prediction in sfac space
    Goal: make every element in sfac appear at least once, without removing an already singleton element.
    """
    device = pred.device
    n, k = prob.shape
    if n == 0 or k == 0 or n < k:
        return pred

    elem2k = {int(sfac[j].item()): j for j in range(k)}
    pred_k = torch.tensor([elem2k[int(x.item())] for x in pred], device=device, dtype=torch.long)

    counts = torch.bincount(pred_k, minlength=k).clone()
    used_atoms = set()
    missing = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()

    while missing:
        changed_any = False
        for miss_k in missing:
            safe_mask = counts[pred_k] > 1

            if used_atoms:
                used = torch.zeros(n, device=device, dtype=torch.bool)
                used[list(used_atoms)] = True
                safe_mask = safe_mask & (~used)

            if not torch.any(safe_mask):
                return pred

            target_prob = prob[:, miss_k]
            cost = (-target_prob).masked_fill(~safe_mask, float("inf"))
            i = int(torch.argmin(cost).item())

            old_k = int(pred_k[i].item())
            if old_k == miss_k:
                continue

            pred[i] = sfac[miss_k]
            pred_k[i] = miss_k
            used_atoms.add(i)

            counts[old_k] -= 1
            counts[miss_k] += 1
            changed_any = True

        if not changed_any:
            break
        missing = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()

    return pred


# -------------------------
# Evaluation (保持与你原代码一致的“两次推理”流程)
# -------------------------
def eval_two_pass_like_original(
    model,
    test_loader,
    criterion,
    dump_test_data=False,
    atom_analysis=False,
    allow_one_mismatch=False,  # False: correct_atom==mol_atom；True: correct_atom==mol_atom 或 mol_atom-1
):
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
            logits = outputs

            sfac = torch.unique(data.y)
            logits = logits[:, sfac]
            logits = F.softmax(logits, dim=-1)

            _, predicted = torch.max(logits, dim=1)
            predicted = sfac[predicted]

            outputs, _ = model(predicted, data.pos, data.batch)
            outputs = outputs[data.mask]
            logits = outputs

            val_loss = criterion(logits, data.y)  # 原代码计算但不使用

            sfac = torch.unique(data.y)
            logits = logits[:, sfac]
            logits = F.softmax(logits, dim=-1)

            _, predicted = torch.max(logits, dim=1)
            predicted = sfac[predicted]
            predicted = enforce_coverage_by_prob(logits, sfac, predicted)

            label = data.y
            correct_atom = (predicted == label).sum().item()

            all_correct_atom += correct_atom
            mol_atom = label.shape[0]
            total_atom += mol_atom

            if allow_one_mismatch:
                ok = (correct_atom == mol_atom) or (correct_atom == mol_atom - 1)
            else:
                ok = (correct_atom == mol_atom)

            if ok:
                correct_mol += 1
            else:
                # 原代码里最终评估会取 indices；训练中验证也会走到 dump 分支
                indices = torch.nonzero(predicted != label)

                if dump_test_data:
                    predicted_cpu = predicted.to("cpu")
                    predicted_sym = [Chem.Atom(int(item.item())).GetSymbol() for item in predicted_cpu]

                    dir_list = [
                        "all_xrd_dataset_test",
                        "all_xrd_dataset_test_2",
                        "all_xrd_dataset_test_3",
                        "all_xrd_dataset_test_4",
                    ]
                    fname = os.path.basename(data.fname[0])
                    fname = os.path.splitext(fname)[0]
                    for dir in dir_list:
                        res_file_path = f"{dir}/{fname}/{fname}_a.res"
                        hkl_file_path = f"{dir}/{fname}/{fname}_a.hkl"
                        if os.path.exists(res_file_path) and os.path.exists(hkl_file_path):
                            break

                    refined_dir = "new_bad_main"
                    os.makedirs(refined_dir, exist_ok=True)
                    os.makedirs(f"{refined_dir}/{fname}", exist_ok=True)
                    new_hkl_file_path = f"{refined_dir}/{fname}/{fname}_AI.hkl"
                    copy_res_file_path = f"{refined_dir}/{fname}/{fname}_a.res"
                    copy_file(hkl_file_path, new_hkl_file_path)
                    copy_file(res_file_path, copy_res_file_path)
                    new_res_file_path = f"{refined_dir}/{fname}/{fname}_AI.ins"
                    update_shelxt(res_file_path, new_res_file_path, predicted_sym)

                    data_cpu = data.to("cpu")
                    d1 = Data(z=predicted_sym, y=data_cpu.y, pos=data_cpu.pos, fname=data_cpu.fname[0])
                    torch.save(d1, f"{refined_dir}/{fname}/mol_{fname}.pt")

            total_mol += 1

            if atom_analysis:
                predicted_cpu = predicted.cpu()
                label_cpu = label.cpu()
                all_pred.append(predicted_cpu)
                all_label.append(label_cpu)

    if atom_analysis:
        all_pred = np.concatenate(all_pred)
        all_label = np.concatenate(all_label)
        print(classification_report(all_label, all_pred))

    atom_accuracy = all_correct_atom / total_atom
    mol_accuracy = correct_mol / total_mol
    total_mse = total_mse / total_mol
    print(missing_cnt)

    # 注意：训练中验证后原代码会 model.train()；最终评估后原代码不再切回 train
    return atom_accuracy, mol_accuracy, total_mse


# -------------------------
# Main
# -------------------------
def main():
    # -------------------------
    # seed
    # -------------------------
    seed = 150
    deterministic_cudnn = False
    set_seed(seed, deterministic_cudnn=deterministic_cudnn)

    # -------------------------
    # paths
    # -------------------------
    pt_dir = "/inspire/ssd/project/project-public/zhengkaipeng-240108120123/all_materials/data/all_anno_density"
    txt_path = "sorted_by_journal_year.txt"

    run_ts = get_run_timestamp()
    metric_log_path = f"train_metric_log_{run_ts}.txt"

    # 你的设定
    test_years = (2020, 2021, 2022, 2023, 2024)

    model_save_name = f"torchmd-net-{run_ts}.pth"
    load_model_path = None

    # -------------------------
    # optimizer / schedule hparams
    # -------------------------
    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # ✅ AdamW
    weight_decay = 1e-2

    # ✅ warmup（按 epoch，不改你原来每个 epoch 调 scheduler.step() 的位置）
    epochs = 100
    warmup_epochs = 0
    min_lr = 0.0

    validation_interval = 500

    # model hparams（保持你原来的 TorchMD_ET 配置）
    rep_hparams = dict(
        hidden_channels=512,
        attn_activation="silu",
        num_heads=8,
        distance_influence="both",
    )

    # -------------------------
    # open log
    # -------------------------
    log_f = open(metric_log_path, "w", encoding="utf-8", buffering=1)
    log_f.write("==== New Run ====\n")
    log_f.write(f"Run timestamp: {run_ts}\n")
    log_f.write(f"Device: {device}\n\n")

    # -------------------------
    # split
    # -------------------------
    all_train_file, all_test_file, missing = split_by_year_txt(
        txt_path=txt_path,
        pt_dir=pt_dir,
        test_years=test_years,
        pt_prefix="equiv_",
        pt_suffix=".pt",
        strict=False,
    )

    print(f"Train Data: {len(all_train_file)}")
    print(f"Test  Data: {len(all_test_file)}")

    log_f.write(f"Test years: {list(test_years)}\n")
    log_f.write("Train years: all other years\n")
    log_f.write(f"Train Data: {len(all_train_file)} | Test Data: {len(all_test_file)}\n\n")

    if missing:
        print(f"[WARN] txt中映射到pt但不存在的数量: {len(missing)}")
        print("  示例(最多10条):")
        for p in missing[:10]:
            print("   ", p)

    # -------------------------
    # build dataset（保持原逻辑）
    # -------------------------
    train_dataset, num_classes, _ = build_simple_in_memory_dataset(all_train_file, is_eval=False)
    test_dataset, _, init_mol_acc = build_simple_in_memory_dataset(all_test_file, is_eval=True)

    num_classes = 98  # 保持你原代码：覆盖为 98
    print(f"Total class: {num_classes}")
    print(f"Initial Mol Test Accuracy: {init_mol_acc * 100:.2f}%")

    # -------------------------
    # log hyperparams（新增：超参记录）
    # -------------------------
    hparams = {
        "seed": seed,
        "deterministic_cudnn": deterministic_cudnn,
        "paths": {"pt_dir": pt_dir, "txt_path": txt_path},
        "split": {"test_years": list(test_years), "strict": False},
        "data": {"train_size": len(all_train_file), "test_size": len(all_test_file)},
        "training": {
            "num_classes": num_classes,
            "epochs": epochs,
            "validation_interval": validation_interval,
            "batch_size_train": 16,
            "batch_size_test": 1,
            "k_fold": 10,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": learning_rate,
            "betas": [beta1, beta2],
            "eps": epsilon,
            "weight_decay": weight_decay,
        },
        "lr_schedule": {
            "name": "Warmup + Cosine (LambdaLR)",
            "warmup_epochs": warmup_epochs,
            "min_lr": min_lr,
        },
        "model": {
            "representation_model": "TorchMD_ET",
            "representation_hparams": rep_hparams,
            "output_model": "EquivariantScalar",
            "output_in_dim": rep_hparams["hidden_channels"],
            "output_num_classes": num_classes,
            "wrapper": "TorchMD_Net",
        },
        "checkpoint": {"model_save_name": model_save_name, "load_model_path": load_model_path},
        "logs": {"metric_log_path": metric_log_path},
    }
    write_hparams(log_f, hparams)

    # -------------------------
    # training（保持原结构）
    # -------------------------

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    representation_model = TorchMD_ET(
        hidden_channels=rep_hparams["hidden_channels"],
        attn_activation=rep_hparams["attn_activation"],
        num_heads=rep_hparams["num_heads"],
        distance_influence=rep_hparams["distance_influence"],
    )
    output_model = EquivariantScalar(rep_hparams["hidden_channels"], num_classes=num_classes)

    model = TorchMD_Net(representation_model=representation_model, output_model=output_model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    max_mol_accuracy = -1

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    # ✅ AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
    )

    # ✅ warmup + cosine（epoch 级；不改你原来的 scheduler.step() 位置）
    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)

        if epochs <= warmup_epochs:
            return 1.0

        progress = (epoch - warmup_epochs) / float(epochs - warmup_epochs)  # 0->1
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))               # 1->0
        min_scale = (min_lr / learning_rate) if learning_rate > 0 else 0.0
        return min_scale + (1.0 - min_scale) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    model.train()
    for epoch in range(epochs):
        for iteration, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)

            outputs, _ = model(data.z, data.pos, data.batch)
            outputs = outputs[data.mask]
            logits = outputs

            loss = criterion(logits, data.y)

            if torch.isnan(loss):
                print("Loss contains NaN!")
                break

            loss.backward()
            optimizer.step()

            # validate（保持原逻辑：每 validation_interval 做一次）
            if epoch > -1 and (iteration + 1) % validation_interval == 0:
                atom_analysis = False
                dump_test_data = False

                atom_accuracy, mol_accuracy, total_mse = eval_two_pass_like_original(
                    model=model,
                    test_loader=test_loader,
                    criterion=criterion,
                    dump_test_data=dump_test_data,
                    atom_analysis=atom_analysis,
                    allow_one_mismatch=False,  # 训练中验证：原逻辑严格 mol 全对
                )

                model.train()  # 原代码验证后会切回 train

                if mol_accuracy > max_mol_accuracy:
                    max_mol_accuracy = mol_accuracy
                    torch.save(model.state_dict(), model_save_name)

                msg1 = (
                    f"Epoch {epoch}, Iteration {iteration + 1}: Loss: {loss.item()}, "
                    f"Atom Test Accuracy: {atom_accuracy * 100:.2f}%, "
                    f"Mol Test Accuracy: {mol_accuracy * 100:.2f}%, "
                    f"Noise MSE: {total_mse:.4f}"
                )
                msg2 = f"Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%"

                print(msg1)
                print(msg2)
                log_f.write(msg1 + "\n")
                log_f.write(msg2 + "\n")

        scheduler.step()

    # 最终评估（保持原逻辑：允许 mol 少一个原子也算对）
    atom_analysis = False
    dump_test_data = False

    model.eval()
    atom_accuracy, mol_accuracy, total_mse = eval_two_pass_like_original(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        dump_test_data=dump_test_data,
        atom_analysis=atom_analysis,
        allow_one_mismatch=True,  # 最终统计：mol_atom 或 mol_atom-1
    )

    msg1 = (
        f"Atom Test Accuracy: {atom_accuracy * 100:.2f}%, "
        f"Mol Test Accuracy: {mol_accuracy * 100:.2f}%, "
        f"Noise MSE: {total_mse:.4f}"
    )
    msg2 = f"Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%"

    print(msg1)
    print(msg2)

    log_f.write(msg1 + "\n")
    log_f.write(msg2 + "\n")

    log_f.close()


if __name__ == "__main__":
    main()
