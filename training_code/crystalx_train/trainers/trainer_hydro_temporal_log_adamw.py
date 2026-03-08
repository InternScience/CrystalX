import os
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
from scipy.spatial.distance import cdist

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from crystalx_train.models.torchmd_et import TorchMD_ET
from crystalx_train.models.noise_output_model import EquivariantScalar
from crystalx_train.models.torchmd_net import TorchMD_Net

from crystalx_train.common.utils import *  # copy_file / get_bond / gen_hfix_ins / update_shelxt_hydro


# =========================
# Device
# =========================
device = torch.device("cuda")
print(device)


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42, deterministic_cudnn: bool = False):
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


# =========================
# Split by year (和第一份脚本一致)
# =========================
def split_by_year_txt(
    txt_path: str,
    pt_dir: str,
    test_years=(2022, 2023, 2024),
    pt_prefix="",
    pt_suffix=".pt",
    strict=False,  # strict=True: 只使用 txt 里列出的样本；False: txt 没提到的 pt 补进 train
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
        # 把目录里所有 pt 中“txt没列出”的补进 train（但不会补进 test）
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


def _stem_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[6:] if stem.startswith("equiv_") else stem


def load_excluded_stems(txt_path: str):
    stems = set()
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            base = os.path.basename(s)
            stem = os.path.splitext(base)[0]
            if stem.startswith("equiv_"):
                stem = stem[6:]
            stems.add(stem)
    return stems


# =========================
# Cross-val (保持原逻辑)
# =========================
def cross_val(train_list, k=5, test_id=0):
    part_size = len(train_list) // k + 1
    split_lists = [train_list[i:i + part_size] for i in range(0, len(train_list), part_size)]

    val_ = split_lists[test_id]

    train_ = []
    for item in split_lists[:test_id]:
        train_ += item
    for item in split_lists[(test_id + 1):]:
        train_ += item
    return train_, val_


# =========================
# Dataset building (保持原逻辑不变)
# =========================
def build_simple_in_memory_dataset(
    file_list,
    is_check_dist=False,
    is_filter=False,
    extra_atom_dist_thresh=None,
):
    dataset = []
    cnt = 0
    max_h = -1
    equiv_cnt = 0
    dist_error_cnt = 0
    c6_to_c3_fix_cnt = 0
    extra_atom_drop_cnt = 0

    for fname in tqdm(file_list):
        mol_info = torch.load(fname)

        z = mol_info["equiv_gt"]
        y = mol_info["gt"]
        z = [item.capitalize() for item in z]
        y = [item.capitalize() for item in y]
        try:
            _z = [Chem.Atom(item).GetAtomicNum() for item in z]
            y = [Chem.Atom(item).GetAtomicNum() for item in y]
        except Exception:
            continue

        hydro_num = mol_info["hydro_gt"]
        if isinstance(hydro_num, torch.Tensor):
            hydro_num = hydro_num.detach().cpu().tolist()
        elif isinstance(hydro_num, np.ndarray):
            hydro_num = hydro_num.tolist()
        hydro_num = [int(h) for h in hydro_num]

        # Label correction: if the corresponding main atom is Carbon and hydro_gt is 6, set it to 3.
        pair_n = min(len(y), len(hydro_num))
        for i in range(pair_n):
            if y[i] == 6 and hydro_num[i] == 6:
                hydro_num[i] = 3
                c6_to_c3_fix_cnt += 1

        max_h = max(max_h, max(hydro_num))

        if is_filter:
            if max(hydro_num) > 4:
                continue
            if 6 not in _z or 7 not in _z or 8 not in _z:
                continue

        _real_cart = mol_info["pos"]
        real_cart = []
        z = []
        main_z = []
        for i in range(_real_cart.shape[0]):
            if _real_cart[i].tolist() not in real_cart:
                real_cart.append(_real_cart[i].tolist())
                main_z.append(_z[i])
                z.append(_z[i])
        real_cart = np.array(real_cart)

        # For symmetry-expanded atoms beyond main atom count (len(hydro_num)),
        # keep only those close to at least one main atom.
        if (
            extra_atom_dist_thresh is not None
            and float(extra_atom_dist_thresh) > 0
            and real_cart.shape[0] > len(hydro_num)
        ):
            main_n = min(len(hydro_num), real_cart.shape[0])
            extra_n = real_cart.shape[0] - main_n
            if main_n > 0 and extra_n > 0:
                main_cart = real_cart[:main_n]
                extra_cart = real_cart[main_n:]
                dist_extra_to_main = cdist(extra_cart, main_cart)
                keep_extra = np.min(dist_extra_to_main, axis=1) < float(extra_atom_dist_thresh)
                keep_idx = list(range(main_n)) + [main_n + i for i, k in enumerate(keep_extra.tolist()) if k]
                dropped_now = int(real_cart.shape[0] - len(keep_idx))
                if dropped_now > 0:
                    extra_atom_drop_cnt += dropped_now
                    real_cart = real_cart[keep_idx]
                    z = [z[i] for i in keep_idx]
                    main_z = [main_z[i] for i in keep_idx]

        mask = np.array([0] * len(z))
        mask[: len(hydro_num)] = 1
        mask = torch.from_numpy(mask).bool()

        if is_check_dist:
            distance_matrix = cdist(real_cart, real_cart)
            distance_matrix = distance_matrix + 10 * np.eye(distance_matrix.shape[0])
            if np.min(distance_matrix) < 0.1:
                dist_error_cnt += 1
                print("lalala")
                continue

        z = torch.tensor(z)
        y = torch.tensor(y)
        main_z = torch.tensor(main_z)
        real_cart = torch.from_numpy(real_cart.astype(np.float32))
        hydro_num = torch.tensor(hydro_num)

        dataset.append(
            Data(
                z=z,
                main_z=main_z,
                y=hydro_num,
                pos=real_cart,
                main_gt=y,
                fname=fname,
                mask=mask,
            )
        )
        cnt += 1

    print(cnt)
    print(equiv_cnt)
    print(dist_error_cnt)
    print(f"C(H6)->C(H3) fixes: {c6_to_c3_fix_cnt}")
    if extra_atom_dist_thresh is not None and float(extra_atom_dist_thresh) > 0:
        print(
            f"Extra atoms dropped by distance threshold (<{float(extra_atom_dist_thresh):.3f}A to any main atom): "
            f"{extra_atom_drop_cnt}"
        )
    return dataset, max_h


# =========================
# Logging helpers (和第一份脚本一致)
# =========================
def write_hparams(log_f, hparams: dict):
    log_f.write("---- Hyperparameters ----\n")
    log_f.write(json.dumps(hparams, indent=2, ensure_ascii=False) + "\n")
    log_f.write("-------------------------\n\n")


# =========================
# Validation (保持原逻辑不变)
# =========================
def eval_validate(model, test_loader, dump_test_data=False, atom_analysis=False):
    missing_cnt = 0
    model.eval()
    correct_predictions = 0
    correct_mol = 0
    total_atoms = 0
    total_mol = 0
    all_pred = []
    all_label = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            try:
                data = data.to(device)
                outputs, _ = model(data.z, data.pos, data.batch)   # ✅ 解包
                pred = outputs[data.mask]
                pred = F.softmax(pred, dim=-1)

            except Exception as e:
                print(e)
                missing_cnt += 1
                continue

            # 原逻辑：只统计 main 元素完全正确的样本
            main_correct = (data.main_gt == data.main_z[data.mask]).all()
            if not main_correct:
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
                refined_dir = "all_refined_10"
                fname = os.path.basename(data.fname[0])
                hkl_file_path = f"{refined_dir}/{fname}/{fname}_AI.hkl"
                ins_file_path = f"{refined_dir}/{fname}/{fname}_AI.ins"
                lst_file_path = f"{refined_dir}/{fname}/{fname}_AI.lst"
                res_file_path = f"{refined_dir}/{fname}/{fname}_AI.res"
                new_hkl_file_path = f"{refined_dir}/{fname}/{fname}_AIhydro.hkl"
                new_res_file_path = f"{refined_dir}/{fname}/{fname}_AIhydro.ins"
                predicted_cpu = predicted.to("cpu")

                copy_file(hkl_file_path, new_hkl_file_path)
                mol_graph = get_bond(lst_file_path)
                hfix_ins = gen_hfix_ins(ins_file_path, mol_graph, predicted_cpu)
                update_shelxt_hydro(res_file_path, new_res_file_path, hfix_ins)

            if atom_analysis:
                all_pred.append(predicted.cpu())
                all_label.append(label.cpu())

    if atom_analysis and len(all_pred) > 0:
        from sklearn.metrics import classification_report
        all_pred = np.concatenate(all_pred)
        all_label = np.concatenate(all_label)
        print(classification_report(all_label, all_pred))

    atom_accuracy = correct_predictions / total_atoms if total_atoms > 0 else 0.0
    mol_accuracy = correct_mol / total_mol if total_mol > 0 else 0.0

    model.train()
    return atom_accuracy, mol_accuracy


# =========================
# Main
# =========================
def main():
    # -------------------------
    # seed
    # -------------------------
    seed = 183
    deterministic_cudnn = False
    set_seed(seed, deterministic_cudnn=deterministic_cudnn)

    # -------------------------
    # paths
    # -------------------------
    pt_dir = "/inspire/ssd/project/project-public/zhengkaipeng-240108120123/all_materials/data/all_anno_density"
    txt_path = "sorted_by_journal_year.txt"
    exclude_disorder_txt = ""

    test_years = (2019, 2020, 2021, 2022, 2023, 2024)
    pt_prefix = "equiv_"      # 如果实际文件名是 equiv_xxx.pt，就改成 "equiv_"
    pt_suffix = ".pt"

    run_ts = get_run_timestamp()
    metric_log_path = f"train_hydro_metric_log_{run_ts}.txt"
    model_save_name = f"torchmd-hydro-{run_ts}.pth"
    load_model_path = None

    # -------------------------
    # training hparams（保持你 hydro 训练的设置）
    # -------------------------
    epochs = 100
    validation_interval = 1000
    batch_size_train = 16
    batch_size_test = 1
    extra_atom_dist_thresh = 3.2

    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 1e-2  # ✅ AdamW

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
    # split by year (像第一份脚本)
    # -------------------------
    train_files, test_files, missing = split_by_year_txt(
        txt_path=txt_path,
        pt_dir=pt_dir,
        test_years=test_years,
        pt_prefix=pt_prefix,
        pt_suffix=pt_suffix,
        strict=False,
    )

    excluded_train = 0
    excluded_test = 0
    if exclude_disorder_txt:
        excluded_stems = load_excluded_stems(exclude_disorder_txt)
        old_train_n = len(train_files)
        old_test_n = len(test_files)
        train_files = [p for p in train_files if _stem_from_path(p) not in excluded_stems]
        test_files = [p for p in test_files if _stem_from_path(p) not in excluded_stems]
        excluded_train = old_train_n - len(train_files)
        excluded_test = old_test_n - len(test_files)

    print(f"Train Data: {len(train_files)}")
    print(f"Test  Data: {len(test_files)}")
    print(f"Excluded disorder from train/test: {excluded_train}/{excluded_test}")

    log_f.write(f"Test years: {list(test_years)}\n")
    log_f.write("Train years: all other years\n")
    log_f.write(f"Train Data: {len(train_files)} | Test Data: {len(test_files)}\n\n")
    log_f.write(f"Exclude disorder txt: {exclude_disorder_txt}\n")
    log_f.write(f"Excluded disorder from train/test: {excluded_train}/{excluded_test}\n\n")

    if missing:
        print(f"[WARN] txt中映射到pt但不存在的数量: {len(missing)}")
        for p in missing[:10]:
            print("   ", p)

    # -------------------------
    # build dataset（保持原逻辑）
    # -------------------------
    train_dataset, max_h_train = build_simple_in_memory_dataset(
        train_files,
        is_check_dist=True,
        is_filter=False,
        extra_atom_dist_thresh=extra_atom_dist_thresh,
    )
    num_classes = max_h_train + 1
    print("num_classes:", num_classes)

    test_dataset, _ = build_simple_in_memory_dataset(
        test_files,
        is_check_dist=True,
        is_filter=False,
        extra_atom_dist_thresh=extra_atom_dist_thresh,
    )

    # -------------------------
    # log hyperparams
    # -------------------------
    hparams = {
        "seed": seed,
        "deterministic_cudnn": deterministic_cudnn,
        "paths": {"pt_dir": pt_dir, "txt_path": txt_path},
        "split": {
            "test_years": list(test_years),
            "pt_prefix": pt_prefix,
            "pt_suffix": pt_suffix,
            "strict": False,
        },
        "data": {"train_files": len(train_files), "test_files": len(test_files)},
        "training": {
            "task": "predict_hydro_num_per_atom",
            "num_classes": num_classes,
            "epochs": epochs,
            "validation_interval": validation_interval,
            "batch_size_train": batch_size_train,
            "batch_size_test": batch_size_test,
            "dist_check": True,
            "filter": False,
            "extra_atom_dist_thresh": extra_atom_dist_thresh,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": learning_rate,
            "betas": [beta1, beta2],
            "eps": epsilon,
            "weight_decay": weight_decay,
        },
        "lr_schedule": {"name": "CosineAnnealingLR", "T_max_epochs": epochs},
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
        "eval": {"rule": "only count samples where main_gt == main_z on masked atoms"},
    }
    write_hparams(log_f, hparams)

    # -------------------------
    # dataloader（像第一份脚本：一个 train_loader + 一个 test_loader）
    # -------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    # -------------------------
    # build model
    # -------------------------
    representation_model = TorchMD_ET(
        hidden_channels=rep_hparams["hidden_channels"],
        attn_activation=rep_hparams["attn_activation"],
        num_heads=rep_hparams["num_heads"],
        distance_influence=rep_hparams["distance_influence"],
    )
    output_model = EquivariantScalar(rep_hparams["hidden_channels"], num_classes=num_classes)
    model = TorchMD_Net(representation_model=representation_model, output_model=output_model)
    model.to(device)

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    criterion = nn.CrossEntropyLoss()

    # ✅ AdamW（像第一份脚本）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
    )

    # 保持你 hydro 原逻辑：CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # -------------------------
    # training（像第一份脚本：定期在 test 上 eval + 保存 best）
    # -------------------------
    max_mol_accuracy = -1.0

    model.train()
    for epoch in range(epochs):
        for iteration, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)

            # ✅ 关键：TorchMD_Net 返回 (outputs, extra)，必须解包
            outputs, _ = model(data.z, data.pos, data.batch)
            pred = outputs[data.mask]

            loss = criterion(pred, data.y)

            if torch.isnan(loss):
                print("Loss contains NaN!")
                break

            loss.backward()
            optimizer.step()

            if (iteration + 1) % validation_interval == 0:
                atom_acc, mol_acc = eval_validate(model, test_loader)

                if mol_acc > max_mol_accuracy:
                    max_mol_accuracy = mol_acc
                    torch.save(model.state_dict(), model_save_name)

                msg1 = (
                    f"Epoch {epoch}, Iteration {iteration + 1}: Loss: {loss.item()}, "
                    f"Atom Test Accuracy: {atom_acc * 100:.2f}%, "
                    f"Mol  Test Accuracy: {mol_acc * 100:.2f}%"
                )
                msg2 = f"Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%"
                print(msg1)
                print(msg2)
                log_f.write(msg1 + "\n")
                log_f.write(msg2 + "\n")

        scheduler.step()

    # -------------------------
    # final eval（像第一份脚本：训练结束再跑一次）
    # -------------------------
    model.eval()
    atom_acc, mol_acc = eval_validate(model, test_loader)
    msg1 = (
        f"[Final] Atom Test Accuracy: {atom_acc * 100:.2f}%, "
        f"Mol Test Accuracy: {mol_acc * 100:.2f}%"
    )
    msg2 = f"Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%"
    print(msg1)
    print(msg2)
    log_f.write(msg1 + "\n")
    log_f.write(msg2 + "\n")

    log_f.close()


if __name__ == "__main__":
    main()
