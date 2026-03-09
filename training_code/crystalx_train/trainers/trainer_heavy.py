import json
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from crystalx_train.models.noise_output_model import EquivariantScalar
from crystalx_train.models.torchmd_et import TorchMD_ET
from crystalx_train.models.torchmd_net import TorchMD_Net


DEVICE = torch.device("cuda")
print(DEVICE)


def set_seed(seed: int = 42, deterministic_cudnn: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_run_timestamp():
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("Asia/Singapore"))
    except Exception:
        now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def split_by_year_txt(
    txt_path: str,
    pt_dir: str,
    test_years=(2022, 2023, 2024),
    pt_prefix="equiv_",
    pt_suffix=".pt",
    strict=False,
):
    test_years = {str(year) for year in test_years}
    train_files, test_files, missing = [], [], []
    seen = set()

    with open(txt_path, "r", encoding="utf-8") as file_obj:
        for line_num, line in enumerate(file_obj, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] malformed line {line_num} in {txt_path}: {line}")
                continue

            year = parts[0]
            cif_stem = os.path.splitext(os.path.basename(parts[-1]))[0]
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
        listed_files = set(train_files) | set(test_files)
        extra_train_files = [
            os.path.join(pt_dir, file_name)
            for file_name in os.listdir(pt_dir)
            if file_name.endswith(pt_suffix) and os.path.join(pt_dir, file_name) not in listed_files
        ]
        train_files.extend(extra_train_files)

    return train_files, test_files, missing


def _deduplicate_positions(atom_numbers, positions):
    unique_pos = []
    unique_atoms = []

    for idx in range(positions.shape[0]):
        position = positions[idx].tolist()
        if position not in unique_pos:
            unique_pos.append(position)
            unique_atoms.append(atom_numbers[idx])

    return unique_atoms, np.asarray(unique_pos, dtype=np.float32)


def _is_distance_valid(real_cart):
    distance_matrix = cdist(real_cart, real_cart) + 10 * np.eye(real_cart.shape[0])
    return np.min(distance_matrix) >= 0.1


def build_simple_in_memory_dataset(file_list, is_eval=False, is_check_dist=True):
    dataset = []
    correct_mol = 0
    total_mol = 0
    dist_error_cnt = 0
    noise_cnt = 0
    element_error_cnt = 0
    sample_cnt = 0

    for fname in tqdm(file_list):
        mol_info = torch.load(fname)

        noise = mol_info["noise_list"]
        if np.max(np.abs(noise)) > 0.1:
            noise_cnt += 1
            continue

        atom_symbols = [item.capitalize() for item in mol_info["z"]]
        label_symbols = [item.capitalize() for item in mol_info["gt"]]
        try:
            atom_numbers = [Chem.Atom(item).GetAtomicNum() for item in atom_symbols]
            label_numbers = [Chem.Atom(item).GetAtomicNum() for item in label_symbols]
        except Exception:
            element_error_cnt += 1
            continue

        unique_atoms, real_cart = _deduplicate_positions(atom_numbers, mol_info["pos"])
        mask = torch.zeros(len(unique_atoms), dtype=torch.bool)
        mask[: len(label_numbers)] = True

        if is_check_dist and not _is_distance_valid(real_cart):
            dist_error_cnt += 1
            continue

        label = torch.tensor(label_numbers)
        candidate_atoms = [sorted(label_numbers, reverse=True)]
        if not is_eval:
            candidate_atoms.append(unique_atoms[: len(label_numbers)])

        current_atoms = torch.tensor(unique_atoms)
        correct_atom = (current_atoms[mask] == label).sum().item()
        if correct_atom == label.shape[0]:
            correct_mol += 1
        total_mol += 1

        real_cart_tensor = torch.from_numpy(real_cart)
        trailing_atoms = unique_atoms[len(label_numbers):]
        for atoms in candidate_atoms:
            dataset.append(
                Data(
                    z=torch.tensor(atoms + trailing_atoms),
                    y=label,
                    pos=real_cart_tensor,
                    fname=fname,
                    mask=mask,
                )
            )
        sample_cnt += 1

    mol_accuracy = correct_mol / total_mol if total_mol > 0 else 0.0
    print(sample_cnt)
    print(dist_error_cnt)
    print(noise_cnt)
    print(element_error_cnt)
    return dataset, mol_accuracy


def write_hparams(log_f, hparams: dict):
    log_f.write("---- Hyperparameters ----\n")
    log_f.write(json.dumps(hparams, indent=2, ensure_ascii=False) + "\n")
    log_f.write("-------------------------\n\n")


@torch.no_grad()
def enforce_coverage_by_prob(prob, sfac, pred):
    num_atoms, num_elements = prob.shape
    if num_atoms == 0 or num_elements == 0 or num_atoms < num_elements:
        return pred

    elem2idx = {int(sfac[idx].item()): idx for idx in range(num_elements)}
    pred_idx = torch.tensor(
        [elem2idx[int(value.item())] for value in pred],
        device=pred.device,
        dtype=torch.long,
    )

    counts = torch.bincount(pred_idx, minlength=num_elements).clone()
    used_atoms = set()
    missing = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()

    while missing:
        changed = False
        for miss_idx in missing:
            safe_mask = counts[pred_idx] > 1

            if used_atoms:
                used_mask = torch.zeros(num_atoms, device=pred.device, dtype=torch.bool)
                used_mask[list(used_atoms)] = True
                safe_mask = safe_mask & (~used_mask)

            if not torch.any(safe_mask):
                return pred

            target_prob = prob[:, miss_idx]
            cost = (-target_prob).masked_fill(~safe_mask, float("inf"))
            atom_idx = int(torch.argmin(cost).item())

            old_idx = int(pred_idx[atom_idx].item())
            if old_idx == miss_idx:
                continue

            pred[atom_idx] = sfac[miss_idx]
            pred_idx[atom_idx] = miss_idx
            used_atoms.add(atom_idx)
            counts[old_idx] -= 1
            counts[miss_idx] += 1
            changed = True

        if not changed:
            break
        missing = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()

    return pred


@torch.no_grad()
def eval_two_pass(model, test_loader, allow_one_mismatch=False):
    model.eval()
    all_correct_atom = 0
    correct_mol = 0
    total_atom = 0
    total_mol = 0

    for data in tqdm(test_loader):
        data = data.to(DEVICE)

        first_pass_logits, _ = model(data.z, data.pos, data.batch)
        sfac = torch.unique(data.y)
        first_pass_pred = sfac[first_pass_logits[:, sfac].argmax(dim=1)]

        second_pass_logits, _ = model(first_pass_pred, data.pos, data.batch)
        masked_logits = second_pass_logits[data.mask][:, sfac]
        masked_prob = F.softmax(masked_logits, dim=-1)
        predicted = sfac[masked_prob.argmax(dim=1)]
        predicted = enforce_coverage_by_prob(masked_prob, sfac, predicted)

        label = data.y
        correct_atom = (predicted == label).sum().item()
        mol_atom = label.shape[0]

        all_correct_atom += correct_atom
        total_atom += mol_atom
        total_mol += 1

        if allow_one_mismatch:
            is_correct = correct_atom >= mol_atom - 1
        else:
            is_correct = correct_atom == mol_atom

        if is_correct:
            correct_mol += 1

    atom_accuracy = all_correct_atom / total_atom if total_atom > 0 else 0.0
    mol_accuracy = correct_mol / total_mol if total_mol > 0 else 0.0
    return atom_accuracy, mol_accuracy


def build_model(rep_hparams, num_classes):
    representation_model = TorchMD_ET(
        hidden_channels=rep_hparams["hidden_channels"],
        attn_activation=rep_hparams["attn_activation"],
        num_heads=rep_hparams["num_heads"],
        distance_influence=rep_hparams["distance_influence"],
    )
    output_model = EquivariantScalar(rep_hparams["hidden_channels"], num_classes=num_classes)
    return TorchMD_Net(representation_model=representation_model, output_model=output_model)


def build_lr_lambda(learning_rate, min_lr, warmup_epochs, epochs):
    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if epochs <= warmup_epochs:
            return 1.0

        progress = (epoch - warmup_epochs) / float(epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_scale = (min_lr / learning_rate) if learning_rate > 0 else 0.0
        return min_scale + (1.0 - min_scale) * cosine

    return lr_lambda


def write_log_header(log_f, run_ts, test_years, train_size, test_size):
    log_f.write("==== New Run ====\n")
    log_f.write(f"Run timestamp: {run_ts}\n")
    log_f.write(f"Device: {DEVICE}\n\n")
    log_f.write(f"Test years: {list(test_years)}\n")
    log_f.write("Train years: all other years\n")
    log_f.write(f"Train Data: {train_size} | Test Data: {test_size}\n\n")


def main():
    seed = 150
    deterministic_cudnn = False
    set_seed(seed, deterministic_cudnn=deterministic_cudnn)

    pt_dir = "/inspire/ssd/project/project-public/zhengkaipeng-240108120123/all_materials/data/all_anno_density"
    txt_path = "sorted_by_journal_year.txt"
    test_years = (2020, 2021, 2022, 2023, 2024)

    run_ts = get_run_timestamp()
    metric_log_path = f"train_metric_log_{run_ts}.txt"
    model_save_name = f"torchmd-net-{run_ts}.pth"
    load_model_path = None

    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 1e-2
    epochs = 100
    warmup_epochs = 0
    min_lr = 0.0
    validation_interval = 500
    batch_size_train = 16
    batch_size_test = 1
    num_classes = 98

    rep_hparams = {
        "hidden_channels": 512,
        "attn_activation": "silu",
        "num_heads": 8,
        "distance_influence": "both",
    }

    train_files, test_files, missing = split_by_year_txt(
        txt_path=txt_path,
        pt_dir=pt_dir,
        test_years=test_years,
        pt_prefix="equiv_",
        pt_suffix=".pt",
        strict=False,
    )

    print(f"Train Data: {len(train_files)}")
    print(f"Test  Data: {len(test_files)}")

    if missing:
        print(f"[WARN] missing pt files: {len(missing)}")
        for path in missing[:10]:
            print("  ", path)

    train_dataset, _ = build_simple_in_memory_dataset(train_files, is_eval=False)
    test_dataset, init_mol_acc = build_simple_in_memory_dataset(test_files, is_eval=True)

    print(f"Total class: {num_classes}")
    print(f"Initial Mol Test Accuracy: {init_mol_acc * 100:.2f}%")

    hparams = {
        "seed": seed,
        "deterministic_cudnn": deterministic_cudnn,
        "paths": {"pt_dir": pt_dir, "txt_path": txt_path},
        "split": {"test_years": list(test_years), "strict": False},
        "data": {"train_size": len(train_files), "test_size": len(test_files)},
        "training": {
            "num_classes": num_classes,
            "epochs": epochs,
            "validation_interval": validation_interval,
            "batch_size_train": batch_size_train,
            "batch_size_test": batch_size_test,
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
        "checkpoint": {
            "model_save_name": model_save_name,
            "load_model_path": load_model_path,
        },
        "logs": {"metric_log_path": metric_log_path},
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

    model = build_model(rep_hparams, num_classes).to(DEVICE)
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(learning_rate, min_lr, warmup_epochs, epochs),
    )
    max_mol_accuracy = -1.0

    with open(metric_log_path, "w", encoding="utf-8", buffering=1) as log_f:
        write_log_header(log_f, run_ts, test_years, len(train_files), len(test_files))
        write_hparams(log_f, hparams)

        model.train()
        for epoch in range(epochs):
            for iteration, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(DEVICE)

                outputs, _ = model(data.z, data.pos, data.batch)
                logits = outputs[data.mask]
                loss = criterion(logits, data.y)

                if torch.isnan(loss):
                    print("Loss contains NaN!")
                    break

                loss.backward()
                optimizer.step()

                if (iteration + 1) % validation_interval == 0:
                    atom_accuracy, mol_accuracy = eval_two_pass(
                        model=model,
                        test_loader=test_loader,
                        allow_one_mismatch=False,
                    )
                    model.train()

                    if mol_accuracy > max_mol_accuracy:
                        max_mol_accuracy = mol_accuracy
                        torch.save(model.state_dict(), model_save_name)

                    msg1 = (
                        f"Epoch {epoch}, Iteration {iteration + 1}: Loss: {loss.item()}, "
                        f"Atom Test Accuracy: {atom_accuracy * 100:.2f}%, "
                        f"Mol Test Accuracy: {mol_accuracy * 100:.2f}%"
                    )
                    msg2 = f"Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%"
                    print(msg1)
                    print(msg2)
                    log_f.write(msg1 + "\n")
                    log_f.write(msg2 + "\n")

            scheduler.step()

        atom_accuracy, mol_accuracy = eval_two_pass(
            model=model,
            test_loader=test_loader,
            allow_one_mismatch=True,
        )
        msg1 = (
            f"Atom Test Accuracy: {atom_accuracy * 100:.2f}%, "
            f"Mol Test Accuracy: {mol_accuracy * 100:.2f}%"
        )
        msg2 = f"Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%"
        print(msg1)
        print(msg2)
        log_f.write(msg1 + "\n")
        log_f.write(msg2 + "\n")


if __name__ == "__main__":
    main()
