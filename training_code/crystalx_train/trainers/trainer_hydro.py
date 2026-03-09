import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    pt_prefix="",
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


def _filter_extra_atoms(real_cart, atom_numbers, hydro_num, extra_atom_dist_thresh):
    if (
        extra_atom_dist_thresh is None
        or float(extra_atom_dist_thresh) <= 0
        or real_cart.shape[0] <= len(hydro_num)
    ):
        return real_cart, atom_numbers, 0

    main_n = min(len(hydro_num), real_cart.shape[0])
    extra_n = real_cart.shape[0] - main_n
    if main_n <= 0 or extra_n <= 0:
        return real_cart, atom_numbers, 0

    main_cart = real_cart[:main_n]
    extra_cart = real_cart[main_n:]
    dist_extra_to_main = cdist(extra_cart, main_cart)
    keep_extra = np.min(dist_extra_to_main, axis=1) < float(extra_atom_dist_thresh)
    keep_idx = list(range(main_n)) + [
        main_n + idx for idx, keep in enumerate(keep_extra.tolist()) if keep
    ]
    dropped_now = int(real_cart.shape[0] - len(keep_idx))
    if dropped_now <= 0:
        return real_cart, atom_numbers, 0

    filtered_cart = real_cart[keep_idx]
    filtered_atoms = [atom_numbers[idx] for idx in keep_idx]
    return filtered_cart, filtered_atoms, dropped_now


def _is_distance_valid(real_cart):
    distance_matrix = cdist(real_cart, real_cart) + 10 * np.eye(real_cart.shape[0])
    return np.min(distance_matrix) >= 0.1


def build_simple_in_memory_dataset(file_list, extra_atom_dist_thresh=None, is_check_dist=True):
    dataset = []
    sample_cnt = 0
    max_h = -1
    dist_error_cnt = 0
    extra_atom_drop_cnt = 0

    for fname in tqdm(file_list):
        mol_info = torch.load(fname)

        atom_symbols = [item.capitalize() for item in mol_info["equiv_gt"]]
        main_atom_symbols = [item.capitalize() for item in mol_info["gt"]]
        try:
            atom_numbers = [Chem.Atom(item).GetAtomicNum() for item in atom_symbols]
            main_atom_numbers = [Chem.Atom(item).GetAtomicNum() for item in main_atom_symbols]
        except Exception:
            continue

        hydro_num = mol_info["hydro_gt"]
        if isinstance(hydro_num, torch.Tensor):
            hydro_num = hydro_num.detach().cpu().tolist()
        elif isinstance(hydro_num, np.ndarray):
            hydro_num = hydro_num.tolist()
        hydro_num = [int(value) for value in hydro_num]

        max_h = max(max_h, max(hydro_num))

        equiv_atoms, real_cart = _deduplicate_positions(atom_numbers, mol_info["pos"])
        real_cart, equiv_atoms, dropped_now = _filter_extra_atoms(
            real_cart,
            equiv_atoms,
            hydro_num,
            extra_atom_dist_thresh,
        )
        extra_atom_drop_cnt += dropped_now

        mask = torch.zeros(len(equiv_atoms), dtype=torch.bool)
        mask[: len(hydro_num)] = True

        if is_check_dist and not _is_distance_valid(real_cart):
            dist_error_cnt += 1
            continue

        dataset.append(
            Data(
                z=torch.tensor(equiv_atoms),
                main_z=torch.tensor(equiv_atoms),
                y=torch.tensor(hydro_num),
                pos=torch.from_numpy(real_cart),
                main_gt=torch.tensor(main_atom_numbers),
                fname=fname,
                mask=mask,
            )
        )
        sample_cnt += 1

    print(sample_cnt)
    print(dist_error_cnt)
    if extra_atom_dist_thresh is not None and float(extra_atom_dist_thresh) > 0:
        print(
            "Extra atoms dropped by distance threshold "
            f"(<{float(extra_atom_dist_thresh):.3f}A to any main atom): {extra_atom_drop_cnt}"
        )

    return dataset, max_h


def write_hparams(log_f, hparams: dict):
    log_f.write("---- Hyperparameters ----\n")
    log_f.write(json.dumps(hparams, indent=2, ensure_ascii=False) + "\n")
    log_f.write("-------------------------\n\n")


@torch.no_grad()
def eval_validate(model, test_loader):
    model.eval()
    correct_predictions = 0
    correct_mol = 0
    total_atoms = 0
    total_mol = 0

    for data in tqdm(test_loader):
        data = data.to(DEVICE)
        outputs, _ = model(data.z, data.pos, data.batch)
        predicted = outputs[data.mask].argmax(dim=1)

        if not (data.main_gt == data.main_z[data.mask]).all():
            continue

        label = data.y
        correct_atom_num = (predicted == label).sum().item()
        all_atom_num = label.shape[0]

        correct_predictions += correct_atom_num
        total_atoms += all_atom_num
        total_mol += 1

        if correct_atom_num == all_atom_num:
            correct_mol += 1

    atom_accuracy = correct_predictions / total_atoms if total_atoms > 0 else 0.0
    mol_accuracy = correct_mol / total_mol if total_mol > 0 else 0.0
    model.train()
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


def write_log_header(log_f, run_ts, test_years, train_size, test_size):
    log_f.write("==== New Run ====\n")
    log_f.write(f"Run timestamp: {run_ts}\n")
    log_f.write(f"Device: {DEVICE}\n\n")
    log_f.write(f"Test years: {list(test_years)}\n")
    log_f.write("Train years: all other years\n")
    log_f.write(f"Train Data: {train_size} | Test Data: {test_size}\n\n")


def main():
    seed = 183
    deterministic_cudnn = False
    set_seed(seed, deterministic_cudnn=deterministic_cudnn)

    pt_dir = "/inspire/ssd/project/project-public/zhengkaipeng-240108120123/all_materials/data/all_anno_density"
    txt_path = "sorted_by_journal_year.txt"
    test_years = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
    pt_prefix = "equiv_"
    pt_suffix = ".pt"

    run_ts = get_run_timestamp()
    metric_log_path = f"train_hydro_metric_log_{run_ts}.txt"
    model_save_name = f"torchmd-hydro-{run_ts}.pth"
    load_model_path = None

    epochs = 100
    validation_interval = 1000
    batch_size_train = 16
    batch_size_test = 1
    extra_atom_dist_thresh = 3.2

    learning_rate = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    weight_decay = 1e-2

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
        pt_prefix=pt_prefix,
        pt_suffix=pt_suffix,
        strict=False,
    )

    print(f"Train Data: {len(train_files)}")
    print(f"Test  Data: {len(test_files)}")

    if missing:
        print(f"[WARN] missing pt files: {len(missing)}")
        for path in missing[:10]:
            print("  ", path)

    train_dataset, max_h_train = build_simple_in_memory_dataset(
        train_files,
        extra_atom_dist_thresh=extra_atom_dist_thresh,
        is_check_dist=True,
    )
    test_dataset, _ = build_simple_in_memory_dataset(
        test_files,
        extra_atom_dist_thresh=extra_atom_dist_thresh,
        is_check_dist=True,
    )
    num_classes = max_h_train + 1
    print("num_classes:", num_classes)

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
        "checkpoint": {
            "model_save_name": model_save_name,
            "load_model_path": load_model_path,
        },
        "logs": {"metric_log_path": metric_log_path},
        "eval": {"rule": "only count samples where main_gt matches masked main_z"},
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
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
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
                        f"Mol Test Accuracy: {mol_acc * 100:.2f}%"
                    )
                    msg2 = f"Max Mol Test Accuracy: {max_mol_accuracy * 100:.2f}%"
                    print(msg1)
                    print(msg2)
                    log_f.write(msg1 + "\n")
                    log_f.write(msg2 + "\n")

            scheduler.step()

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


if __name__ == "__main__":
    main()
