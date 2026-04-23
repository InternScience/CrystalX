from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from crystalx_train.common import (
    EvalMetrics,
    RepresentationConfig,
    SplitSpec,
    build_model,
    deduplicate_positions,
    get_run_timestamp,
    is_distance_valid,
    load_dataset_pt,
    preview_missing_files,
    resolve_device,
    set_seed,
    split_by_year_txt,
    symbols_to_atomic_numbers,
    to_serializable,
    write_hparams,
    write_log_header,
)


DEFAULT_PT_DIR = "/inspire/ssd/project/project-public/zhengkaipeng-240108120123/all_materials/data/all_anno_density"
DEFAULT_TXT_PATH = "sorted_by_journal_year.txt"
DEFAULT_TEST_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)


@dataclass(frozen=True)
class HeavyTrainingConfig:
    seed: int = 150
    deterministic_cudnn: bool = False
    device: str = "auto"
    pt_dir: str = DEFAULT_PT_DIR
    txt_path: str = DEFAULT_TXT_PATH
    test_years: tuple[int, ...] = DEFAULT_TEST_YEARS
    pt_prefix: str = "equiv_"
    pt_suffix: str = ".pt"
    strict: bool = False
    split_mode: str = "year"
    random_train_ratio: float = 0.8
    random_test_ratio: float = 0.2
    split_seed: int = 150
    check_dist: bool = True
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 1e-2
    epochs: int = 100
    warmup_epochs: int = 0
    min_lr: float = 0.0
    validation_interval: int = 500
    batch_size_train: int = 16
    batch_size_test: int = 1
    num_classes: int = 98
    hidden_channels: int = 512
    attn_activation: str = "silu"
    num_heads: int = 8
    distance_influence: str = "both"
    metric_log_path: str | None = None
    model_save_name: str | None = None
    load_model_path: str | None = None


def build_heavy_dataset(
    file_list: list[str],
    *,
    is_eval: bool,
    check_dist: bool,
) -> tuple[list[Data], float, dict[str, int]]:
    from torch_geometric.data import Data

    dataset: list[Data] = []
    correct_mol = 0
    total_mol = 0
    stats = {
        "kept_samples": 0,
        "dataset_records": 0,
        "dist_drop": 0,
        "noise_drop": 0,
        "element_drop": 0,
        "missing_key_drop": 0,
    }
    required_keys = {"z", "gt", "pos"}

    desc = "Build heavy eval dataset" if is_eval else "Build heavy train dataset"
    for fname in tqdm(file_list, desc=desc):
        mol_info = load_dataset_pt(fname)
        if not required_keys.issubset(mol_info.keys()):
            stats["missing_key_drop"] += 1
            continue

        if "noise_list" in mol_info:
            noise = np.asarray(mol_info["noise_list"])
            if noise.size > 0 and float(np.max(np.abs(noise))) > 0.1:
                stats["noise_drop"] += 1
                continue

        atom_symbols = [item.capitalize() for item in mol_info["z"]]
        label_symbols = [item.capitalize() for item in mol_info["gt"]]
        try:
            atom_numbers = symbols_to_atomic_numbers(atom_symbols)
            label_numbers = symbols_to_atomic_numbers(label_symbols)
        except Exception:
            stats["element_drop"] += 1
            continue

        if not label_numbers:
            stats["missing_key_drop"] += 1
            continue

        unique_atoms, real_cart = deduplicate_positions(atom_numbers, mol_info["pos"])
        mask = torch.zeros(len(unique_atoms), dtype=torch.bool)
        mask[: len(label_numbers)] = True

        if check_dist and not is_distance_valid(real_cart):
            stats["dist_drop"] += 1
            continue

        label = torch.tensor(label_numbers, dtype=torch.long)
        candidate_inputs = [sorted(label_numbers, reverse=True)]
        if not is_eval:
            candidate_inputs.append(unique_atoms[: len(label_numbers)])

        current_atoms = torch.tensor(unique_atoms, dtype=torch.long)
        if bool(torch.equal(current_atoms[mask], label)):
            correct_mol += 1
        total_mol += 1

        real_cart_tensor = torch.from_numpy(real_cart)
        trailing_atoms = unique_atoms[len(label_numbers) :]
        for atoms in candidate_inputs:
            dataset.append(
                Data(
                    z=torch.tensor(list(atoms) + trailing_atoms, dtype=torch.long),
                    y=label,
                    pos=real_cart_tensor,
                    fname=str(fname),
                    mask=mask,
                )
            )

        stats["kept_samples"] += 1
        stats["dataset_records"] = len(dataset)

    mol_accuracy = correct_mol / total_mol if total_mol > 0 else 0.0
    return dataset, mol_accuracy, stats


@torch.no_grad()
def enforce_coverage_by_prob(prob: torch.Tensor, sfac: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
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
    used_atoms: set[int] = set()
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
def evaluate_heavy_two_pass(
    model: torch.nn.Module,
    test_loader,
    *,
    device: torch.device,
    allow_one_mismatch: bool,
) -> EvalMetrics:
    model.eval()
    all_correct_atom = 0
    correct_mol = 0
    total_atom = 0
    total_mol = 0

    for data in tqdm(test_loader, desc="Eval heavy"):
        data = data.to(device)

        first_pass_logits, _ = model(data.z, data.pos, data.batch)
        sfac = torch.unique(data.y)
        first_pass_pred = sfac[first_pass_logits[:, sfac].argmax(dim=1)]

        second_pass_logits, _ = model(first_pass_pred, data.pos, data.batch)
        masked_logits = second_pass_logits[data.mask][:, sfac]
        masked_prob = F.softmax(masked_logits, dim=-1)
        predicted = sfac[masked_prob.argmax(dim=1)]
        predicted = enforce_coverage_by_prob(masked_prob, sfac, predicted)

        label = data.y
        correct_atom = int((predicted == label).sum().item())
        mol_atom = int(label.shape[0])

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
    return EvalMetrics(
        atom_accuracy=atom_accuracy,
        mol_accuracy=mol_accuracy,
        total_atoms=total_atom,
        total_mols=total_mol,
    )


def build_lr_lambda(learning_rate: float, min_lr: float, warmup_epochs: int, epochs: int):
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if epochs <= warmup_epochs:
            return 1.0

        progress = (epoch - warmup_epochs) / float(epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_scale = (min_lr / learning_rate) if learning_rate > 0 else 0.0
        return min_scale + (1.0 - min_scale) * cosine

    return lr_lambda


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CrystalX heavy-atom classifier.")

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--seed", type=int, default=150)
    runtime_group.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    runtime_group.add_argument("--deterministic_cudnn", action="store_true")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--pt_dir", type=str, default=DEFAULT_PT_DIR)
    data_group.add_argument("--txt_path", type=str, default=DEFAULT_TXT_PATH)
    data_group.add_argument("--test_years", type=int, nargs="+", default=list(DEFAULT_TEST_YEARS))
    data_group.add_argument("--pt_prefix", type=str, default="equiv_")
    data_group.add_argument("--pt_suffix", type=str, default=".pt")
    data_group.add_argument("--strict", action="store_true")
    data_group.add_argument("--split_mode", type=str, choices=["year", "random"], default="year")
    data_group.add_argument("--random_train_ratio", type=float, default=0.8)
    data_group.add_argument("--random_test_ratio", type=float, default=0.2)
    data_group.add_argument("--split_seed", type=int, default=150)
    data_group.add_argument("--no_dist_check", action="store_true")

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--learning_rate", type=float, default=1e-4)
    optimization_group.add_argument("--beta1", type=float, default=0.9)
    optimization_group.add_argument("--beta2", type=float, default=0.999)
    optimization_group.add_argument("--epsilon", type=float, default=1e-8)
    optimization_group.add_argument("--weight_decay", type=float, default=1e-2)
    optimization_group.add_argument("--epochs", type=int, default=100)
    optimization_group.add_argument("--warmup_epochs", type=int, default=0)
    optimization_group.add_argument("--min_lr", type=float, default=0.0)
    optimization_group.add_argument("--validation_interval", type=int, default=500)
    optimization_group.add_argument("--batch_size_train", type=int, default=16)
    optimization_group.add_argument("--batch_size_test", type=int, default=1)

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--num_classes", type=int, default=98)
    model_group.add_argument("--hidden_channels", type=int, default=512)
    model_group.add_argument("--attn_activation", type=str, default="silu")
    model_group.add_argument("--num_heads", type=int, default=8)
    model_group.add_argument("--distance_influence", type=str, default="both")

    io_group = parser.add_argument_group("outputs")
    io_group.add_argument("--metric_log_path", type=str, default="")
    io_group.add_argument("--model_save_name", type=str, default="")
    io_group.add_argument("--load_model_path", type=str, default="")

    return parser


def config_from_args(args: argparse.Namespace) -> HeavyTrainingConfig:
    return HeavyTrainingConfig(
        seed=args.seed,
        deterministic_cudnn=args.deterministic_cudnn,
        device=args.device,
        pt_dir=args.pt_dir,
        txt_path=args.txt_path,
        test_years=tuple(args.test_years),
        pt_prefix=args.pt_prefix,
        pt_suffix=args.pt_suffix,
        strict=args.strict,
        split_mode=args.split_mode,
        random_train_ratio=args.random_train_ratio,
        random_test_ratio=args.random_test_ratio,
        split_seed=args.split_seed,
        check_dist=not args.no_dist_check,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        validation_interval=args.validation_interval,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        num_classes=args.num_classes,
        hidden_channels=args.hidden_channels,
        attn_activation=args.attn_activation,
        num_heads=args.num_heads,
        distance_influence=args.distance_influence,
        metric_log_path=args.metric_log_path or None,
        model_save_name=args.model_save_name or None,
        load_model_path=args.load_model_path or None,
    )


def build_run_hparams(
    *,
    config: HeavyTrainingConfig,
    device: torch.device,
    rep_config: RepresentationConfig,
    train_files: list[str],
    test_files: list[str],
    train_stats: dict[str, int],
    test_stats: dict[str, int],
    init_mol_acc: float,
    metric_log_path: str,
    model_save_name: str,
    run_ts: str,
) -> dict[str, Any]:
    return {
        "run_timestamp": run_ts,
        "config": to_serializable(config),
        "device": str(device),
        "data": {
            "train_files": len(train_files),
            "test_files": len(test_files),
            "train_dataset_stats": train_stats,
            "test_dataset_stats": test_stats,
            "initial_test_mol_accuracy": init_mol_acc,
        },
        "model": {
            "representation_model": "TorchMD_ET",
            "representation_hparams": to_serializable(rep_config),
            "output_model": "EquivariantScalar",
            "output_num_classes": config.num_classes,
            "wrapper": "TorchMD_Net",
        },
        "checkpoint": {
            "model_save_name": model_save_name,
            "load_model_path": config.load_model_path,
        },
        "logs": {"metric_log_path": metric_log_path},
    }


def run_training(config: HeavyTrainingConfig) -> None:
    from torch_geometric.loader import DataLoader

    device = resolve_device(config.device)
    print("Device:", device)
    set_seed(config.seed, deterministic_cudnn=config.deterministic_cudnn)

    split_spec = SplitSpec(
        txt_path=config.txt_path,
        pt_dir=config.pt_dir,
        test_years=config.test_years,
        pt_prefix=config.pt_prefix,
        pt_suffix=config.pt_suffix,
        strict=config.strict,
        split_mode=config.split_mode,
        random_train_ratio=config.random_train_ratio,
        random_test_ratio=config.random_test_ratio,
        split_seed=config.split_seed,
    )
    train_files, test_files, missing = split_by_year_txt(split_spec)
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    preview_missing_files(missing)

    train_dataset, _, train_stats = build_heavy_dataset(
        train_files,
        is_eval=False,
        check_dist=config.check_dist,
    )
    test_dataset, init_mol_acc, test_stats = build_heavy_dataset(
        test_files,
        is_eval=True,
        check_dist=config.check_dist,
    )

    if not train_dataset:
        raise ValueError("Heavy training dataset is empty after preprocessing.")
    if not test_dataset:
        raise ValueError("Heavy test dataset is empty after preprocessing.")

    rep_config = RepresentationConfig(
        hidden_channels=config.hidden_channels,
        attn_activation=config.attn_activation,
        num_heads=config.num_heads,
        distance_influence=config.distance_influence,
    )

    run_ts = get_run_timestamp()
    metric_log_path = config.metric_log_path or f"train_metric_log_{run_ts}.txt"
    model_save_name = config.model_save_name or f"torchmd-net-{run_ts}.pth"

    print(f"Initial heavy mol test accuracy: {init_mol_acc * 100:.2f}%")
    print(f"Heavy num_classes: {config.num_classes}")

    hparams = build_run_hparams(
        config=config,
        device=device,
        rep_config=rep_config,
        train_files=train_files,
        test_files=test_files,
        train_stats=train_stats,
        test_stats=test_stats,
        init_mol_acc=init_mol_acc,
        metric_log_path=metric_log_path,
        model_save_name=model_save_name,
        run_ts=run_ts,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_train,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size_test,
        shuffle=False,
    )

    model = build_model(rep_config, config.num_classes).to(device)
    if config.load_model_path:
        model.load_state_dict(torch.load(config.load_model_path, map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=build_lr_lambda(
            config.learning_rate,
            config.min_lr,
            config.warmup_epochs,
            config.epochs,
        ),
    )
    best_mol_accuracy = float("-inf")
    global_step = 0

    with open(metric_log_path, "w", encoding="utf-8", buffering=1) as log_f:
        write_log_header(
            log_f,
            run_ts=run_ts,
            device=device,
            test_years=config.test_years,
            train_size=len(train_files),
            test_size=len(test_files),
        )
        write_hparams(log_f, hparams)

        model.train()
        for epoch in range(config.epochs):
            for step_idx, data in enumerate(train_loader, 1):
                global_step += 1
                optimizer.zero_grad()
                data = data.to(device)

                outputs, _ = model(data.z, data.pos, data.batch)
                logits = outputs[data.mask]
                loss = criterion(logits, data.y)

                if torch.isnan(loss):
                    raise RuntimeError(f"Loss became NaN at epoch={epoch} step={global_step}.")

                loss.backward()
                optimizer.step()

                should_validate = (
                    config.validation_interval > 0
                    and global_step % config.validation_interval == 0
                ) or step_idx == len(train_loader)
                if not should_validate:
                    continue

                metrics = evaluate_heavy_two_pass(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    allow_one_mismatch=False,
                )
                model.train()

                if metrics.mol_accuracy > best_mol_accuracy:
                    best_mol_accuracy = metrics.mol_accuracy
                    torch.save(model.state_dict(), model_save_name)

                msg1 = (
                    f"Epoch {epoch + 1}/{config.epochs}, Step {global_step}: "
                    f"loss={loss.item():.6f}, "
                    f"atom_acc={metrics.atom_accuracy * 100:.2f}%, "
                    f"mol_acc={metrics.mol_accuracy * 100:.2f}%"
                )
                msg2 = f"Best mol_acc: {best_mol_accuracy * 100:.2f}%"
                print(msg1)
                print(msg2)
                log_f.write(msg1 + "\n")
                log_f.write(msg2 + "\n")

            scheduler.step()

        final_metrics = evaluate_heavy_two_pass(
            model=model,
            test_loader=test_loader,
            device=device,
            allow_one_mismatch=True,
        )
        final_msg = (
            f"[Final] atom_acc={final_metrics.atom_accuracy * 100:.2f}%, "
            f"mol_acc={final_metrics.mol_accuracy * 100:.2f}% "
            "(allow_one_mismatch=True)"
        )
        best_msg = f"Best mol_acc: {best_mol_accuracy * 100:.2f}%"
        print(final_msg)
        print(best_msg)
        log_f.write(final_msg + "\n")
        log_f.write(best_msg + "\n")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    run_training(config)


if __name__ == "__main__":
    main()
