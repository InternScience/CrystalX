from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from crystalx_train.common import (
    EvalMetrics,
    RepresentationConfig,
    SplitSpec,
    build_model,
    deduplicate_positions,
    get_run_timestamp,
    is_distance_valid,
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
class HydroTrainingConfig:
    seed: int = 183
    deterministic_cudnn: bool = False
    device: str = "auto"
    pt_dir: str = DEFAULT_PT_DIR
    txt_path: str = DEFAULT_TXT_PATH
    test_years: tuple[int, ...] = DEFAULT_TEST_YEARS
    pt_prefix: str = "equiv_"
    pt_suffix: str = ".pt"
    strict: bool = False
    check_dist: bool = True
    extra_atom_dist_thresh: float = 3.2
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 1e-2
    epochs: int = 100
    validation_interval: int = 1000
    batch_size_train: int = 16
    batch_size_test: int = 1
    num_classes: int = 0
    hidden_channels: int = 512
    attn_activation: str = "silu"
    num_heads: int = 8
    distance_influence: str = "both"
    metric_log_path: str | None = None
    model_save_name: str | None = None
    load_model_path: str | None = None


def filter_extra_atoms(
    real_cart: np.ndarray,
    atom_numbers: list[int],
    hydro_num: list[int],
    extra_atom_dist_thresh: float | None,
) -> tuple[np.ndarray, list[int], int]:
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


def build_hydro_dataset(
    file_list: list[str],
    *,
    extra_atom_dist_thresh: float | None,
    check_dist: bool,
) -> tuple[list[Data], int, dict[str, int]]:
    from torch_geometric.data import Data

    dataset: list[Data] = []
    max_h = -1
    stats = {
        "kept_samples": 0,
        "dataset_records": 0,
        "dist_drop": 0,
        "element_drop": 0,
        "missing_key_drop": 0,
        "shape_drop": 0,
        "extra_atom_drop_total": 0,
    }
    required_keys = {"equiv_gt", "gt", "hydro_gt", "pos"}

    for fname in tqdm(file_list, desc="Build hydro dataset"):
        mol_info = torch.load(fname)
        if not required_keys.issubset(mol_info.keys()):
            stats["missing_key_drop"] += 1
            continue

        atom_symbols = [item.capitalize() for item in mol_info["equiv_gt"]]
        main_atom_symbols = [item.capitalize() for item in mol_info["gt"]]
        try:
            atom_numbers = symbols_to_atomic_numbers(atom_symbols)
            main_atom_numbers = symbols_to_atomic_numbers(main_atom_symbols)
        except Exception:
            stats["element_drop"] += 1
            continue

        hydro_num = mol_info["hydro_gt"]
        if isinstance(hydro_num, torch.Tensor):
            hydro_num = hydro_num.detach().cpu().tolist()
        elif isinstance(hydro_num, np.ndarray):
            hydro_num = hydro_num.tolist()
        hydro_num = [int(value) for value in hydro_num]
        if not hydro_num:
            stats["shape_drop"] += 1
            continue

        max_h = max(max_h, max(hydro_num))

        equiv_atoms, real_cart = deduplicate_positions(atom_numbers, mol_info["pos"])
        real_cart, equiv_atoms, dropped_now = filter_extra_atoms(
            real_cart,
            equiv_atoms,
            hydro_num,
            extra_atom_dist_thresh,
        )
        stats["extra_atom_drop_total"] += dropped_now

        mask = torch.zeros(len(equiv_atoms), dtype=torch.bool)
        mask[: len(hydro_num)] = True

        if check_dist and not is_distance_valid(real_cart):
            stats["dist_drop"] += 1
            continue

        dataset.append(
            Data(
                z=torch.tensor(equiv_atoms, dtype=torch.long),
                main_z=torch.tensor(equiv_atoms, dtype=torch.long),
                y=torch.tensor(hydro_num, dtype=torch.long),
                pos=torch.from_numpy(real_cart),
                main_gt=torch.tensor(main_atom_numbers, dtype=torch.long),
                fname=str(fname),
                mask=mask,
            )
        )
        stats["kept_samples"] += 1
        stats["dataset_records"] = len(dataset)

    return dataset, max_h, stats


@torch.no_grad()
def evaluate_hydro(
    model: torch.nn.Module,
    test_loader,
    *,
    device: torch.device,
) -> EvalMetrics:
    model.eval()
    correct_predictions = 0
    correct_mol = 0
    total_atoms = 0
    total_mol = 0
    skipped_alignment = 0

    for data in tqdm(test_loader, desc="Eval hydro"):
        data = data.to(device)
        outputs, _ = model(data.z, data.pos, data.batch)
        predicted = outputs[data.mask].argmax(dim=1)

        if not bool((data.main_gt == data.main_z[data.mask]).all().item()):
            skipped_alignment += 1
            continue

        label = data.y
        correct_atom_num = int((predicted == label).sum().item())
        all_atom_num = int(label.shape[0])

        correct_predictions += correct_atom_num
        total_atoms += all_atom_num
        total_mol += 1

        if correct_atom_num == all_atom_num:
            correct_mol += 1

    atom_accuracy = correct_predictions / total_atoms if total_atoms > 0 else 0.0
    mol_accuracy = correct_mol / total_mol if total_mol > 0 else 0.0
    return EvalMetrics(
        atom_accuracy=atom_accuracy,
        mol_accuracy=mol_accuracy,
        total_atoms=total_atoms,
        total_mols=total_mol,
        skipped_mols=skipped_alignment,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CrystalX hydrogen-count classifier.")

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--seed", type=int, default=183)
    runtime_group.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    runtime_group.add_argument("--deterministic_cudnn", action="store_true")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--pt_dir", type=str, default=DEFAULT_PT_DIR)
    data_group.add_argument("--txt_path", type=str, default=DEFAULT_TXT_PATH)
    data_group.add_argument("--test_years", type=int, nargs="+", default=list(DEFAULT_TEST_YEARS))
    data_group.add_argument("--pt_prefix", type=str, default="equiv_")
    data_group.add_argument("--pt_suffix", type=str, default=".pt")
    data_group.add_argument("--strict", action="store_true")
    data_group.add_argument("--no_dist_check", action="store_true")
    data_group.add_argument("--extra_atom_dist_thresh", type=float, default=3.2)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--learning_rate", type=float, default=1e-4)
    optimization_group.add_argument("--beta1", type=float, default=0.9)
    optimization_group.add_argument("--beta2", type=float, default=0.999)
    optimization_group.add_argument("--epsilon", type=float, default=1e-8)
    optimization_group.add_argument("--weight_decay", type=float, default=1e-2)
    optimization_group.add_argument("--epochs", type=int, default=100)
    optimization_group.add_argument("--validation_interval", type=int, default=1000)
    optimization_group.add_argument("--batch_size_train", type=int, default=16)
    optimization_group.add_argument("--batch_size_test", type=int, default=1)

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="Set explicitly; use 0 to infer from training labels.",
    )
    model_group.add_argument("--hidden_channels", type=int, default=512)
    model_group.add_argument("--attn_activation", type=str, default="silu")
    model_group.add_argument("--num_heads", type=int, default=8)
    model_group.add_argument("--distance_influence", type=str, default="both")

    io_group = parser.add_argument_group("outputs")
    io_group.add_argument("--metric_log_path", type=str, default="")
    io_group.add_argument("--model_save_name", type=str, default="")
    io_group.add_argument("--load_model_path", type=str, default="")

    return parser


def config_from_args(args: argparse.Namespace) -> HydroTrainingConfig:
    return HydroTrainingConfig(
        seed=args.seed,
        deterministic_cudnn=args.deterministic_cudnn,
        device=args.device,
        pt_dir=args.pt_dir,
        txt_path=args.txt_path,
        test_years=tuple(args.test_years),
        pt_prefix=args.pt_prefix,
        pt_suffix=args.pt_suffix,
        strict=args.strict,
        check_dist=not args.no_dist_check,
        extra_atom_dist_thresh=args.extra_atom_dist_thresh,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
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
    config: HydroTrainingConfig,
    device: torch.device,
    rep_config: RepresentationConfig,
    train_files: list[str],
    test_files: list[str],
    train_stats: dict[str, int],
    test_stats: dict[str, int],
    num_classes: int,
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
        },
        "training": {
            "task": "predict_hydro_num_per_atom",
            "num_classes": num_classes,
            "eval_rule": "only count samples where main_gt matches masked main_z",
        },
        "model": {
            "representation_model": "TorchMD_ET",
            "representation_hparams": to_serializable(rep_config),
            "output_model": "EquivariantScalar",
            "output_num_classes": num_classes,
            "wrapper": "TorchMD_Net",
        },
        "checkpoint": {
            "model_save_name": model_save_name,
            "load_model_path": config.load_model_path,
        },
        "logs": {"metric_log_path": metric_log_path},
    }


def run_training(config: HydroTrainingConfig) -> None:
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
    )
    train_files, test_files, missing = split_by_year_txt(split_spec)
    print(f"Train files: {len(train_files)}")
    print(f"Test files: {len(test_files)}")
    preview_missing_files(missing)

    train_dataset, max_h_train, train_stats = build_hydro_dataset(
        train_files,
        extra_atom_dist_thresh=config.extra_atom_dist_thresh,
        check_dist=config.check_dist,
    )
    test_dataset, _, test_stats = build_hydro_dataset(
        test_files,
        extra_atom_dist_thresh=config.extra_atom_dist_thresh,
        check_dist=config.check_dist,
    )

    if not train_dataset:
        raise ValueError("Hydrogen training dataset is empty after preprocessing.")
    if not test_dataset:
        raise ValueError("Hydrogen test dataset is empty after preprocessing.")
    if max_h_train < 0 and config.num_classes <= 0:
        raise ValueError("Cannot infer hydro num_classes from an empty or invalid training label set.")

    num_classes = config.num_classes if config.num_classes > 0 else max_h_train + 1
    rep_config = RepresentationConfig(
        hidden_channels=config.hidden_channels,
        attn_activation=config.attn_activation,
        num_heads=config.num_heads,
        distance_influence=config.distance_influence,
    )

    run_ts = get_run_timestamp()
    metric_log_path = config.metric_log_path or f"train_hydro_metric_log_{run_ts}.txt"
    model_save_name = config.model_save_name or f"torchmd-hydro-{run_ts}.pth"

    print(f"Hydro num_classes: {num_classes}")

    hparams = build_run_hparams(
        config=config,
        device=device,
        rep_config=rep_config,
        train_files=train_files,
        test_files=test_files,
        train_stats=train_stats,
        test_stats=test_stats,
        num_classes=num_classes,
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

    model = build_model(rep_config, num_classes).to(device)
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
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
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
                pred = outputs[data.mask]
                loss = criterion(pred, data.y)

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

                metrics = evaluate_hydro(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                )
                model.train()

                if metrics.mol_accuracy > best_mol_accuracy:
                    best_mol_accuracy = metrics.mol_accuracy
                    torch.save(model.state_dict(), model_save_name)

                msg1 = (
                    f"Epoch {epoch + 1}/{config.epochs}, Step {global_step}: "
                    f"loss={loss.item():.6f}, "
                    f"atom_acc={metrics.atom_accuracy * 100:.2f}%, "
                    f"mol_acc={metrics.mol_accuracy * 100:.2f}%, "
                    f"alignment_skipped={metrics.skipped_mols}"
                )
                msg2 = f"Best mol_acc: {best_mol_accuracy * 100:.2f}%"
                print(msg1)
                print(msg2)
                log_f.write(msg1 + "\n")
                log_f.write(msg2 + "\n")

            scheduler.step()

        final_metrics = evaluate_hydro(
            model=model,
            test_loader=test_loader,
            device=device,
        )
        final_msg = (
            f"[Final] atom_acc={final_metrics.atom_accuracy * 100:.2f}%, "
            f"mol_acc={final_metrics.mol_accuracy * 100:.2f}%, "
            f"alignment_skipped={final_metrics.skipped_mols}"
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
