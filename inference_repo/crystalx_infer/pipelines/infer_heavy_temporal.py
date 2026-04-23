"""Heavy-atom evaluation entrypoint for CrystalX inference."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch

from crystalx_infer.common.checkpoints import HEAVY_CHECKPOINT, resolve_checkpoint_path
from crystalx_infer.common.chem import atomic_symbol_from_z
from crystalx_infer.common.datasets import build_heavy_eval_dataset, split_by_year_txt
from crystalx_infer.common.heavy import run_two_pass_heavy_prediction
from crystalx_infer.common.modeling import DEFAULT_CHECKPOINT_HIDDEN_CHANNELS, load_torchmd_model
from crystalx_infer.common.paths import locate_source_case_files, stem_from_pt_path
from crystalx_infer.common.runtime import get_run_timestamp, resolve_device, set_seed
from crystalx_infer.common.shelx import copy_file, update_shelxt


@dataclass
class HeavyEvalConfig:
    pt_dir: str
    txt_path: str
    model_path: str = HEAVY_CHECKPOINT.filename
    hf_repo_id: str = ""
    test_years: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
    split_mode: str = "year"
    random_train_ratio: float = 0.8
    random_test_ratio: float = 0.2
    split_seed: int = 150
    batch_size: int = 1
    seed: int = 150
    device: str = "auto"
    num_classes: int = 98
    hidden_channels: int = DEFAULT_CHECKPOINT_HIDDEN_CHANNELS
    restrict_to_sfac: bool = False
    allow_one_mismatch: bool = False
    dump_wrong: bool = False
    dump_dir: str = ""
    dump_correct: bool = False
    dump_correct_dir: str = ""
    search_roots: tuple[str, ...] = ("all_test_shelx",)
    cif_root: str = "all_cif"


def _dump_case(
    data,
    predicted,
    cif_root: str,
    out_root_dir: str,
    search_roots: tuple[str, ...],
    tag: str,
):
    from torch_geometric.data import Data

    pred_symbols = [atomic_symbol_from_z(int(item)) for item in predicted.detach().cpu().tolist()]
    fname0 = data.fname[0] if isinstance(data.fname, (list, tuple)) else data.fname
    stem = stem_from_pt_path(fname0)
    located = locate_source_case_files(stem, search_roots)

    out_case_dir = os.path.join(out_root_dir, stem)
    os.makedirs(out_case_dir, exist_ok=True)

    cif_path = os.path.join(cif_root, f"{stem}.cif")
    if os.path.exists(cif_path):
        copy_file(cif_path, os.path.join(out_case_dir, f"{stem}.cif"))
    else:
        print(f"[WARN] cif not found: {cif_path}")

    if located:
        copy_file(located["hkl"], os.path.join(out_case_dir, f"{stem}_{tag}.hkl"))
        copy_file(located["res"], os.path.join(out_case_dir, f"{stem}_a.res"))
        try:
            update_shelxt(
                located["res"],
                os.path.join(out_case_dir, f"{stem}_{tag}.ins"),
                pred_symbols,
            )
        except Exception as exc:
            print("[update_shelxt ERROR]", exc)
            print("[update_shelxt RES]", located["res"])
    else:
        print(f"[WARN] Source SHELX case not found for stem={stem}")

    data_cpu = data.to("cpu")
    torch.save(
        Data(z=pred_symbols, y=data_cpu.y, pos=data_cpu.pos, fname=str(fname0)),
        os.path.join(out_case_dir, f"mol_{stem}.pt"),
    )


@torch.no_grad()
def infer_two_pass(config: HeavyEvalConfig, model, loader, device):
    model.eval()

    all_correct_atom = 0
    total_atom = 0
    correct_mol = 0
    total_mol = 0
    dumped_wrong = 0
    dumped_correct = 0

    if config.dump_wrong:
        os.makedirs(config.dump_dir, exist_ok=True)
    if config.dump_correct:
        os.makedirs(config.dump_correct_dir, exist_ok=True)

    for data in loader:
        data = data.to(device)
        candidate_elements = torch.unique(data.y) if config.restrict_to_sfac else None
        _, predicted, _ = run_two_pass_heavy_prediction(
            model=model,
            input_z=data.z,
            pos=data.pos,
            batch=data.batch,
            mask=data.mask,
            candidate_elements=candidate_elements,
            enforce_element_coverage=config.restrict_to_sfac,
        )

        label = data.y
        correct_atom = int((predicted == label).sum().item())
        mol_atom = int(label.shape[0])
        all_correct_atom += correct_atom
        total_atom += mol_atom

        is_correct = (correct_atom == mol_atom) or (
            config.allow_one_mismatch and correct_atom == mol_atom - 1
        )
        if is_correct:
            correct_mol += 1
            if config.dump_correct:
                _dump_case(
                    data=data,
                    predicted=predicted,
                    cif_root=config.cif_root,
                    out_root_dir=config.dump_correct_dir,
                    search_roots=config.search_roots,
                    tag="AI",
                )
                dumped_correct += 1
        elif config.dump_wrong:
            _dump_case(
                data=data,
                predicted=predicted,
                cif_root=config.cif_root,
                out_root_dir=config.dump_dir,
                search_roots=config.search_roots,
                tag="AI",
            )
            dumped_wrong += 1

        total_mol += 1

    atom_acc = all_correct_atom / max(total_atom, 1)
    mol_acc = correct_mol / max(total_mol, 1)
    return atom_acc, mol_acc, dumped_wrong, dumped_correct


def run_evaluation(config: HeavyEvalConfig) -> None:
    from torch_geometric.loader import DataLoader

    set_seed(config.seed, deterministic_cudnn=False)
    device = resolve_device(config.device)
    print("Device:", device)

    if config.batch_size != 1 and (config.dump_wrong or config.dump_correct):
        raise ValueError("batch_size must be 1 when dump_wrong or dump_correct is enabled.")

    _, test_files, missing = split_by_year_txt(
        txt_path=config.txt_path,
        pt_dir=config.pt_dir,
        test_years=config.test_years,
        pt_prefix="equiv_",
        pt_suffix=".pt",
        strict=False,
        split_mode=config.split_mode,
        random_train_ratio=config.random_train_ratio,
        random_test_ratio=config.random_test_ratio,
        split_seed=config.split_seed,
    )
    print(f"Test files: {len(test_files)} | Missing mapped pt: {len(missing)}")

    test_dataset, build_stats = build_heavy_eval_dataset(
        test_files,
        is_eval=True,
        is_check_dist=True,
    )
    print(f"[Build] {build_stats}")

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    model_path = resolve_checkpoint_path(
        requested_path=config.model_path,
        kind="heavy",
        repo_id=config.hf_repo_id,
    )
    model, resolved_num_classes = load_torchmd_model(
        model_path,
        device=device,
        num_classes=config.num_classes,
        hidden_channels=config.hidden_channels,
    )
    print(f"num_classes: {resolved_num_classes} | model_path={model_path}")

    atom_acc, mol_acc, dumped_wrong, dumped_correct = infer_two_pass(
        config=config,
        model=model,
        loader=test_loader,
        device=device,
    )

    print(f"[{model_path}] Atom Acc: {atom_acc*100:.2f}% | Mol Acc: {mol_acc*100:.2f}%")
    if config.dump_wrong:
        print(f"[Dump] wrong dumped: {dumped_wrong} -> {config.dump_dir}")
    if config.dump_correct:
        print(f"[Dump] correct dumped: {dumped_correct} -> {config.dump_correct_dir}")


def build_parser() -> argparse.ArgumentParser:
    timestamp = get_run_timestamp()
    parser = argparse.ArgumentParser(description="CrystalX heavy-atom evaluator")
    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=HeavyEvalConfig.model_path)
    parser.add_argument("--hf_repo_id", type=str, default=HeavyEvalConfig.hf_repo_id)
    parser.add_argument(
        "--test_years",
        type=int,
        nargs="+",
        default=list(HeavyEvalConfig.test_years),
    )
    parser.add_argument("--split_mode", type=str, choices=["year", "random"], default=HeavyEvalConfig.split_mode)
    parser.add_argument("--random_train_ratio", type=float, default=HeavyEvalConfig.random_train_ratio)
    parser.add_argument("--random_test_ratio", type=float, default=HeavyEvalConfig.random_test_ratio)
    parser.add_argument("--split_seed", type=int, default=HeavyEvalConfig.split_seed)
    parser.add_argument("--batch_size", type=int, default=HeavyEvalConfig.batch_size)
    parser.add_argument("--seed", type=int, default=HeavyEvalConfig.seed)
    parser.add_argument("--device", type=str, default=HeavyEvalConfig.device, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num_classes", type=int, default=HeavyEvalConfig.num_classes)
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=HeavyEvalConfig.hidden_channels,
        help="Checkpoint hidden width. Use 0 to infer automatically from the checkpoint.",
    )
    parser.add_argument("--restrict_to_sfac", action="store_true")
    parser.add_argument("--allow_one_mismatch", action="store_true")
    parser.add_argument("--dump_wrong", action="store_true")
    parser.add_argument("--dump_dir", type=str, default=f"infer_bad_{timestamp}")
    parser.add_argument("--dump_correct", action="store_true")
    parser.add_argument("--dump_correct_dir", type=str, default=f"infer_good_{timestamp}")
    parser.add_argument("--search_roots", type=str, nargs="*", default=list(HeavyEvalConfig.search_roots))
    parser.add_argument("--cif_root", type=str, default=HeavyEvalConfig.cif_root)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = HeavyEvalConfig(
        pt_dir=args.pt_dir,
        txt_path=args.txt_path,
        model_path=args.model_path,
        hf_repo_id=args.hf_repo_id,
        test_years=tuple(args.test_years),
        split_mode=args.split_mode,
        random_train_ratio=args.random_train_ratio,
        random_test_ratio=args.random_test_ratio,
        split_seed=args.split_seed,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        num_classes=args.num_classes,
        hidden_channels=args.hidden_channels,
        restrict_to_sfac=args.restrict_to_sfac,
        allow_one_mismatch=args.allow_one_mismatch,
        dump_wrong=args.dump_wrong,
        dump_dir=args.dump_dir,
        dump_correct=args.dump_correct,
        dump_correct_dir=args.dump_correct_dir,
        search_roots=tuple(args.search_roots),
        cif_root=args.cif_root,
    )
    run_evaluation(config)


if __name__ == "__main__":
    main()
