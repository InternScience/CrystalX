"""Hydrogen-count evaluation entrypoint for CrystalX inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch

from crystalx_infer.common.checkpoints import HYDRO_CHECKPOINT, resolve_checkpoint_path
from crystalx_infer.common.chem import atomic_num_from_symbol, atomic_symbol_from_z
from crystalx_infer.common.datasets import build_hydro_eval_dataset, split_by_year_txt
from crystalx_infer.common.hydrogen import (
    adjust_prediction_by_mol_graph,
    build_graph_from_equiv_ase,
    build_graph_from_equiv_rdkit,
    gen_hfix_ins,
    predict_hydrogen_counts,
)
from crystalx_infer.common.modeling import DEFAULT_CHECKPOINT_HIDDEN_CHANNELS, load_torchmd_model
from crystalx_infer.common.paths import resolve_refined_case_files
from crystalx_infer.common.runtime import resolve_device, set_seed
from crystalx_infer.common.shelx import (
    copy_file,
    extract_non_numeric_prefix,
    get_bond,
    load_shelxt,
    update_shelxt_hydro,
)


@dataclass
class HydroEvalConfig:
    pt_dir: str
    txt_path: str
    model_path: str = HYDRO_CHECKPOINT.filename
    hf_repo_id: str = ""
    test_years: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
    pt_prefix: str = "equiv_"
    pt_suffix: str = ".pt"
    strict: bool = False
    batch_size: int = 1
    seed: int = 183
    num_classes: int = 0
    device: str = "auto"
    hidden_channels: int = DEFAULT_CHECKPOINT_HIDDEN_CHANNELS
    is_filter: bool = False
    no_dist_check: bool = False
    atom_analysis: bool = False
    dump_test_data: bool = False
    refined_dir: str = "all_refined_10"
    disable_mol_graph_adjust: bool = False
    print_alignment_check: bool = False
    print_alignment_max_cases: int = 20
    graph_builder: str = "rdkit"
    ase_cutoff_mult: float = 1.10
    ase_extra_cutoff: float = 0.00
    ase_skin: float = 0.00
    rdkit_cov_factor: float = 1.30


def _check_alignment_with_ins(atom_names, main_z_mask):
    ins_symbols = [extract_non_numeric_prefix(name).capitalize() for name in atom_names]
    ins_z = []
    for symbol in ins_symbols:
        if not symbol:
            ins_z.append(-1)
            continue
        try:
            ins_z.append(atomic_num_from_symbol(symbol))
        except Exception:
            ins_z.append(-1)

    target_z = [int(value) for value in main_z_mask.detach().cpu().tolist()]
    n = min(len(ins_z), len(target_z))
    mismatch_idx = [idx for idx in range(n) if ins_z[idx] != target_z[idx]]
    return {
        "ins_len": len(ins_z),
        "target_len": len(target_z),
        "len_mismatch": len(ins_z) != len(target_z),
        "mismatch_count": len(mismatch_idx),
        "mismatch_idx": mismatch_idx,
        "ins_symbols": ins_symbols,
        "target_z": target_z,
    }


def _build_graph(config: HydroEvalConfig, main_z_mask, pos_mask):
    if config.graph_builder == "rdkit":
        return build_graph_from_equiv_rdkit(
            main_z_mask=main_z_mask,
            pos_mask=pos_mask,
            rdkit_cov_factor=config.rdkit_cov_factor,
        )
    return build_graph_from_equiv_ase(
        main_z_mask=main_z_mask,
        pos_mask=pos_mask,
        ase_cutoff_mult=config.ase_cutoff_mult,
        ase_extra_cutoff=config.ase_extra_cutoff,
        ase_skin=config.ase_skin,
    )


@torch.no_grad()
def eval_validate(config: HydroEvalConfig, model, test_loader, device):
    model.eval()
    missing_cnt = 0
    correct_predictions = 0
    correct_mol = 0
    total_atoms = 0
    total_mol = 0
    all_pred = []
    all_label = []
    dump_fail = 0
    dump_ok = 0
    graph_adjust_atom_changed = 0
    graph_adjust_case_changed = 0
    graph_adjust_case_skipped = 0
    graph_adjust_atoms_checked = 0
    align_checked_cases = 0
    align_mismatch_cases = 0
    align_len_mismatch_cases = 0
    align_printed_cases = 0

    for data in test_loader:
        try:
            data = data.to(device)
            prob, predicted = predict_hydrogen_counts(
                model=model,
                input_z=data.z,
                pos=data.pos,
                batch=data.batch,
                mask=data.mask,
            )
        except Exception as exc:
            print(exc)
            missing_cnt += 1
            continue

        main_correct = bool((data.main_gt == data.main_z[data.mask]).all().item())
        if not main_correct:
            continue

        if not config.disable_mol_graph_adjust:
            try:
                atom_symbols, degree_list = _build_graph(
                    config=config,
                    main_z_mask=data.main_z[data.mask],
                    pos_mask=data.pos[data.mask],
                )
                predicted, changed, checked = adjust_prediction_by_mol_graph(
                    prob=prob,
                    predicted=predicted,
                    atom_symbols=atom_symbols,
                    degree_list=degree_list,
                )
                graph_adjust_atoms_checked += checked
                graph_adjust_atom_changed += changed
                if changed > 0:
                    graph_adjust_case_changed += 1
            except Exception as exc:
                print("[mol-graph adjust error]", exc)
                graph_adjust_case_skipped += 1

        fname0 = data.fname[0] if isinstance(data.fname, (list, tuple)) else data.fname
        case_files = None
        if config.print_alignment_check or config.dump_test_data:
            case_files = resolve_refined_case_files(config.refined_dir, fname0)

        if config.print_alignment_check and case_files is not None:
            try:
                _, atom_names, _, _ = load_shelxt(
                    case_files["ins"],
                    begin_flag="ANIS",
                    is_atom_name=True,
                )
                align_info = _check_alignment_with_ins(
                    atom_names=atom_names,
                    main_z_mask=data.main_z[data.mask],
                )
                align_checked_cases += 1
                has_mismatch = align_info["len_mismatch"] or align_info["mismatch_count"] > 0
                if has_mismatch:
                    align_mismatch_cases += 1
                    if align_info["len_mismatch"]:
                        align_len_mismatch_cases += 1
                    if align_printed_cases < config.print_alignment_max_cases:
                        preview_idx = align_info["mismatch_idx"][:10]
                        preview = []
                        for idx in preview_idx:
                            preview.append(
                                f"{idx}:{align_info['ins_symbols'][idx]}->"
                                f"{atomic_symbol_from_z(align_info['target_z'][idx])}"
                            )
                        print(
                            "[ALIGN-MISMATCH] "
                            f"sample={fname0} "
                            f"ins_len={align_info['ins_len']} "
                            f"target_len={align_info['target_len']} "
                            f"mismatch_count={align_info['mismatch_count']} "
                            f"preview={preview}"
                        )
                        align_printed_cases += 1
                    elif align_printed_cases == config.print_alignment_max_cases:
                        print("[ALIGN-MISMATCH] more mismatches suppressed...")
                        align_printed_cases += 1
            except Exception as exc:
                print("[ALIGN-CHECK error]", exc)

        label = data.y
        correct_atom_num = int((predicted == label).sum().item())
        correct_predictions += correct_atom_num
        total_atoms += int(label.shape[0])
        total_mol += 1
        if correct_atom_num == int(label.shape[0]):
            correct_mol += 1

        if config.dump_test_data:
            if case_files is None:
                dump_fail += 1
            else:
                hkl_file_path = case_files["hkl"]
                ins_file_path = case_files["ins"]
                lst_file_path = case_files["lst"]
                res_file_path = case_files["res"]
                case_dir = case_files["case_dir"]
                case_name = case_files["case_name"]
                if not (hkl_file_path and res_file_path):
                    dump_fail += 1
                else:
                    new_hkl_file_path = f"{case_dir}/{case_name}_AIhydro.hkl"
                    new_res_file_path = f"{case_dir}/{case_name}_AIhydro.ins"
                    copy_file(hkl_file_path, new_hkl_file_path)
                    mol_graph = get_bond(lst_file_path)
                    hfix_ins = gen_hfix_ins(ins_file_path, mol_graph, predicted.to("cpu"))
                    update_shelxt_hydro(res_file_path, new_res_file_path, hfix_ins)
                    dump_ok += 1

        if config.atom_analysis:
            all_pred.append(predicted.cpu().numpy())
            all_label.append(label.cpu().numpy())

    if config.atom_analysis and all_pred:
        from sklearn.metrics import classification_report

        print(classification_report(np.concatenate(all_label), np.concatenate(all_pred)))

    atom_accuracy = correct_predictions / total_atoms if total_atoms > 0 else 0.0
    mol_accuracy = correct_mol / total_mol if total_mol > 0 else 0.0
    adjust_stats = {
        "enabled": not config.disable_mol_graph_adjust,
        "graph_builder": config.graph_builder,
        "atoms_changed": graph_adjust_atom_changed,
        "cases_changed": graph_adjust_case_changed,
        "cases_skipped": graph_adjust_case_skipped,
        "atoms_checked": graph_adjust_atoms_checked,
        "align_checked_cases": align_checked_cases,
        "align_mismatch_cases": align_mismatch_cases,
        "align_len_mismatch_cases": align_len_mismatch_cases,
    }
    return atom_accuracy, mol_accuracy, missing_cnt, dump_ok, dump_fail, adjust_stats


def run_evaluation(config: HydroEvalConfig) -> None:
    from torch_geometric.loader import DataLoader

    set_seed(config.seed, deterministic_cudnn=False)
    device = resolve_device(config.device)
    print("Device:", device)

    _, test_files, missing = split_by_year_txt(
        txt_path=config.txt_path,
        pt_dir=config.pt_dir,
        test_years=config.test_years,
        pt_prefix=config.pt_prefix,
        pt_suffix=config.pt_suffix,
        strict=config.strict,
    )
    print(f"Test files: {len(test_files)} | Missing mapped pt: {len(missing)}")

    test_dataset, build_stats, max_h_test = build_hydro_eval_dataset(
        test_files,
        is_check_dist=not config.no_dist_check,
        is_filter=config.is_filter,
    )
    print(f"[Build] {build_stats} | max_h_test={max_h_test}")

    if config.batch_size != 1 and (
        (not config.disable_mol_graph_adjust) or config.dump_test_data or config.print_alignment_check
    ):
        raise ValueError(
            "batch_size must be 1 when using mol-graph adjustment, dump_test_data, or print_alignment_check."
        )

    model_path = resolve_checkpoint_path(
        requested_path=config.model_path,
        kind="hydro",
        repo_id=config.hf_repo_id,
    )
    model, resolved_num_classes = load_torchmd_model(
        model_path,
        device=device,
        num_classes=config.num_classes,
        hidden_channels=config.hidden_channels,
    )
    print(f"num_classes: {resolved_num_classes} | model_path={model_path}")

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    atom_acc, mol_acc, missing_cnt, dump_ok, dump_fail, adjust_stats = eval_validate(
        config=config,
        model=model,
        test_loader=test_loader,
        device=device,
    )

    print(
        f"[{model_path}] Atom Acc: {atom_acc * 100:.2f}% | "
        f"Mol Acc: {mol_acc * 100:.2f}%"
    )
    print(f"[Infer] forward errors skipped: {missing_cnt}")
    if adjust_stats["enabled"]:
        print(
            "[MolGraphAdjust] "
            f"builder={adjust_stats['graph_builder']} | "
            f"atoms_changed={adjust_stats['atoms_changed']} | "
            f"cases_changed={adjust_stats['cases_changed']} | "
            f"atoms_checked={adjust_stats['atoms_checked']} | "
            f"cases_skipped_no_graph={adjust_stats['cases_skipped']}"
        )
        if config.print_alignment_check:
            print(
                "[AlignmentCheck] "
                f"checked={adjust_stats['align_checked_cases']} | "
                f"mismatch_cases={adjust_stats['align_mismatch_cases']} | "
                f"len_mismatch_cases={adjust_stats['align_len_mismatch_cases']}"
            )
    if config.dump_test_data:
        print(
            f"[Dump] success: {dump_ok} | fail_missing_inputs: {dump_fail} | "
            f"root: {config.refined_dir}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CrystalX hydrogen-count evaluator")
    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=HydroEvalConfig.model_path)
    parser.add_argument("--hf_repo_id", type=str, default=HydroEvalConfig.hf_repo_id)
    parser.add_argument("--test_years", type=int, nargs="+", default=list(HydroEvalConfig.test_years))
    parser.add_argument("--pt_prefix", type=str, default=HydroEvalConfig.pt_prefix)
    parser.add_argument("--pt_suffix", type=str, default=HydroEvalConfig.pt_suffix)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--batch_size", type=int, default=HydroEvalConfig.batch_size)
    parser.add_argument("--seed", type=int, default=HydroEvalConfig.seed)
    parser.add_argument("--num_classes", type=int, default=HydroEvalConfig.num_classes)
    parser.add_argument("--device", type=str, default=HydroEvalConfig.device, choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=HydroEvalConfig.hidden_channels,
        help="Checkpoint hidden width. Use 0 to infer automatically from the checkpoint.",
    )
    parser.add_argument("--is_filter", action="store_true")
    parser.add_argument("--no_dist_check", action="store_true")
    parser.add_argument("--atom_analysis", action="store_true")
    parser.add_argument("--dump_test_data", action="store_true")
    parser.add_argument("--refined_dir", type=str, default=HydroEvalConfig.refined_dir)
    parser.add_argument("--disable_mol_graph_adjust", action="store_true")
    parser.add_argument("--print_alignment_check", action="store_true")
    parser.add_argument("--print_alignment_max_cases", type=int, default=HydroEvalConfig.print_alignment_max_cases)
    parser.add_argument("--graph_builder", type=str, default=HydroEvalConfig.graph_builder, choices=["ase", "rdkit"])
    parser.add_argument("--ase_cutoff_mult", type=float, default=HydroEvalConfig.ase_cutoff_mult)
    parser.add_argument("--ase_extra_cutoff", type=float, default=HydroEvalConfig.ase_extra_cutoff)
    parser.add_argument("--ase_skin", type=float, default=HydroEvalConfig.ase_skin)
    parser.add_argument("--rdkit_cov_factor", type=float, default=HydroEvalConfig.rdkit_cov_factor)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = HydroEvalConfig(
        pt_dir=args.pt_dir,
        txt_path=args.txt_path,
        model_path=args.model_path,
        hf_repo_id=args.hf_repo_id,
        test_years=tuple(args.test_years),
        pt_prefix=args.pt_prefix,
        pt_suffix=args.pt_suffix,
        strict=args.strict,
        batch_size=args.batch_size,
        seed=args.seed,
        num_classes=args.num_classes,
        device=args.device,
        hidden_channels=args.hidden_channels,
        is_filter=args.is_filter,
        no_dist_check=args.no_dist_check,
        atom_analysis=args.atom_analysis,
        dump_test_data=args.dump_test_data,
        refined_dir=args.refined_dir,
        disable_mol_graph_adjust=args.disable_mol_graph_adjust,
        print_alignment_check=args.print_alignment_check,
        print_alignment_max_cases=args.print_alignment_max_cases,
        graph_builder=args.graph_builder,
        ase_cutoff_mult=args.ase_cutoff_mult,
        ase_extra_cutoff=args.ase_extra_cutoff,
        ase_skin=args.ase_skin,
        rdkit_cov_factor=args.rdkit_cov_factor,
    )
    run_evaluation(config)


if __name__ == "__main__":
    main()
