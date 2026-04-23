"""Joint heavy-atom and hydrogen-count evaluation entrypoint."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import torch

from crystalx_infer.common.checkpoints import (
    HEAVY_CHECKPOINT,
    HYDRO_CHECKPOINT,
    resolve_checkpoint_path,
)
from crystalx_infer.common.chem import atomic_symbol_from_z
from crystalx_infer.common.datasets import build_joint_dataset, split_by_year_txt
from crystalx_infer.common.heavy import run_two_pass_heavy_prediction
from crystalx_infer.common.hydrogen import (
    adjust_prediction_by_mol_graph,
    build_graph_from_equiv_ase,
    build_graph_from_equiv_rdkit,
    hydro_group_key,
    hydro_group_name,
    predict_hydrogen_counts,
)
from crystalx_infer.common.modeling import DEFAULT_CHECKPOINT_HIDDEN_CHANNELS, load_torchmd_model
from crystalx_infer.common.paths import load_excluded_stems, stem_from_pt_path
from crystalx_infer.common.runtime import get_run_timestamp, resolve_device, set_seed


@dataclass
class JointEvalConfig:
    pt_dir: str
    txt_path: str
    heavy_model_path: str = HEAVY_CHECKPOINT.filename
    hydro_model_path: str = HYDRO_CHECKPOINT.filename
    hf_repo_id: str = ""
    test_years: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
    pt_prefix: str = "equiv_"
    pt_suffix: str = ".pt"
    strict: bool = False
    split_mode: str = "year"
    random_train_ratio: float = 0.8
    random_test_ratio: float = 0.2
    split_seed: int = 150
    seed: int = 183
    device: str = "auto"
    batch_size: int = 1
    heavy_num_classes: int = 0
    hydro_num_classes: int = 0
    hidden_channels: int = DEFAULT_CHECKPOINT_HIDDEN_CHANNELS
    restrict_to_sfac: bool = False
    allow_one_mismatch: bool = False
    allow_one_correction: bool = False
    allow_k_corrections: int = 0
    hydro_allow_k_corrections: int = 0
    disable_hydro_graph_adjust: bool = False
    graph_builder: str = "rdkit"
    ase_cutoff_mult: float = 1.10
    ase_extra_cutoff: float = 0.00
    ase_skin: float = 0.00
    rdkit_cov_factor: float = 1.30
    heavy_noise_max: float = 0.1
    no_dist_check: bool = False
    dist_min: float = 0.1
    cif_root: str = "all_cif"
    exclude_disorder_txt: str = ""
    heavy_init_cif_out: str = ""
    heavy_cif_out: str = ""
    both_cif_out: str = ""
    heavy_elem_f1_out: str = ""
    hydro_group_f1_out: str = ""


@torch.no_grad()
def apply_k_corrections_until_hit(prob, pred, gt, k, class_labels=None, ensure_coverage=False):
    if prob.ndim != 2 or pred.ndim != 1 or gt.ndim != 1:
        return pred, False, 0
    n, cls = prob.shape
    if n <= 0 or cls < 2 or k <= 0:
        return pred, False, 0

    top2_prob, top2_idx = torch.topk(prob, k=2, dim=1)
    margins = top2_prob[:, 0] - top2_prob[:, 1]
    order = torch.argsort(margins, descending=False)

    base = pred.clone()
    pool_k = min(int(k), int(n))
    candidate_atoms = [int(item.item()) for item in order[:pool_k]]
    tried = 0

    def _apply_flips(atom_indices):
        candidate = base.clone()
        for atom_idx in atom_indices:
            second_idx = int(top2_idx[atom_idx, 1].item())
            if class_labels is None:
                candidate[atom_idx] = int(second_idx)
            else:
                candidate[atom_idx] = class_labels[second_idx]
        return candidate

    def _coverage_ok(candidate):
        if not (ensure_coverage and class_labels is not None):
            return True
        return all(bool((candidate == label).any().item()) for label in class_labels)

    for flip_count in range(1, pool_k + 1):
        for atom_combo in itertools.combinations(candidate_atoms, flip_count):
            candidate = _apply_flips(atom_combo)
            if not _coverage_ok(candidate):
                continue
            tried += 1
            if bool((candidate == gt).all().item()):
                return candidate, True, tried

    return base, False, tried


def _build_graph(config: JointEvalConfig, main_z_mask, pos_mask):
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


def _update_heavy_elem_stats(stats_map, gt_tensor, pred_tensor):
    all_elem = torch.unique(torch.cat([gt_tensor, pred_tensor], dim=0))
    for elem_z_t in all_elem:
        elem_z = int(elem_z_t.item())
        elem_stat = stats_map.setdefault(
            elem_z,
            {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0},
        )
        gt_is_elem = gt_tensor == elem_z
        pred_is_elem = pred_tensor == elem_z
        elem_stat["tp"] += int((gt_is_elem & pred_is_elem).sum().item())
        elem_stat["fp"] += int((~gt_is_elem & pred_is_elem).sum().item())
        elem_stat["fn"] += int((gt_is_elem & ~pred_is_elem).sum().item())
        elem_stat["gt_count"] += int(gt_is_elem.sum().item())
        elem_stat["pred_count"] += int(pred_is_elem.sum().item())


def _update_hydro_group_stats(stats_map, main_z_hydro, degree_list, gt_hydro, pred_hydro):
    atom_n = min(
        int(main_z_hydro.shape[0]),
        int(gt_hydro.shape[0]),
        int(pred_hydro.shape[0]),
    )
    for idx in range(atom_n):
        z_i = int(main_z_hydro[idx].item())
        degree_i = int(degree_list[idx]) if idx < len(degree_list) else 0
        gt_h_i = int(gt_hydro[idx].item())
        pred_h_i = int(pred_hydro[idx].item())
        gt_key = hydro_group_key(z_i, degree_i, gt_h_i)
        pred_key = hydro_group_key(z_i, degree_i, pred_h_i)

        gt_stat = stats_map.setdefault(
            gt_key,
            {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0},
        )
        pred_stat = stats_map.setdefault(
            pred_key,
            {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0},
        )

        gt_stat["gt_count"] += 1
        pred_stat["pred_count"] += 1
        if gt_key == pred_key:
            gt_stat["tp"] += 1
        else:
            gt_stat["fn"] += 1
            pred_stat["fp"] += 1


@torch.no_grad()
def infer_joint(config: JointEvalConfig, heavy_model, hydro_model, loader, device):
    heavy_model.eval()
    hydro_model.eval()

    stats = {
        "total": 0,
        "heavy_init_correct": 0,
        "heavy_init_atom_correct": 0,
        "heavy_init_atom_total": 0,
        "heavy_correct": 0,
        "heavy_atom_correct": 0,
        "heavy_atom_total": 0,
        "hydro_eligible": 0,
        "hydro_correct": 0,
        "hydro_atom_correct": 0,
        "hydro_atom_total": 0,
        "both_correct": 0,
        "heavy_fail_forward": 0,
        "hydro_fail_forward": 0,
        "hydro_adjust_changed_atoms": 0,
        "heavy_one_correction_applied": 0,
        "heavy_k_correction_tried_structures": 0,
        "heavy_k_correction_total_trials": 0,
        "heavy_k_correction_hit": 0,
        "hydro_k_correction_tried_structures": 0,
        "hydro_k_correction_total_trials": 0,
        "hydro_k_correction_hit": 0,
        "heavy_mol_total_by_gt_len": {},
        "heavy_mol_correct_by_gt_len": {},
        "heavy_elem_pr": {},
        "hydro_group_pr": {},
    }
    heavy_init_cif_paths = []
    heavy_cif_paths = []
    both_cif_paths = []

    for data in loader:
        data = data.to(device)
        stats["total"] += 1

        pred_init = data.heavy_raw_z[data.heavy_mask]
        correct_atom_init = int((pred_init == data.heavy_y).sum().item())
        mol_atom = int(data.heavy_y.shape[0])
        stats["heavy_init_atom_correct"] += correct_atom_init
        stats["heavy_init_atom_total"] += mol_atom
        stats["heavy_mol_total_by_gt_len"][mol_atom] = (
            stats["heavy_mol_total_by_gt_len"].get(mol_atom, 0) + 1
        )
        heavy_init_ok = correct_atom_init == mol_atom
        if heavy_init_ok:
            stats["heavy_init_correct"] += 1
            stem = stem_from_pt_path(data.fname[0] if isinstance(data.fname, (list, tuple)) else data.fname)
            cif_path = f"{config.cif_root}/{stem}.cif" if config.cif_root else f"{stem}.cif"
            heavy_init_cif_paths.append(cif_path)

        heavy_ok = False
        try:
            candidate_elements = torch.unique(data.heavy_y) if config.restrict_to_sfac else None
            prob_heavy, pred_heavy, _ = run_two_pass_heavy_prediction(
                model=heavy_model,
                input_z=data.heavy_init_z,
                pos=data.pos,
                batch=data.batch,
                mask=data.heavy_mask,
                candidate_elements=candidate_elements,
                enforce_element_coverage=config.restrict_to_sfac,
            )

            correct_atom_before = int((pred_heavy == data.heavy_y).sum().item())
            correction_budget = int(config.allow_k_corrections)
            if correction_budget <= 0 and config.allow_one_correction:
                correction_budget = 1
            if correction_budget > 0 and correct_atom_before < mol_atom:
                pred_heavy, hit, used_trials = apply_k_corrections_until_hit(
                    prob_heavy,
                    pred_heavy,
                    data.heavy_y,
                    k=correction_budget,
                    class_labels=candidate_elements,
                    ensure_coverage=bool(config.restrict_to_sfac),
                )
                stats["heavy_k_correction_tried_structures"] += 1
                stats["heavy_k_correction_total_trials"] += int(used_trials)
                if hit:
                    stats["heavy_k_correction_hit"] += 1
                    stats["heavy_one_correction_applied"] += 1

            correct_atom = int((pred_heavy == data.heavy_y).sum().item())
            stats["heavy_atom_correct"] += correct_atom
            stats["heavy_atom_total"] += mol_atom
            _update_heavy_elem_stats(stats["heavy_elem_pr"], data.heavy_y, pred_heavy)

            heavy_ok = (correct_atom == mol_atom) or (
                config.allow_one_mismatch and correct_atom == mol_atom - 1
            )
        except Exception as exc:
            print("[heavy infer error]", exc)
            stats["heavy_fail_forward"] += 1

        if heavy_ok:
            stats["heavy_correct"] += 1
            stats["heavy_mol_correct_by_gt_len"][mol_atom] = (
                stats["heavy_mol_correct_by_gt_len"].get(mol_atom, 0) + 1
            )
            stem = stem_from_pt_path(data.fname[0] if isinstance(data.fname, (list, tuple)) else data.fname)
            cif_path = f"{config.cif_root}/{stem}.cif" if config.cif_root else f"{stem}.cif"
            heavy_cif_paths.append(cif_path)

        hydro_ok = False
        try:
            main_correct = bool((data.main_gt_hydro == data.main_z[data.hydro_mask]).all().item())
            if main_correct:
                stats["hydro_eligible"] += 1
                prob_hydro, pred_hydro = predict_hydrogen_counts(
                    model=hydro_model,
                    input_z=data.hydro_input_z,
                    pos=data.pos,
                    batch=data.batch,
                    mask=data.hydro_mask,
                )

                if not config.disable_hydro_graph_adjust:
                    atom_symbols, degree_list = _build_graph(
                        config=config,
                        main_z_mask=data.main_z[data.hydro_mask],
                        pos_mask=data.pos[data.hydro_mask],
                    )
                    pred_hydro, changed, _ = adjust_prediction_by_mol_graph(
                        prob=prob_hydro,
                        predicted=pred_hydro,
                        atom_symbols=atom_symbols,
                        degree_list=degree_list,
                    )
                    stats["hydro_adjust_changed_atoms"] += int(changed)

                hydro_atom_total = int(data.hydro_y.shape[0])
                hydro_correct_atom_before = int((pred_hydro == data.hydro_y).sum().item())
                if config.hydro_allow_k_corrections > 0 and hydro_correct_atom_before < hydro_atom_total:
                    pred_hydro, hit, used_trials = apply_k_corrections_until_hit(
                        prob_hydro,
                        pred_hydro,
                        data.hydro_y,
                        k=int(config.hydro_allow_k_corrections),
                        class_labels=None,
                        ensure_coverage=False,
                    )
                    stats["hydro_k_correction_tried_structures"] += 1
                    stats["hydro_k_correction_total_trials"] += int(used_trials)
                    if hit:
                        stats["hydro_k_correction_hit"] += 1

                hydro_correct_atom = int((pred_hydro == data.hydro_y).sum().item())
                stats["hydro_atom_correct"] += hydro_correct_atom
                stats["hydro_atom_total"] += hydro_atom_total

                try:
                    _, rdkit_degree = build_graph_from_equiv_rdkit(
                        main_z_mask=data.main_z[data.hydro_mask],
                        pos_mask=data.pos[data.hydro_mask],
                        rdkit_cov_factor=config.rdkit_cov_factor,
                    )
                except Exception:
                    rdkit_degree = [0] * int(data.main_z[data.hydro_mask].shape[0])

                _update_hydro_group_stats(
                    stats["hydro_group_pr"],
                    data.main_z[data.hydro_mask],
                    rdkit_degree,
                    data.hydro_y,
                    pred_hydro,
                )
                hydro_ok = bool((pred_hydro == data.hydro_y).all().item())
        except Exception as exc:
            print("[hydro infer error]", exc)
            stats["hydro_fail_forward"] += 1

        if hydro_ok:
            stats["hydro_correct"] += 1

        if heavy_ok and hydro_ok:
            stats["both_correct"] += 1
            stem = stem_from_pt_path(data.fname[0] if isinstance(data.fname, (list, tuple)) else data.fname)
            cif_path = f"{config.cif_root}/{stem}.cif" if config.cif_root else f"{stem}.cif"
            both_cif_paths.append(cif_path)

    return stats, both_cif_paths, heavy_init_cif_paths, heavy_cif_paths


def _write_lines(path: str, lines) -> None:
    with open(path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def _build_heavy_elem_lines(stats) -> list[str]:
    lines = [
        "atomic_num\telement\tf1(%)\tprecision(%)\trecall(%)\tgt_count\tpred_count\ttp\tfp\tfn\tsupport"
    ]
    for elem_z in sorted(stats["heavy_elem_pr"].keys()):
        elem_stat = stats["heavy_elem_pr"][elem_z]
        tp = int(elem_stat["tp"])
        fp = int(elem_stat["fp"])
        fn = int(elem_stat["fn"])
        gt_count = int(elem_stat["gt_count"])
        pred_count = int(elem_stat["pred_count"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2.0 * tp) / max((2 * tp + fp + fn), 1)
        element_name = atomic_symbol_from_z(elem_z)
        print(
            f"[ElemF1] {element_name}(Z={elem_z}) "
            f"f1={f1*100:.2f}% precision={precision*100:.2f}% recall={recall*100:.2f}% "
            f"(gt={gt_count} pred={pred_count} tp={tp} fp={fp} fn={fn})"
        )
        lines.append(
            f"{elem_z}\t{element_name}\t{f1*100:.4f}\t{precision*100:.4f}\t{recall*100:.4f}\t"
            f"{gt_count}\t{pred_count}\t{tp}\t{fp}\t{fn}\t{gt_count}"
        )
    return lines


def _build_hydro_group_lines(stats) -> list[str]:
    lines = [
        "atomic_num\telement\tdegree\thydro_count\tgroup\tf1(%)\tprecision(%)\trecall(%)\tgt_count\tpred_count\ttp\tfp\tfn\tsupport"
    ]
    for key in sorted(stats["hydro_group_pr"].keys(), key=lambda item: (int(item[0]), int(item[1]), int(item[2]))):
        z_i, degree_i, hydro_count = key
        group_stat = stats["hydro_group_pr"][key]
        tp = int(group_stat["tp"])
        fp = int(group_stat["fp"])
        fn = int(group_stat["fn"])
        gt_count = int(group_stat["gt_count"])
        pred_count = int(group_stat["pred_count"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2.0 * tp) / max((2 * tp + fp + fn), 1)
        element_name = "ALL" if (int(z_i) == 0 and int(hydro_count) == 0) else atomic_symbol_from_z(z_i)
        group_name = hydro_group_name(z_i, degree_i, hydro_count)
        print(
            f"[HydroGroupF1] {group_name}(Z={z_i},deg={degree_i}) "
            f"f1={f1*100:.2f}% precision={precision*100:.2f}% recall={recall*100:.2f}% "
            f"(gt={gt_count} pred={pred_count} tp={tp} fp={fp} fn={fn})"
        )
        lines.append(
            f"{z_i}\t{element_name}\t{degree_i}\t{hydro_count}\t{group_name}\t"
            f"{f1*100:.4f}\t{precision*100:.4f}\t{recall*100:.4f}\t"
            f"{gt_count}\t{pred_count}\t{tp}\t{fp}\t{fn}\t{gt_count}"
        )
    return lines


def run_evaluation(config: JointEvalConfig) -> None:
    from torch_geometric.loader import DataLoader

    if config.batch_size != 1:
        raise ValueError("batch_size must be 1 for joint per-structure evaluation.")

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
        split_mode=config.split_mode,
        random_train_ratio=config.random_train_ratio,
        random_test_ratio=config.random_test_ratio,
        split_seed=config.split_seed,
    )
    excluded_cnt = 0
    if config.exclude_disorder_txt:
        excluded_stems = load_excluded_stems(config.exclude_disorder_txt)
        old_n = len(test_files)
        test_files = [path for path in test_files if stem_from_pt_path(path) not in excluded_stems]
        excluded_cnt = old_n - len(test_files)
    print(
        f"Test files: {len(test_files)} | Missing mapped pt: {len(missing)} | "
        f"Excluded disorder: {excluded_cnt}"
    )

    dataset, build_stats = build_joint_dataset(
        test_files,
        heavy_noise_max=config.heavy_noise_max,
        is_check_dist=not config.no_dist_check,
        dist_min=config.dist_min,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    heavy_model_path = resolve_checkpoint_path(
        requested_path=config.heavy_model_path,
        kind="heavy",
        repo_id=config.hf_repo_id,
    )
    hydro_model_path = resolve_checkpoint_path(
        requested_path=config.hydro_model_path,
        kind="hydro",
        repo_id=config.hf_repo_id,
    )

    heavy_model, heavy_num_classes = load_torchmd_model(
        heavy_model_path,
        device=device,
        num_classes=config.heavy_num_classes,
        hidden_channels=config.hidden_channels,
    )
    hydro_model, hydro_num_classes = load_torchmd_model(
        hydro_model_path,
        device=device,
        num_classes=config.hydro_num_classes,
        hidden_channels=config.hidden_channels,
    )
    print(
        f"heavy_num_classes={heavy_num_classes} hydro_num_classes={hydro_num_classes} "
        f"heavy_model_path={heavy_model_path} hydro_model_path={hydro_model_path}"
    )

    stats, both_cif_paths, heavy_init_cif_paths, heavy_cif_paths = infer_joint(
        config=config,
        heavy_model=heavy_model,
        hydro_model=hydro_model,
        loader=loader,
        device=device,
    )

    _write_lines(config.heavy_init_cif_out, heavy_init_cif_paths)
    _write_lines(config.heavy_cif_out, heavy_cif_paths)
    _write_lines(config.both_cif_out, both_cif_paths)

    elem_lines = _build_heavy_elem_lines(stats)
    hydro_lines = _build_hydro_group_lines(stats)
    _write_lines(config.heavy_elem_f1_out, elem_lines)
    _write_lines(config.hydro_group_f1_out, hydro_lines)

    total = max(stats["total"], 1)
    hydro_den = max(stats["hydro_eligible"], 1)
    heavy_init_atom_den = max(stats["heavy_init_atom_total"], 1)
    heavy_atom_den = max(stats["heavy_atom_total"], 1)
    hydro_atom_den = max(stats["hydro_atom_total"], 1)
    print(
        f"[Joint] total={stats['total']} "
        f"heavy_init_correct={stats['heavy_init_correct']} ({stats['heavy_init_correct']/total*100:.2f}%) "
        f"heavy_init_atom_acc={stats['heavy_init_atom_correct']/heavy_init_atom_den*100:.2f}% "
        f"heavy_correct={stats['heavy_correct']} ({stats['heavy_correct']/total*100:.2f}%) "
        f"heavy_atom_acc={stats['heavy_atom_correct']/heavy_atom_den*100:.2f}% "
        f"hydro_correct={stats['hydro_correct']} ({stats['hydro_correct']/hydro_den*100:.2f}% on eligible={stats['hydro_eligible']}) "
        f"hydro_atom_acc={stats['hydro_atom_correct']/hydro_atom_den*100:.2f}% "
        f"both_correct={stats['both_correct']} ({stats['both_correct']/total*100:.2f}%)"
    )
    print(
        f"[Joint] heavy_fail_forward={stats['heavy_fail_forward']} "
        f"hydro_fail_forward={stats['hydro_fail_forward']} "
        f"hydro_adjust_changed_atoms={stats['hydro_adjust_changed_atoms']} "
        f"heavy_one_correction_applied={stats['heavy_one_correction_applied']} "
        f"heavy_k_correction_tried_structures={stats['heavy_k_correction_tried_structures']} "
        f"heavy_k_correction_total_trials={stats['heavy_k_correction_total_trials']} "
        f"heavy_k_correction_hit={stats['heavy_k_correction_hit']} "
        f"hydro_k_correction_tried_structures={stats['hydro_k_correction_tried_structures']} "
        f"hydro_k_correction_total_trials={stats['hydro_k_correction_total_trials']} "
        f"hydro_k_correction_hit={stats['hydro_k_correction_hit']}"
    )
    print(f"[BuildStats] {build_stats}")
    print(f"[HeavyInitCorrect] saved {len(heavy_init_cif_paths)} paths -> {config.heavy_init_cif_out}")
    print(f"[HeavyCorrect] saved {len(heavy_cif_paths)} paths -> {config.heavy_cif_out}")
    print(f"[BothCorrect] saved {len(both_cif_paths)} paths -> {config.both_cif_out}")
    print(f"[HeavyElemF1] saved {len(elem_lines)-1} elements -> {config.heavy_elem_f1_out}")
    print(f"[HydroGroupF1] saved {len(hydro_lines)-1} groups -> {config.hydro_group_f1_out}")


def build_parser() -> argparse.ArgumentParser:
    timestamp = get_run_timestamp()
    parser = argparse.ArgumentParser(description="CrystalX joint evaluator")
    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--heavy_model_path", type=str, default=JointEvalConfig.heavy_model_path)
    parser.add_argument("--hydro_model_path", type=str, default=JointEvalConfig.hydro_model_path)
    parser.add_argument("--hf_repo_id", type=str, default=JointEvalConfig.hf_repo_id)
    parser.add_argument("--test_years", type=int, nargs="+", default=list(JointEvalConfig.test_years))
    parser.add_argument("--pt_prefix", type=str, default=JointEvalConfig.pt_prefix)
    parser.add_argument("--pt_suffix", type=str, default=JointEvalConfig.pt_suffix)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--split_mode", type=str, choices=["year", "random"], default=JointEvalConfig.split_mode)
    parser.add_argument("--random_train_ratio", type=float, default=JointEvalConfig.random_train_ratio)
    parser.add_argument("--random_test_ratio", type=float, default=JointEvalConfig.random_test_ratio)
    parser.add_argument("--split_seed", type=int, default=JointEvalConfig.split_seed)
    parser.add_argument("--seed", type=int, default=JointEvalConfig.seed)
    parser.add_argument("--device", type=str, default=JointEvalConfig.device, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=JointEvalConfig.batch_size)
    parser.add_argument("--heavy_num_classes", type=int, default=JointEvalConfig.heavy_num_classes)
    parser.add_argument("--hydro_num_classes", type=int, default=JointEvalConfig.hydro_num_classes)
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=JointEvalConfig.hidden_channels,
        help="Checkpoint hidden width. Use 0 to infer automatically from each checkpoint.",
    )
    parser.add_argument("--restrict_to_sfac", action="store_true")
    parser.add_argument("--allow_one_mismatch", action="store_true")
    parser.add_argument("--allow_one_correction", action="store_true")
    parser.add_argument("--allow_k_corrections", type=int, default=JointEvalConfig.allow_k_corrections)
    parser.add_argument("--hydro_allow_k_corrections", type=int, default=JointEvalConfig.hydro_allow_k_corrections)
    parser.add_argument("--disable_hydro_graph_adjust", action="store_true")
    parser.add_argument("--graph_builder", type=str, default=JointEvalConfig.graph_builder, choices=["ase", "rdkit"])
    parser.add_argument("--ase_cutoff_mult", type=float, default=JointEvalConfig.ase_cutoff_mult)
    parser.add_argument("--ase_extra_cutoff", type=float, default=JointEvalConfig.ase_extra_cutoff)
    parser.add_argument("--ase_skin", type=float, default=JointEvalConfig.ase_skin)
    parser.add_argument("--rdkit_cov_factor", type=float, default=JointEvalConfig.rdkit_cov_factor)
    parser.add_argument("--heavy_noise_max", type=float, default=JointEvalConfig.heavy_noise_max)
    parser.add_argument("--no_dist_check", action="store_true")
    parser.add_argument("--dist_min", type=float, default=JointEvalConfig.dist_min)
    parser.add_argument("--cif_root", type=str, default=JointEvalConfig.cif_root)
    parser.add_argument("--exclude_disorder_txt", type=str, default=JointEvalConfig.exclude_disorder_txt)
    parser.add_argument("--heavy_init_cif_out", type=str, default=f"heavy_init_correct_cif_{timestamp}.txt")
    parser.add_argument("--heavy_cif_out", type=str, default=f"heavy_correct_cif_{timestamp}.txt")
    parser.add_argument("--both_cif_out", type=str, default=f"both_correct_cif_{timestamp}.txt")
    parser.add_argument("--heavy_elem_f1_out", type=str, default=f"heavy_elem_f1_{timestamp}.txt")
    parser.add_argument("--hydro_group_f1_out", type=str, default=f"hydro_group_f1_{timestamp}.txt")
    parser.add_argument(
        "--hydro_label_f1_out",
        dest="hydro_group_f1_out",
        type=str,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = JointEvalConfig(
        pt_dir=args.pt_dir,
        txt_path=args.txt_path,
        heavy_model_path=args.heavy_model_path,
        hydro_model_path=args.hydro_model_path,
        hf_repo_id=args.hf_repo_id,
        test_years=tuple(args.test_years),
        pt_prefix=args.pt_prefix,
        pt_suffix=args.pt_suffix,
        strict=args.strict,
        split_mode=args.split_mode,
        random_train_ratio=args.random_train_ratio,
        random_test_ratio=args.random_test_ratio,
        split_seed=args.split_seed,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        heavy_num_classes=args.heavy_num_classes,
        hydro_num_classes=args.hydro_num_classes,
        hidden_channels=args.hidden_channels,
        restrict_to_sfac=args.restrict_to_sfac,
        allow_one_mismatch=args.allow_one_mismatch,
        allow_one_correction=args.allow_one_correction,
        allow_k_corrections=args.allow_k_corrections,
        hydro_allow_k_corrections=args.hydro_allow_k_corrections,
        disable_hydro_graph_adjust=args.disable_hydro_graph_adjust,
        graph_builder=args.graph_builder,
        ase_cutoff_mult=args.ase_cutoff_mult,
        ase_extra_cutoff=args.ase_extra_cutoff,
        ase_skin=args.ase_skin,
        rdkit_cov_factor=args.rdkit_cov_factor,
        heavy_noise_max=args.heavy_noise_max,
        no_dist_check=args.no_dist_check,
        dist_min=args.dist_min,
        cif_root=args.cif_root,
        exclude_disorder_txt=args.exclude_disorder_txt,
        heavy_init_cif_out=args.heavy_init_cif_out,
        heavy_cif_out=args.heavy_cif_out,
        both_cif_out=args.both_cif_out,
        heavy_elem_f1_out=args.heavy_elem_f1_out,
        hydro_group_f1_out=args.hydro_group_f1_out,
    )
    run_evaluation(config)


if __name__ == "__main__":
    main()
