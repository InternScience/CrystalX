"""Single-case hydrogen-count prediction entrypoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from crystalx_infer.common.checkpoints import HYDRO_CHECKPOINT, resolve_checkpoint_path
from crystalx_infer.common.hydrogen import (
    build_graph_from_ins,
    build_hfix_summary,
    build_hydro_inputs,
    compute_alignment_metrics,
    gen_hfix_ins,
    load_heavy_from_template,
    load_template_atom_names,
    predict_hydrogen_counts,
)
from crystalx_infer.common.modeling import DEFAULT_CHECKPOINT_HIDDEN_CHANNELS, load_torchmd_model
from crystalx_infer.common.paths import HydroPredictionPaths
from crystalx_infer.common.runtime import resolve_device
from crystalx_infer.common.shelx import copy_file, get_bond, update_shelxt_hydro


@dataclass
class HydroPredictConfig:
    fname: str = ""
    work_dir: str = ""
    res_path: str = ""
    hkl_path: str = ""
    model_path: str = HYDRO_CHECKPOINT.filename
    hf_repo_id: str = ""
    device: str = "auto"
    num_classes: int = 8
    hidden_channels: int = DEFAULT_CHECKPOINT_HIDDEN_CHANNELS
    topk: int = 5


def build_topk_payload(prob, atom_names, predicted_hydrogen_counts, requested_topk):
    if requested_topk <= 0:
        raise ValueError("--topk must be >= 1")

    if prob.ndim != 2:
        raise ValueError(f"Expected 2D probability tensor, got shape={tuple(prob.shape)}")

    prob_cpu = prob.detach().cpu()
    predicted_cpu = predicted_hydrogen_counts.detach().cpu()
    atom_count = int(prob_cpu.shape[0])
    if int(predicted_cpu.shape[0]) != atom_count:
        raise ValueError(
            "Hydrogen probability rows do not match predicted labels: "
            f"prob_rows={atom_count} predicted={int(predicted_cpu.shape[0])}"
        )
    if len(atom_names) != atom_count:
        raise ValueError(
            "Hydrogen probability rows do not match atom names: "
            f"prob_rows={atom_count} atom_names={len(atom_names)}"
        )
    effective_topk = min(int(requested_topk), int(prob_cpu.shape[1]))
    topk_prob, topk_idx = prob_cpu.topk(k=effective_topk, dim=1)

    atoms = []
    for atom_idx in range(atom_count):
        ranked = []
        for rank_idx, (prob_value, class_idx) in enumerate(
            zip(topk_prob[atom_idx].tolist(), topk_idx[atom_idx].tolist()),
            start=1,
        ):
            ranked.append(
                {
                    "rank": rank_idx,
                    "hydrogen_count": int(class_idx),
                    "probability": float(prob_value),
                }
            )

        atoms.append(
            {
                "atom_index": atom_idx + 1,
                "atom_name": atom_names[atom_idx],
                "predicted_hydrogen_count": int(predicted_cpu[atom_idx].item()),
                "topk": ranked,
            }
        )

    return {
        "fname": None,
        "requested_topk": int(requested_topk),
        "effective_topk": int(effective_topk),
        "atoms": atoms,
    }


def run_prediction(config: HydroPredictConfig) -> None:
    import numpy as np
    import torch
    from cctbx.xray.structure import structure
    from iotbx.shelx import crystal_symmetry_from_ins
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    device = resolve_device(config.device)
    paths = HydroPredictionPaths.from_values(
        work_dir=config.work_dir,
        fname=config.fname,
        res_path=config.res_path,
        hkl_path=config.hkl_path,
    )
    if config.topk <= 0:
        raise ValueError("--topk must be >= 1")
    if not paths.res_file_path.exists() and not paths.ins_file_path.exists():
        raise FileNotFoundError(
            f"Hydro input template not found: res={paths.res_file_path} ins={paths.ins_file_path}"
        )

    real_frac, shelxt_pred, sym_src_path = load_heavy_from_template(
        res_file_path=str(paths.res_file_path),
        ins_file_path=str(paths.ins_file_path),
    )
    if len(shelxt_pred) == 0:
        raise ValueError(f"No heavy-atom rows found in hydro template: {sym_src_path}")

    crystal_symmetry = crystal_symmetry_from_ins.extract_from(sym_src_path)
    crystal_structure = structure(crystal_symmetry=crystal_symmetry)
    real_num, real_cart, z = build_hydro_inputs(real_frac, shelxt_pred, crystal_structure)
    main_align_ok, main_align_max_abs = compute_alignment_metrics(
        real_frac,
        real_cart,
        crystal_structure,
    )

    mask = torch.zeros(len(z), dtype=torch.bool)
    mask[: len(shelxt_pred)] = True
    test_loader = DataLoader(
        [
            Data(
                z=torch.tensor(z, dtype=torch.long),
                pos=torch.from_numpy(np.array(real_cart, dtype=np.float32)),
                mask=mask,
            )
        ],
        batch_size=1,
        shuffle=False,
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
    print(f"[INFO] device={device} num_classes={resolved_num_classes} model={model_path}")

    prob = None
    predicted = None
    with torch.inference_mode():
        for data in tqdm(test_loader, desc="Predict hydro"):
            data = data.to(device)
            prob, predicted = predict_hydrogen_counts(
                model=model,
                input_z=data.z,
                pos=data.pos,
                batch=data.batch,
                mask=data.mask,
            )
            predicted = predicted.to("cpu")

    if prob is None or predicted is None:
        raise RuntimeError("Hydrogen prediction did not produce probability outputs.")
    pred_h_per_atom = [int(value) for value in predicted.tolist()]
    pred_h_total = int(sum(pred_h_per_atom))

    mol_graph = None
    used_lst = ""
    for candidate in paths.lst_candidates:
        if candidate.exists():
            mol_graph = get_bond(str(candidate))
            used_lst = str(candidate)
            break

    if mol_graph is None:
        template_for_graph = (
            str(paths.res_file_path) if paths.res_file_path.exists() else str(paths.ins_file_path)
        )
        print(
            "[WARN] No .lst found for bond table, fallback to RDKit covalent-radius "
            "connectivity: bond if distance <= (r_i + r_j) * 1.20 and distance > 0.10 A."
        )
        mol_graph = build_graph_from_ins(template_for_graph)
    else:
        print(f"[INFO] Bond table from: {used_lst}")

    template_path = str(paths.res_file_path) if paths.res_file_path.exists() else str(paths.ins_file_path)
    atom_names_in_template = load_template_atom_names(template_path)
    for atom_name in atom_names_in_template:
        mol_graph.setdefault(atom_name, [])

    if len(atom_names_in_template) != len(pred_h_per_atom):
        raise ValueError(
            "Hydro prediction length mismatch: "
            f"template_heavy_atoms={len(atom_names_in_template)} "
            f"vs predicted={len(pred_h_per_atom)} "
            f"(template={template_path})"
        )

    hfix_ins = gen_hfix_ins(template_path, mol_graph, predicted)
    update_shelxt_hydro(template_path, str(paths.output_ins_path), hfix_ins)
    if paths.hkl_input_path is not None:
        copy_file(str(paths.hkl_input_path), str(paths.output_hkl_path))

    topk_payload = build_topk_payload(
        prob=prob,
        atom_names=atom_names_in_template,
        predicted_hydrogen_counts=predicted,
        requested_topk=config.topk,
    )
    topk_payload["fname"] = paths.output_stem
    with open(paths.topk_path, "w", encoding="utf-8") as file_obj:
        json.dump(topk_payload, file_obj, indent=2)

    hfix_total_h = build_hfix_summary(hfix_ins)
    pred_summary = {
        "fname": paths.output_stem,
        "pred_h_total": pred_h_total,
        "pred_h_per_atom": pred_h_per_atom,
        "pred_atom_count": int(len(pred_h_per_atom)),
        "heavy_atom_count_template": int(real_num),
        "coord_align_ok": bool(main_align_ok),
        "coord_align_max_abs_diff": float(main_align_max_abs),
        "hfix_line_count": int(len(hfix_ins)),
        "hfix_total_h": int(hfix_total_h),
        "pred_vs_hfix_match": bool(int(pred_h_total) == int(hfix_total_h)),
        "topk_json": paths.topk_path.name,
        "requested_topk": int(config.topk),
        "effective_topk": int(topk_payload["effective_topk"]),
    }
    with open(paths.summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(pred_summary, file_obj, indent=2)

    if paths.hkl_input_path is not None:
        print(f"[INFO] Copied hydro HKL: {paths.output_hkl_path.name}")
    else:
        print("[INFO] No HKL input provided; skipped writing hydro.hkl output.")
    print(f"[INFO] Saved hydro prediction summary: {paths.summary_path.name}")
    print(f"[INFO] Saved hydro top-k probabilities: {paths.topk_path.name}")
    print(f"[INFO] Predicted total H: {pred_h_total}")
    print(
        f"[INFO] HFIX implied total H: {pred_summary['hfix_total_h']} | "
        f"pred_vs_hfix_match={pred_summary['pred_vs_hfix_match']}"
    )
    print(
        f"[INFO] Coord align check: ok={pred_summary['coord_align_ok']} "
        f"max_abs_diff={pred_summary['coord_align_max_abs_diff']:.3e}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CrystalX hydrogen-count predictor. Use --res_path for standalone .res input, "
            "or keep --fname/--work_dir for the legacy naming convention."
        )
    )
    parser.add_argument(
        "--fname",
        type=str,
        default=HydroPredictConfig.fname,
        help="Output stem. In legacy mode this is the case prefix; in --res_path mode it overrides the derived stem.",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=HydroPredictConfig.work_dir,
        help="Legacy case directory, or output directory when --res_path is used.",
    )
    parser.add_argument(
        "--res_path",
        type=str,
        default=HydroPredictConfig.res_path,
        help="Standalone hydro-stage input .res path. When set, the predictor reads heavy atoms and symmetry directly from this file.",
    )
    parser.add_argument(
        "--hkl_path",
        type=str,
        default=HydroPredictConfig.hkl_path,
        help="Optional .hkl paired with --res_path. When provided, it is copied to the hydro output.",
    )
    parser.add_argument("--model_path", type=str, default=HydroPredictConfig.model_path)
    parser.add_argument("--hf_repo_id", type=str, default=HydroPredictConfig.hf_repo_id)
    parser.add_argument("--device", type=str, default=HydroPredictConfig.device, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num_classes", type=int, default=HydroPredictConfig.num_classes)
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=HydroPredictConfig.hidden_channels,
        help="Checkpoint hidden width. Use 0 to infer automatically from the checkpoint.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=HydroPredictConfig.topk,
        help="How many highest-probability classes to save per atom in the JSON report.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_prediction(HydroPredictConfig(**vars(args)))


if __name__ == "__main__":
    main()
