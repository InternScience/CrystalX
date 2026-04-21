"""Single-case heavy-atom prediction entrypoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from crystalx_infer.common.checkpoints import HEAVY_CHECKPOINT, resolve_checkpoint_path
from crystalx_infer.common.chem import atomic_num_from_symbol, atomic_symbol_from_z
from crystalx_infer.common.heavy import (
    build_sorted_init_z_from_ratio,
    deduplicate_cartesian_positions,
    parse_sfac_unit_from_shelx,
    run_two_pass_heavy_prediction,
    trim_shelxt_pred_for_unit_divisibility,
)
from crystalx_infer.common.modeling import DEFAULT_CHECKPOINT_HIDDEN_CHANNELS, load_torchmd_model
from crystalx_infer.common.paths import HeavyPredictionPaths
from crystalx_infer.common.runtime import resolve_device
from crystalx_infer.common.shelx import (
    copy_file,
    get_equiv_pos2,
    load_shelxt,
    load_shelxt_final,
    update_shelxt,
)


@dataclass
class HeavyPredictConfig:
    fname: str = ""
    work_dir: str = ""
    res_path: str = ""
    hkl_path: str = ""
    model_path: str = HEAVY_CHECKPOINT.filename
    hf_repo_id: str = ""
    device: str = "auto"
    num_classes: int = 98
    hidden_channels: int = DEFAULT_CHECKPOINT_HIDDEN_CHANNELS
    refine_round: int = 16
    topk: int = 5


def build_topk_payload(prob, candidate_elements, atom_names, predicted_atomic_numbers, requested_topk):
    if requested_topk <= 0:
        raise ValueError("--topk must be >= 1")

    if prob.ndim != 2:
        raise ValueError(f"Expected 2D probability tensor, got shape={tuple(prob.shape)}")

    candidate_elements_cpu = candidate_elements.detach().cpu()
    prob_cpu = prob.detach().cpu()
    predicted_cpu = predicted_atomic_numbers.detach().cpu()
    atom_count = int(prob_cpu.shape[0])
    if int(predicted_cpu.shape[0]) != atom_count:
        raise ValueError(
            "Heavy probability rows do not match predicted labels: "
            f"prob_rows={atom_count} predicted={int(predicted_cpu.shape[0])}"
        )
    if len(atom_names) != atom_count:
        raise ValueError(
            "Heavy probability rows do not match atom names: "
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
            atomic_number = int(candidate_elements_cpu[int(class_idx)].item())
            ranked.append(
                {
                    "rank": rank_idx,
                    "atomic_number": atomic_number,
                    "symbol": atomic_symbol_from_z(atomic_number),
                    "probability": float(prob_value),
                }
            )

        predicted_atomic_number = int(predicted_cpu[atom_idx].item())
        atoms.append(
            {
                "atom_index": atom_idx + 1,
                "input_atom_name": atom_names[atom_idx],
                "predicted_atomic_number": predicted_atomic_number,
                "predicted_symbol": atomic_symbol_from_z(predicted_atomic_number),
                "coverage_adjusted": bool(
                    ranked
                    and predicted_atomic_number != int(ranked[0]["atomic_number"])
                ),
                "topk": ranked,
            }
        )

    return {
        "fname": None,
        "requested_topk": int(requested_topk),
        "effective_topk": int(effective_topk),
        "candidate_elements": [
            {
                "atomic_number": int(atomic_number.item()),
                "symbol": atomic_symbol_from_z(int(atomic_number.item())),
            }
            for atomic_number in candidate_elements_cpu
        ],
        "atoms": atoms,
    }


def load_heavy_input_from_res(res_path: str):
    real_frac, shelxt_pred, _, isotropy_list = load_shelxt(
        res_path,
        is_check_sfac=True,
    )
    _, atom_names, _, _ = load_shelxt(
        res_path,
        is_atom_name=True,
    )
    if len(shelxt_pred) > 0:
        return real_frac, shelxt_pred, atom_names, isotropy_list

    # Fallback for refined/template .res files that store atom rows after FVAR/ANIS.
    real_frac, shelxt_pred = load_shelxt_final(
        res_path,
        begin_flag="FVAR",
        is_hydro=False,
    )
    _, atom_names = load_shelxt_final(
        res_path,
        begin_flag="FVAR",
        is_atom_name=True,
        is_hydro=False,
    )
    if len(shelxt_pred) > 0:
        return real_frac, shelxt_pred, atom_names, None

    real_frac, shelxt_pred = load_shelxt_final(
        res_path,
        begin_flag="ANIS",
        is_hydro=False,
    )
    _, atom_names = load_shelxt_final(
        res_path,
        begin_flag="ANIS",
        is_atom_name=True,
        is_hydro=False,
    )
    return real_frac, shelxt_pred, atom_names, None


def run_prediction(config: HeavyPredictConfig) -> None:
    import numpy as np
    import torch
    from cctbx.xray.structure import structure
    from iotbx.shelx import crystal_symmetry_from_ins
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    device = resolve_device(config.device)
    paths = HeavyPredictionPaths.from_values(
        work_dir=config.work_dir,
        fname=config.fname,
        res_path=config.res_path,
        hkl_path=config.hkl_path,
    )
    if config.topk <= 0:
        raise ValueError("--topk must be >= 1")
    if not paths.res_input_path.exists():
        raise FileNotFoundError(f"Heavy input .res not found: {paths.res_input_path}")

    real_frac, shelxt_pred, atom_names, isotropy_list = load_heavy_input_from_res(
        str(paths.res_input_path)
    )
    if len(shelxt_pred) == 0:
        raise ValueError(f"No heavy-atom rows found in {paths.res_input_path}")

    sfac = None
    unit = None
    non_h_unit_total = 0
    try:
        sfac, unit = parse_sfac_unit_from_shelx(str(paths.res_input_path))
        non_h_unit_total = sum(
            int(count) for symbol, count in zip(sfac, unit) if symbol.upper() != "H"
        )
    except Exception as exc:
        print(f"[WARN] Cannot parse SFAC/UNIT from {paths.res_input_path.name}, skip trim: {exc}")

    if non_h_unit_total > 0:
        real_frac, shelxt_pred, isotropy_list, _ = trim_shelxt_pred_for_unit_divisibility(
            real_frac,
            shelxt_pred,
            isotropy_list,
            non_h_unit_total,
        )
    atom_names = atom_names[: len(shelxt_pred)]

    real_num = len(shelxt_pred)
    crystal_symmetry = crystal_symmetry_from_ins.extract_from(str(paths.res_input_path))
    coarse_structure = structure(crystal_symmetry=crystal_symmetry)

    real_frac_array = np.array(real_frac)
    expanded_cart, expanded_symbols = get_equiv_pos2(
        real_frac_array,
        shelxt_pred,
        coarse_structure,
        radius=3.2,
    )
    expanded_z = [atomic_num_from_symbol(symbol.capitalize()) for symbol in expanded_symbols]

    real_cart, z = deduplicate_cartesian_positions(expanded_cart, expanded_z)
    if len(z) >= real_num:
        if sfac is not None and unit is not None:
            z = build_sorted_init_z_from_ratio(sfac, unit, real_num) + z[real_num:]
        else:
            print("[WARN] SFAC/UNIT init fallback to SHELXT z: SFAC/UNIT unavailable.")

    mask = torch.zeros(len(z), dtype=torch.bool)
    mask[:real_num] = True
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
        kind="heavy",
        repo_id=config.hf_repo_id,
    )
    model, resolved_num_classes = load_torchmd_model(
        model_path,
        device=device,
        num_classes=config.num_classes,
        hidden_channels=config.hidden_channels,
    )
    print(f"[INFO] device={device} num_classes={resolved_num_classes} model={model_path}")

    predicted_symbols = []
    topk_payload = None
    with torch.inference_mode():
        for data in tqdm(test_loader, desc="Predict heavy"):
            data = data.to(device)
            candidate_elements = torch.unique(data.z)
            prob, predicted, _ = run_two_pass_heavy_prediction(
                model=model,
                input_z=data.z,
                pos=data.pos,
                batch=data.batch,
                mask=data.mask,
                candidate_elements=candidate_elements,
                enforce_element_coverage=True,
            )
            predicted_cpu = predicted.to("cpu")
            predicted_symbols = [atomic_symbol_from_z(item) for item in predicted_cpu.tolist()]
            topk_payload = build_topk_payload(
                prob=prob,
                candidate_elements=candidate_elements,
                atom_names=atom_names,
                predicted_atomic_numbers=predicted_cpu,
                requested_topk=config.topk,
            )

    if topk_payload is None:
        raise RuntimeError("Heavy prediction did not produce probability outputs.")
    topk_payload["fname"] = paths.output_stem

    update_shelxt(
        str(paths.res_input_path),
        str(paths.output_ins_path),
        predicted_symbols,
        no_sfac=True,
        refine_round=config.refine_round,
    )
    if paths.hkl_input_path is not None:
        copy_file(str(paths.hkl_input_path), str(paths.output_hkl_path))
    with open(paths.topk_path, "w", encoding="utf-8") as file_obj:
        json.dump(topk_payload, file_obj, indent=2)
    print(
        f"[INFO] Wrote heavy prediction: {paths.output_ins_path.name} | "
        f"atoms={len(predicted_symbols)}"
    )
    if paths.hkl_input_path is not None:
        print(f"[INFO] Copied heavy HKL: {paths.output_hkl_path.name}")
    else:
        print("[INFO] No HKL input provided; skipped writing heavy _AI.hkl.")
    print(f"[INFO] Saved heavy top-k probabilities: {paths.topk_path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CrystalX heavy-atom predictor. Use --res_path for standalone .res input, "
            "or keep --fname/--work_dir for the legacy SHELXT naming convention."
        )
    )
    parser.add_argument(
        "--fname",
        type=str,
        default=HeavyPredictConfig.fname,
        help="Output stem. In legacy mode this is the case prefix; in --res_path mode it overrides the derived stem.",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=HeavyPredictConfig.work_dir,
        help="Legacy case directory, or output directory when --res_path is used.",
    )
    parser.add_argument(
        "--res_path",
        type=str,
        default=HeavyPredictConfig.res_path,
        help="Standalone heavy-stage input .res path. When set, the predictor reads atom names, coordinates, and SFAC/UNIT from this file.",
    )
    parser.add_argument(
        "--hkl_path",
        type=str,
        default=HeavyPredictConfig.hkl_path,
        help="Optional .hkl paired with --res_path. When provided, it is copied to the _AI.hkl output.",
    )
    parser.add_argument("--model_path", type=str, default=HeavyPredictConfig.model_path)
    parser.add_argument("--hf_repo_id", type=str, default=HeavyPredictConfig.hf_repo_id)
    parser.add_argument("--device", type=str, default=HeavyPredictConfig.device, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num_classes", type=int, default=HeavyPredictConfig.num_classes)
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=HeavyPredictConfig.hidden_channels,
        help="Checkpoint hidden width. Use 0 to infer automatically from the checkpoint.",
    )
    parser.add_argument("--refine_round", type=int, default=HeavyPredictConfig.refine_round)
    parser.add_argument(
        "--topk",
        type=int,
        default=HeavyPredictConfig.topk,
        help="How many highest-probability classes to save per atom in the JSON report.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_prediction(HeavyPredictConfig(**vars(args)))


if __name__ == "__main__":
    main()
