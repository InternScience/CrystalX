import os
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from rdkit import Chem
from scipy.spatial.distance import cdist

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from crystalx_infer.models.torchmd_et import TorchMD_ET
from crystalx_infer.models.noise_output_model import EquivariantScalar
from crystalx_infer.models.torchmd_net import TorchMD_Net
from crystalx_infer.common.utils import (
    copy_file,
    gen_hfix_ins,
    get_bond,
    load_shelxt,
    update_shelxt_hydro,
)


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


def split_by_year_txt(
    txt_path: str,
    pt_dir: str,
    test_years=(2022, 2023, 2024),
    pt_prefix="",
    pt_suffix=".pt",
    strict=False,
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
                print(f"[WARN] txt line {ln} format error: {line}")
                continue

            year = parts[0]
            cif_name = parts[-1]
            cif_stem = os.path.splitext(os.path.basename(cif_name))[0]
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


def build_simple_in_memory_dataset(file_list, is_check_dist=False, is_filter=False):
    dataset = []
    cnt = 0
    max_h = -1
    equiv_cnt = 0
    dist_error_cnt = 0
    c_h_gt3_skip_cnt = 0

    for fname in tqdm(file_list, desc="Build dataset"):
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

        # Skip sample if any Carbon has hydro_gt > 3.
        pair_n = min(len(y), len(hydro_num))
        has_c_h_gt3 = False
        for i in range(pair_n):
            if y[i] == 6 and hydro_num[i] > 3:
                has_c_h_gt3 = True
                break
        if has_c_h_gt3:
            c_h_gt3_skip_cnt += 1
            continue

        max_h = max(max_h, max(hydro_num))

        if is_filter:
            if max(hydro_num) > 4:
                continue
            if 6 not in _z or 7 not in _z or 8 not in _z:
                continue

        _real_cart = mol_info["pos"]
        real_cart = []
        z_num = []
        main_z = []
        for i in range(_real_cart.shape[0]):
            posi = _real_cart[i].tolist()
            if posi not in real_cart:
                real_cart.append(posi)
                z_num.append(_z[i])
                main_z.append(_z[i])
        real_cart = np.array(real_cart, dtype=np.float32)

        mask = np.array([0] * len(z_num))
        mask[: len(hydro_num)] = 1
        mask = torch.from_numpy(mask).bool()

        if is_check_dist:
            distance_matrix = cdist(real_cart, real_cart)
            distance_matrix = distance_matrix + 10 * np.eye(distance_matrix.shape[0])
            if np.min(distance_matrix) < 0.1:
                dist_error_cnt += 1
                continue

        dataset.append(
            Data(
                z=torch.tensor(z_num, dtype=torch.long),
                main_z=torch.tensor(main_z, dtype=torch.long),
                y=torch.tensor(hydro_num, dtype=torch.long),
                pos=torch.from_numpy(real_cart),
                main_gt=torch.tensor(y, dtype=torch.long),
                fname=fname,
                mask=mask,
            )
        )
        cnt += 1

    print(
        f"[Build] kept={cnt} equiv_cnt={equiv_cnt} dist_drop={dist_error_cnt} "
        f"skip_C_h_gt3={c_h_gt3_skip_cnt}"
    )
    return dataset, max_h


def infer_num_classes_from_state_dict(state_dict):
    key = "output_model.output_network.1.update_net.2.bias"
    if key in state_dict and state_dict[key].ndim == 1:
        dim = int(state_dict[key].shape[0])
        if dim % 2 == 0 and dim > 0:
            return dim // 2

    key = "output_model.output_network.1.update_net.2.weight"
    if key in state_dict and state_dict[key].ndim == 2:
        dim = int(state_dict[key].shape[0])
        if dim % 2 == 0 and dim > 0:
            return dim // 2

    key = "output_model.output_network.1.vec2_proj.weight"
    if key in state_dict and state_dict[key].ndim == 2:
        return int(state_dict[key].shape[0])

    raise ValueError("Cannot infer num_classes from checkpoint. Please pass --num_classes.")


def _resolve_case_files(refined_dir, fname0):
    stem_full = os.path.splitext(os.path.basename(fname0))[0]
    stem_short = stem_full[6:] if stem_full.startswith("equiv_") else stem_full
    candidates = [
        os.path.join(refined_dir, stem_full),
        os.path.join(refined_dir, stem_short),
    ]
    for c in candidates:
        case_name = os.path.basename(c)
        files = {
            "case_dir": c,
            "case_name": case_name,
            "hkl": os.path.join(c, f"{case_name}_AI.hkl"),
            "ins": os.path.join(c, f"{case_name}_AI.ins"),
            "lst": os.path.join(c, f"{case_name}_AI.lst"),
            "res": os.path.join(c, f"{case_name}_AI.res"),
        }
        if os.path.exists(files["ins"]) and os.path.exists(files["lst"]):
            return files
    return None


def _atom_symbol_from_label(atom_label):
    token = "".join(ch for ch in atom_label if ch.isalpha())
    if not token:
        return ""
    if len(token) == 1:
        return token.upper()
    return token[0].upper() + token[1:].lower()


def _atomic_symbol_from_z(z):
    try:
        return Chem.Atom(int(z)).GetSymbol()
    except Exception:
        return ""


def _build_mol_graph_from_equiv(
    main_z_mask,
    pos_mask,
    ase_cutoff_mult=1.10,
    ase_extra_cutoff=0.00,
    ase_skin=0.00,
):
    """
    Build an index-aligned graph from equiv_gt(main_z) + coordinates using ASE NeighborList.
    Returns per-atom symbols and degrees in the same order as model prediction.
    """
    try:
        from ase import Atoms
        from ase.neighborlist import NeighborList, natural_cutoffs
    except Exception as e:
        raise ImportError(
            "ASE is required for mol-graph adjustment. Install with: pip install ase"
        ) from e

    z_list = [int(x) for x in main_z_mask.detach().cpu().tolist()]
    atom_symbols = [_atomic_symbol_from_z(z) for z in z_list]
    n = len(z_list)
    if n == 0:
        return atom_symbols, []

    pos = pos_mask.detach().cpu().numpy()
    if pos.ndim != 2 or pos.shape[0] != n:
        raise ValueError(f"Invalid pos shape for graph build: {tuple(pos.shape)}, n={n}")

    atoms = Atoms(symbols=atom_symbols, positions=pos)
    cutoffs = natural_cutoffs(atoms, mult=ase_cutoff_mult)
    cutoffs = [float(c) + float(ase_extra_cutoff) for c in cutoffs]
    nl = NeighborList(
        cutoffs=cutoffs,
        self_interaction=False,
        bothways=True,
        skin=ase_skin,
    )
    nl.update(atoms)

    degree = []
    for i in range(n):
        neigh_idx, _ = nl.get_neighbors(i)
        degree.append(int(len(neigh_idx)))
    return atom_symbols, degree


def _build_mol_graph_from_equiv_rdkit(
    main_z_mask,
    pos_mask,
    rdkit_cov_factor=1.30,
):
    """
    Build an index-aligned graph from equiv_gt(main_z) + coordinates using RDKit.
    Returns per-atom symbols and degrees in the same order as model prediction.
    """
    try:
        from rdkit import Chem as _Chem
        from rdkit.Geometry import Point3D
        from rdkit.Chem import rdDetermineBonds
    except Exception as e:
        raise ImportError(
            "RDKit with rdDetermineBonds is required for rdkit graph builder."
        ) from e

    z_list = [int(x) for x in main_z_mask.detach().cpu().tolist()]
    atom_symbols = [_atomic_symbol_from_z(z) for z in z_list]
    n = len(z_list)
    if n == 0:
        return atom_symbols, []

    pos = pos_mask.detach().cpu().numpy()
    if pos.ndim != 2 or pos.shape[0] != n:
        raise ValueError(f"Invalid pos shape for graph build: {tuple(pos.shape)}, n={n}")

    rw = _Chem.RWMol()
    for z in z_list:
        rw.AddAtom(_Chem.Atom(int(z)))
    mol = rw.GetMol()

    conf = _Chem.Conformer(n)
    for i in range(n):
        x, y, z = pos[i].tolist()
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    try:
        rdDetermineBonds.DetermineConnectivity(mol, covFactor=float(rdkit_cov_factor))
    except TypeError:
        # Compatibility with older RDKit signatures
        rdDetermineBonds.DetermineConnectivity(mol)

    degree = [int(mol.GetAtomWithIdx(i).GetDegree()) for i in range(n)]
    return atom_symbols, degree


def _is_reasonable_hydrogen(atom_symbol, degree, h):
    if h < 0:
        return False
    if atom_symbol == "O":
        if h == 0:
            return True
        if h == 1:
            # Hydroxyl / coordinated OH
            return degree <= 2
        if h == 2:
            # Water-like oxygen (solvent), including weakly coordinated cases
            return degree <= 2
        if h == 3:
            # Rare hydronium-like case
            return degree == 0
        return False
    if atom_symbol == "N":
        if h == 0:
            return True
        if h == 1:
            return degree <= 3
        if h == 2:
            return degree <= 2
        if h == 3:
            return degree <= 1
        if h == 4:
            # Ammonium-like solvent/counter-ion
            return degree == 0
        return False
    if atom_symbol == "C":
        return 0 <= h <= max(0, 4 - degree)
    if atom_symbol == "B":
        return 0 <= h <= max(0, 3 - degree)
    return True


def _check_alignment_with_ins(atom_names, main_z_mask):
    ins_symbols = [_atom_symbol_from_label(name) for name in atom_names]
    ins_z = []
    for sym in ins_symbols:
        if not sym:
            ins_z.append(-1)
            continue
        try:
            ins_z.append(Chem.Atom(sym).GetAtomicNum())
        except Exception:
            ins_z.append(-1)

    target_z = [int(x) for x in main_z_mask.detach().cpu().tolist()]
    n = min(len(ins_z), len(target_z))
    mismatch_idx = [i for i in range(n) if ins_z[i] != target_z[i]]
    return {
        "ins_len": len(ins_z),
        "target_len": len(target_z),
        "len_mismatch": len(ins_z) != len(target_z),
        "mismatch_count": len(mismatch_idx),
        "mismatch_idx": mismatch_idx,
        "ins_symbols": ins_symbols,
        "ins_z": ins_z,
        "target_z": target_z,
    }


def _adjust_prediction_by_mol_graph(prob, predicted, atom_symbols, degree_list):
    """
    If top-1 hydrogen count is chemically unreasonable under mol-graph constraints,
    back off to top-2/top-3... for each atom independently.
    """
    sorted_indices = torch.argsort(prob, dim=1, descending=True)
    adjusted = predicted.clone()

    changed = 0
    valid_checked = 0
    atom_n = min(len(atom_symbols), len(degree_list), adjusted.shape[0])
    for i in range(atom_n):
        atom_symbol = atom_symbols[i]
        degree = int(degree_list[i])

        chosen = int(adjusted[i].item())
        for cand in sorted_indices[i]:
            h = int(cand.item())
            if _is_reasonable_hydrogen(atom_symbol, degree, h):
                chosen = h
                break
        if chosen != int(adjusted[i].item()):
            changed += 1
        adjusted[i] = chosen
        valid_checked += 1

    return adjusted, changed, valid_checked


@torch.no_grad()
def eval_validate(
    model,
    test_loader,
    device,
    dump_test_data=False,
    atom_analysis=False,
    refined_dir="all_refined_10",
    adjust_by_mol_graph=True,
    print_alignment_check=False,
    print_alignment_max_cases=20,
    graph_builder="ase",
    ase_cutoff_mult=1.10,
    ase_extra_cutoff=0.00,
    ase_skin=0.00,
    rdkit_cov_factor=1.30,
):
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

    for data in tqdm(test_loader, desc="Infer"):
        try:
            data = data.to(device)
            outputs, _ = model(data.z, data.pos, data.batch)
            pred = outputs[data.mask]
            pred = F.softmax(pred, dim=-1)
        except Exception as e:
            print(e)
            missing_cnt += 1
            continue

        main_correct = (data.main_gt == data.main_z[data.mask]).all()
        if not main_correct:
            continue

        _, predicted = torch.max(pred, dim=1)

        fname0 = data.fname[0] if isinstance(data.fname, (list, tuple)) else data.fname
        case_files = None
        if print_alignment_check or dump_test_data:
            case_files = _resolve_case_files(refined_dir, fname0)

        if adjust_by_mol_graph:
            try:
                if graph_builder == "rdkit":
                    atom_symbols, degree_list = _build_mol_graph_from_equiv_rdkit(
                        main_z_mask=data.main_z[data.mask],
                        pos_mask=data.pos[data.mask],
                        rdkit_cov_factor=rdkit_cov_factor,
                    )
                else:
                    atom_symbols, degree_list = _build_mol_graph_from_equiv(
                        main_z_mask=data.main_z[data.mask],
                        pos_mask=data.pos[data.mask],
                        ase_cutoff_mult=ase_cutoff_mult,
                        ase_extra_cutoff=ase_extra_cutoff,
                        ase_skin=ase_skin,
                    )
                predicted, changed, checked = _adjust_prediction_by_mol_graph(
                    prob=pred,
                    predicted=predicted,
                    atom_symbols=atom_symbols,
                    degree_list=degree_list,
                )
                graph_adjust_atoms_checked += checked
                graph_adjust_atom_changed += changed
                if changed > 0:
                    graph_adjust_case_changed += 1
            except Exception as e:
                print("[mol-graph adjust error]", e)
                graph_adjust_case_skipped += 1

        if print_alignment_check:
            if case_files is None:
                pass
            else:
                try:
                    _, atom_names, _, _ = load_shelxt(
                        case_files["ins"], begin_flag="ANIS", is_atom_name=True
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
                        if align_printed_cases < print_alignment_max_cases:
                            preview_idx = align_info["mismatch_idx"][:10]
                            preview = []
                            for idx in preview_idx:
                                ins_sym = align_info["ins_symbols"][idx]
                                tgt_z = align_info["target_z"][idx]
                                try:
                                    tgt_sym = Chem.Atom(int(tgt_z)).GetSymbol()
                                except Exception:
                                    tgt_sym = str(tgt_z)
                                preview.append(f"{idx}:{ins_sym}->{tgt_sym}")
                            print(
                                "[ALIGN-MISMATCH] "
                                f"sample={os.path.basename(str(fname0))} "
                                f"ins_len={align_info['ins_len']} target_len={align_info['target_len']} "
                                f"mismatch_count={align_info['mismatch_count']} "
                                f"preview={preview}"
                            )
                            align_printed_cases += 1
                        elif align_printed_cases == print_alignment_max_cases:
                            print("[ALIGN-MISMATCH] more mismatches suppressed...")
                            align_printed_cases += 1
                except Exception as e:
                    print("[ALIGN-CHECK error]", e)

        label = data.y

        correct_atom_num = (predicted == label).sum().item()
        correct_predictions += correct_atom_num

        all_atom_num = label.shape[0]
        total_atoms += all_atom_num
        total_mol += 1
        if correct_atom_num == all_atom_num:
            correct_mol += 1

        if dump_test_data:
            if case_files is None:
                dump_fail += 1
            else:
                hkl_file_path = case_files["hkl"]
                ins_file_path = case_files["ins"]
                lst_file_path = case_files["lst"]
                res_file_path = case_files["res"]
                case_dir = case_files["case_dir"]
                case_name = case_files["case_name"]
                if not (os.path.exists(hkl_file_path) and os.path.exists(res_file_path)):
                    dump_fail += 1
                    continue
                new_hkl_file_path = os.path.join(case_dir, f"{case_name}_AIhydro.hkl")
                new_res_file_path = os.path.join(case_dir, f"{case_name}_AIhydro.ins")
                predicted_cpu = predicted.to("cpu")
                copy_file(hkl_file_path, new_hkl_file_path)
                mol_graph = get_bond(lst_file_path)
                hfix_ins = gen_hfix_ins(ins_file_path, mol_graph, predicted_cpu)
                update_shelxt_hydro(res_file_path, new_res_file_path, hfix_ins)
                dump_ok += 1

        if atom_analysis:
            all_pred.append(predicted.cpu().numpy())
            all_label.append(label.cpu().numpy())

    if atom_analysis and len(all_pred) > 0:
        from sklearn.metrics import classification_report

        all_pred = np.concatenate(all_pred)
        all_label = np.concatenate(all_label)
        print(classification_report(all_label, all_pred))

    atom_accuracy = correct_predictions / total_atoms if total_atoms > 0 else 0.0
    mol_accuracy = correct_mol / total_mol if total_mol > 0 else 0.0

    model.train()
    adjust_stats = {
        "enabled": adjust_by_mol_graph,
        "graph_builder": graph_builder,
        "atoms_changed": graph_adjust_atom_changed,
        "cases_changed": graph_adjust_case_changed,
        "cases_skipped": graph_adjust_case_skipped,
        "atoms_checked": graph_adjust_atoms_checked,
        "align_checked_cases": align_checked_cases,
        "align_mismatch_cases": align_mismatch_cases,
        "align_len_mismatch_cases": align_len_mismatch_cases,
    }
    return atom_accuracy, mol_accuracy, missing_cnt, dump_ok, dump_fail, adjust_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument(
        "--test_years",
        type=int,
        nargs="+",
        default=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
    )
    parser.add_argument("--pt_prefix", type=str, default="equiv_")
    parser.add_argument("--pt_suffix", type=str, default=".pt")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=183)

    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--is_filter", action="store_true")
    parser.add_argument("--no_dist_check", action="store_true")
    parser.add_argument("--atom_analysis", action="store_true")
    parser.add_argument("--dump_test_data", action="store_true")
    parser.add_argument("--refined_dir", type=str, default="all_refined_10")
    parser.add_argument(
        "--disable_mol_graph_adjust",
        action="store_true",
        help="disable top-k fallback adjustment using mol graph constraints",
    )
    parser.add_argument(
        "--print_alignment_check",
        action="store_true",
        help="print mismatched atom-order cases between ins order and model target order",
    )
    parser.add_argument(
        "--print_alignment_max_cases",
        type=int,
        default=20,
        help="max number of mismatched cases to print",
    )
    parser.add_argument(
        "--ase_cutoff_mult",
        type=float,
        default=1.10,
        help="ASE natural_cutoffs multiplier",
    )
    parser.add_argument(
        "--ase_extra_cutoff",
        type=float,
        default=0.00,
        help="extra cutoff added to each atom cutoff in ASE NeighborList",
    )
    parser.add_argument(
        "--ase_skin",
        type=float,
        default=0.00,
        help="ASE NeighborList skin parameter",
    )
    parser.add_argument(
        "--graph_builder",
        type=str,
        default="rdkit",
        choices=["ase", "rdkit"],
        help="graph builder backend used for mol-graph adjustment",
    )
    parser.add_argument(
        "--rdkit_cov_factor",
        type=float,
        default=1.30,
        help="RDKit DetermineConnectivity covalent factor",
    )

    args = parser.parse_args()

    set_seed(args.seed, deterministic_cudnn=False)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Device:", device)

    _, test_files, missing = split_by_year_txt(
        txt_path=args.txt_path,
        pt_dir=args.pt_dir,
        test_years=tuple(args.test_years),
        pt_prefix=args.pt_prefix,
        pt_suffix=args.pt_suffix,
        strict=args.strict,
    )
    print(f"Test files: {len(test_files)} | Missing mapped pt: {len(missing)}")

    test_dataset, max_h_test = build_simple_in_memory_dataset(
        test_files,
        is_check_dist=not args.no_dist_check,
        is_filter=args.is_filter,
    )
    print(f"Test dataset: {len(test_dataset)} | max_h_test={max_h_test}")

    if args.batch_size != 1 and (
        (not args.disable_mol_graph_adjust) or args.dump_test_data or args.print_alignment_check
    ):
        raise ValueError(
            "batch_size must be 1 when using mol-graph adjustment, dump_test_data, or print_alignment_check."
        )

    state = torch.load(args.model_path, map_location="cpu")
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state

    if args.num_classes > 0:
        num_classes = args.num_classes
    else:
        num_classes = infer_num_classes_from_state_dict(state_dict)
    print("num_classes:", num_classes)

    hydro_hidden_channels = 512

    representation_model = TorchMD_ET(
        hidden_channels=hydro_hidden_channels,
        attn_activation="silu",
        num_heads=8,
        distance_influence="both",
    )
    output_model = EquivariantScalar(hydro_hidden_channels, num_classes=num_classes)
    model = TorchMD_Net(representation_model=representation_model, output_model=output_model)
    model.to(device)
    model.load_state_dict(state_dict)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    atom_acc, mol_acc, missing_cnt, dump_ok, dump_fail, adjust_stats = eval_validate(
        model=model,
        test_loader=test_loader,
        device=device,
        dump_test_data=args.dump_test_data,
        atom_analysis=args.atom_analysis,
        refined_dir=args.refined_dir,
        adjust_by_mol_graph=(not args.disable_mol_graph_adjust),
        print_alignment_check=args.print_alignment_check,
        print_alignment_max_cases=args.print_alignment_max_cases,
        graph_builder=args.graph_builder,
        ase_cutoff_mult=args.ase_cutoff_mult,
        ase_extra_cutoff=args.ase_extra_cutoff,
        ase_skin=args.ase_skin,
        rdkit_cov_factor=args.rdkit_cov_factor,
    )

    print(
        f"[{args.model_path}] Atom Acc: {atom_acc * 100:.2f}% | "
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
        if args.print_alignment_check:
            print(
                "[AlignmentCheck] "
                f"checked={adjust_stats['align_checked_cases']} | "
                f"mismatch_cases={adjust_stats['align_mismatch_cases']} | "
                f"len_mismatch_cases={adjust_stats['align_len_mismatch_cases']}"
            )
    if args.dump_test_data:
        print(
            f"[Dump] success: {dump_ok} | fail_missing_inputs: {dump_fail} | "
            f"root: {args.refined_dir}"
        )


if __name__ == "__main__":
    main()
