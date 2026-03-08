import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from crystalx_infer.common.utils import cdist, copy_file, update_shelxt
from crystalx_infer.models.torchmd_et import TorchMD_ET
from crystalx_infer.models.noise_output_model import EquivariantScalar
from crystalx_infer.models.torchmd_net import TorchMD_Net


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int = 42, deterministic_cudnn: bool = False):
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


# -------------------------
# Split
# -------------------------
def split_by_year_txt(
    txt_path: str,
    pt_dir: str,
    test_years=(2022, 2023, 2024),
    pt_prefix="equiv_",
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
                print(f"[WARN] txt第{ln}行格式不对：{line}")
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


# -------------------------
# Dataset build
# -------------------------
def build_simple_in_memory_dataset(file_list, is_eval=True, is_check_dist=True):
    dataset = []
    dist_error_cnt = 0
    noise_cnt = 0
    element_error_cnt = 0
    cnt = 0

    for fname in tqdm(file_list, desc="Build dataset"):
        mol_info = torch.load(fname)

        noise = mol_info["noise_list"]
        if np.max(np.abs(noise)) > 0.1:
            noise_cnt += 1
            continue

        z = mol_info["z"]
        y = mol_info["gt"]
        z = [item.capitalize() for item in z]
        y = [item.capitalize() for item in y]

        try:
            _z = [Chem.Atom(item).GetAtomicNum() for item in z]
            y = [Chem.Atom(item).GetAtomicNum() for item in y]
        except Exception as e:
            print("[Element error]", e, "in", fname)
            element_error_cnt += 1
            continue

        _real_cart = mol_info["pos"]
        real_cart = []
        z_num = []
        for i in range(_real_cart.shape[0]):
            if _real_cart[i].tolist() not in real_cart:
                real_cart.append(_real_cart[i].tolist())
                z_num.append(_z[i])
        real_cart = np.array(real_cart, dtype=np.float32)

        # mask: only evaluate/predict first len(y) atoms
        mask = np.zeros(len(z_num), dtype=np.int64)
        mask[: len(y)] = 1
        mask = torch.from_numpy(mask).bool()

        if is_check_dist:
            distance_matrix = cdist(real_cart, real_cart)
            distance_matrix = distance_matrix + 10 * np.eye(distance_matrix.shape[0])
            if np.min(distance_matrix) < 0.1:
                dist_error_cnt += 1
                continue

        # eval: only one init z (sorted y)
        if is_eval:
            all_z = [sorted(y, reverse=True)]
        else:
            all_z = [sorted(y, reverse=True), z_num[: len(y)]]

        y = torch.tensor(y, dtype=torch.long)
        real_cart = torch.from_numpy(real_cart)

        for pz in all_z:
            pz = list(pz) + z_num[len(y) :]
            pz = torch.tensor(pz, dtype=torch.long)
            dataset.append(Data(z=pz, y=y, pos=real_cart, fname=fname, mask=mask))
        cnt += 1

    print(
        f"[Build] kept={cnt} dist_drop={dist_error_cnt} noise_drop={noise_cnt} element_drop={element_error_cnt}"
    )
    return dataset


@torch.no_grad()
def enforce_coverage_by_prob(prob, sfac, pred):
    """
    prob: [N, K] softmax prob over sfac-subspace
    sfac: [K] unique(gt) atomic numbers
    pred: [N] current prediction (must be in sfac)
    Goal: cover every element in sfac at least once, without making any existing element disappear.
    """
    device = pred.device
    N, K = prob.shape
    if N == 0 or K == 0:
        return pred
    if N < K:
        return pred

    elem2k = {int(sfac[j].item()): j for j in range(K)}
    pred_k = torch.tensor([elem2k[int(x.item())] for x in pred], device=device, dtype=torch.long)

    counts = torch.bincount(pred_k, minlength=K).clone()
    ar = torch.arange(N, device=device)
    cur_prob = prob[ar, pred_k]  # [N]

    used_atoms = set()
    missing = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()

    while missing:
        changed_any = False
        for miss_k in missing:
            safe_mask = counts[pred_k] > 1  # [N]

            if used_atoms:
                used = torch.zeros(N, device=device, dtype=torch.bool)
                used[list(used_atoms)] = True
                safe_mask = safe_mask & (~used)

            if not torch.any(safe_mask):
                return pred

            target_prob = prob[:, miss_k]  # [N]
            cost = (-target_prob).masked_fill(~safe_mask, float("inf"))
            i = int(torch.argmin(cost).item())

            old_k = int(pred_k[i].item())
            if old_k == miss_k:
                continue

            pred[i] = sfac[miss_k]
            pred_k[i] = miss_k
            used_atoms.add(i)

            counts[old_k] -= 1
            counts[miss_k] += 1
            cur_prob[i] = target_prob[i]
            changed_any = True

        if not changed_any:
            break
        missing = (counts == 0).nonzero(as_tuple=False).view(-1).tolist()

    return pred


# -------------------------
# Two-pass inference (+ dump_wrong / dump_correct + update_shelxt失败统计)
# -------------------------
@torch.no_grad()
def infer_two_pass(
    model,
    loader,
    device,
    restrict_to_sfac=True,
    allow_one_mismatch=False,
    dump_wrong=False,
    dump_dir="infer_bad",
    dump_correct=False,
    dump_correct_dir="infer_good",
    search_roots=None,
):
    model.eval()

    all_correct_atom = 0
    total_atom = 0
    correct_mol = 0
    total_mol = 0

    dumped_wrong = 0
    dumped_correct = 0

    # update_shelxt 失败统计
    update_fail_stems = []   # 保留顺序的 list（去重后加入）
    update_fail_set = set()  # 用于去重
    update_fail_cnt = 0      # 失败发生次数（可能同一个 stem 多次）

    if dump_wrong:
        os.makedirs(dump_dir, exist_ok=True)
    if dump_correct:
        os.makedirs(dump_correct_dir, exist_ok=True)

    def _dump_case(data, pred2, out_root_dir, tag="AI", search_roots=None):
        """
        注意：这份脚本默认 batch_size=1 更合适（因为这里按单个case去找文件/写ins）。
        如果你一定要 batch_size>1，建议把 dump 逻辑改成对 data.to_data_list() 逐个处理。
        """
        nonlocal update_fail_cnt, update_fail_stems, update_fail_set

        # 预测元素（mask范围内）
        pred_sym = [Chem.Atom(int(x.item())).GetSymbol() for x in pred2.detach().cpu()]

        # fname 字段：通常 batch_size=1 时是 list/tuple，取第一个
        fname0 = data.fname[0] if isinstance(data.fname, (list, tuple)) else data.fname
        fname = os.path.basename(fname0)
        stem = os.path.splitext(fname)[0]          # e.g. equiv_7248215
        stem2 = stem[6:] if stem.startswith("equiv_") else stem  # 7248215

        # 找原始 res/hkl
        res_file_path, hkl_file_path = None, None
        if search_roots is None:
            search_roots = [
                "all_xrd_dataset_test",
                "all_xrd_dataset_test_2",
                "all_xrd_dataset_test_3",
                "all_xrd_dataset_test_4",
            ]

        for root in search_roots:
            rp = os.path.join(root, stem2, f"{stem2}_a.res")
            hp = os.path.join(root, stem2, f"{stem2}_a.hkl")
            if os.path.exists(rp) and os.path.exists(hp):
                res_file_path, hkl_file_path = rp, hp
                break

        out_case_dir = os.path.join(out_root_dir, stem2)
        os.makedirs(out_case_dir, exist_ok=True)

        # copy cif
        cif_root = args.cif_root
        cif_path = os.path.join(cif_root, f"{stem2}.cif")
        if os.path.exists(cif_path):
            copy_file(cif_path, os.path.join(out_case_dir, f"{stem2}.cif"))
        else:
            print(f"[WARN] cif not found: {cif_path}")

        # copy hkl/res + write ins
        if hkl_file_path:
            copy_file(hkl_file_path, os.path.join(out_case_dir, f"{stem2}_{tag}.hkl"))

        if res_file_path:
            copy_file(res_file_path, os.path.join(out_case_dir, f"{stem2}_a.res"))

            out_ins = os.path.join(out_case_dir, f"{stem2}_{tag}.ins")
            try:
                update_shelxt(res_file_path, out_ins, pred_sym)
            except Exception as e:
                # 记录失败 stem
                update_fail_cnt += 1
                if stem2 not in update_fail_set:
                    update_fail_set.add(stem2)
                    update_fail_stems.append(stem2)

                print("[update_shelxt ERROR]", e)
                print("[update_shelxt RES]", res_file_path)

        # 保存 pt 便于debug
        data_cpu = data.to("cpu")
        fname_save = fname0 if isinstance(fname0, str) else str(fname0)
        torch.save(
            Data(z=pred_sym, y=data_cpu.y, pos=data_cpu.pos, fname=fname_save),
            os.path.join(out_case_dir, f"mol_{stem2}.pt"),
        )

    for data in tqdm(loader, desc="Infer"):
        data = data.to(device)

        # -----------------
        # Pass 1
        # -----------------
        outputs, _ = model(data.z, data.pos, data.batch)
        logits = outputs

        if restrict_to_sfac:
            sfac = torch.unique(data.y)
            logits_ = logits[:, sfac]
            prob = F.softmax(logits_, dim=-1)
            pred_idx = torch.argmax(prob, dim=1)
            predicted = sfac[pred_idx]
        else:
            prob = F.softmax(logits, dim=-1)
            predicted = torch.argmax(prob, dim=1)

        # -----------------
        # Pass 2 (mask)
        # -----------------
        outputs2, _ = model(predicted, data.pos, data.batch)
        outputs2 = outputs2[data.mask]
        logits2 = outputs2

        if restrict_to_sfac:
            sfac = torch.unique(data.y)
            logits2_ = logits2[:, sfac]
            prob2 = F.softmax(logits2_, dim=-1)
            pred_idx2 = torch.argmax(prob2, dim=1)
            pred2 = sfac[pred_idx2]

            # 强制覆盖：保证每个 sfac 元素至少出现一次
            pred2 = enforce_coverage_by_prob(prob2, sfac, pred2)
        else:
            prob2 = F.softmax(logits2, dim=-1)
            pred2 = torch.argmax(prob2, dim=1)

        label = data.y
        correct_atom = (pred2 == label).sum().item()
        mol_atom = label.shape[0]

        all_correct_atom += correct_atom
        total_atom += mol_atom

        ok = (correct_atom == mol_atom) or (
            allow_one_mismatch and correct_atom == mol_atom - 1
        )

        if ok:
            correct_mol += 1
            if dump_correct:
                _dump_case(data, pred2, dump_correct_dir, tag="AI", search_roots=search_roots)
                dumped_correct += 1
        else:
            if dump_wrong:
                _dump_case(data, pred2, dump_dir, tag="AI", search_roots=search_roots)
                dumped_wrong += 1

        total_mol += 1

    atom_acc = all_correct_atom / max(total_atom, 1)
    mol_acc = correct_mol / max(total_mol, 1)

    if dump_wrong:
        print(f"[Dump] wrong dumped: {dumped_wrong} -> {dump_dir}")
    if dump_correct:
        print(f"[Dump] correct dumped: {dumped_correct} -> {dump_correct_dir}")

    # 打印 update_shelxt 失败统计
    print(
        f"[update_shelxt] fail events: {update_fail_cnt} | unique stems: {len(update_fail_stems)}"
    )

    # 可选：把失败 stem 列表写到文件（只要开启了任意 dump，就写到一个根目录里）
    if (dump_wrong or dump_correct) and len(update_fail_stems) > 0:
        root_dir = dump_dir if dump_wrong else dump_correct_dir
        out_txt = os.path.join(root_dir, "update_shelxt_failed_stems.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            for s in update_fail_stems:
                f.write(f"{s}\n")
        print("[update_shelxt] failed stem list saved to:", out_txt)

    return atom_acc, mol_acc


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument(
        "--test_years",
        type=int,
        nargs="+",
        default=[2019, 2020, 2021, 2022, 2023, 2024],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=150)

    parser.add_argument(
        "--restrict_to_sfac",
        action="store_true",
        help="match original eval口径（只在unique(gt)类别里选）",
    )
    parser.add_argument(
        "--allow_one_mismatch",
        action="store_true",
        help="mol允许错1个也算对（也影响dump_correct判定）",
    )

    # dump wrong (existing)
    parser.add_argument("--dump_wrong", action="store_true")
    parser.add_argument("--dump_dir", type=str, default=f"infer_bad_{get_run_timestamp()}")

    # dump correct (new)
    parser.add_argument("--dump_correct", action="store_true")
    parser.add_argument("--dump_correct_dir", type=str, default=f"infer_good_{get_run_timestamp()}")

    parser.add_argument("--search_roots", type=str, nargs="*", default=["all_test_shelx"])
    parser.add_argument("--cif_root", type=str, default="all_cif")

    args = parser.parse_args()

    set_seed(args.seed, deterministic_cudnn=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # split: only need test set
    _, test_files, missing = split_by_year_txt(
        txt_path=args.txt_path,
        pt_dir=args.pt_dir,
        test_years=tuple(args.test_years),
        pt_prefix="equiv_",
        pt_suffix=".pt",
        strict=False,
    )
    print(f"Test files: {len(test_files)} | Missing mapped pt: {len(missing)}")

    # dataset/loader
    test_dataset = build_simple_in_memory_dataset(test_files, is_eval=True, is_check_dist=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # build model
    rep = TorchMD_ET(attn_activation="silu", num_heads=8, distance_influence="both")
    out = EquivariantScalar(256, num_classes=98)
    model = TorchMD_Net(representation_model=rep, output_model=out).to(device)

    # load weights
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # infer
    atom_acc, mol_acc = infer_two_pass(
        model=model,
        loader=test_loader,
        device=device,
        restrict_to_sfac=args.restrict_to_sfac,
        allow_one_mismatch=args.allow_one_mismatch,
        dump_wrong=args.dump_wrong,
        dump_dir=args.dump_dir,
        dump_correct=args.dump_correct,
        dump_correct_dir=args.dump_correct_dir,
        search_roots=args.search_roots,
    )

    print(f"[{args.model_path}] Atom Acc: {atom_acc*100:.2f}% | Mol Acc: {mol_acc*100:.2f}%")
    if args.dump_wrong:
        print("Dumped wrong cases to:", args.dump_dir)
    if args.dump_correct:
        print("Dumped correct cases to:", args.dump_correct_dir)


if __name__ == "__main__":
    main()
