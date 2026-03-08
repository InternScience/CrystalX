import argparse
import datetime as dt
import json
import os
import re
from collections import Counter


INSTRUCTION_KEYWORDS = {
    "TITL",
    "REM",
    "CELL",
    "ZERR",
    "LATT",
    "SYMM",
    "SFAC",
    "UNIT",
    "L.S.",
    "BOND",
    "CONF",
    "HTAB",
    "LIST",
    "FMAP",
    "PLAN",
    "ACTA",
    "WGHT",
    "FVAR",
    "AFIX",
    "HKLF",
    "END",
    "EQIV",
}


class ResultFileNotFoundError(FileNotFoundError):
    pass


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def parse_initial_ins(ins_path: str):
    sfac = None
    unit = None
    with open(ins_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            up = s.upper()
            if up.startswith("SFAC "):
                sfac = [x.capitalize() for x in s.split()[1:]]
            elif up.startswith("UNIT "):
                vals = s.split()[1:]
                unit = [int(float(v)) for v in vals]
    if not sfac:
        raise ValueError(f"No SFAC found in {ins_path}")
    if not unit:
        raise ValueError(f"No UNIT found in {ins_path}")
    if len(sfac) != len(unit):
        raise ValueError(
            f"SFAC/UNIT length mismatch in {ins_path}: {len(sfac)} vs {len(unit)}"
        )
    return sfac, unit


def parse_aihydroweight_atoms(ai_path: str):
    sfac = None
    counts = Counter()

    with open(ai_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            head = parts[0].upper()

            if head == "SFAC":
                sfac = [x.capitalize() for x in parts[1:]]
                continue

            if head == "HKLF":
                break

            if sfac is None:
                continue

            if head in INSTRUCTION_KEYWORDS:
                continue
            if len(parts) < 5:
                continue
            if not (_is_float(parts[2]) and _is_float(parts[3]) and _is_float(parts[4])):
                continue

            try:
                sfac_idx = int(float(parts[1]))
            except Exception:
                continue

            if sfac_idx < 1 or sfac_idx > len(sfac):
                continue

            elem = sfac[sfac_idx - 1]
            counts[elem] += 1

    if sfac is None:
        raise ValueError(f"No SFAC found in {ai_path}")
    return sfac, counts


def ratio_match_exact(base_counts, ai_counts):
    if len(base_counts) != len(ai_counts):
        return False

    for b, a in zip(base_counts, ai_counts):
        if (b == 0) != (a == 0):
            return False

    nz_idx = [i for i, b in enumerate(base_counts) if b > 0]
    if not nz_idx:
        return sum(ai_counts) == 0

    ref = nz_idx[0]
    b_ref = base_counts[ref]
    a_ref = ai_counts[ref]
    for i in nz_idx[1:]:
        if base_counts[i] * a_ref != ai_counts[i] * b_ref:
            return False
    return True


def _parse_timestamp_dir_name(dir_name: str, case_name: str):
    pattern = r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_(.+)$"
    m = re.match(pattern, dir_name)
    if not m:
        return None
    date_part, time_part, tail = m.group(1), m.group(2), m.group(3)
    if tail != case_name:
        return None
    ts = f"{date_part}_{time_part}"
    try:
        return dt.datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return None


def _find_timestamp_run_dirs(case_dir: str, case_name: str):
    run_dirs = []
    for d in os.listdir(case_dir):
        rd = os.path.join(case_dir, d)
        if not os.path.isdir(rd):
            continue
        ts = _parse_timestamp_dir_name(d, case_name)
        if ts is None:
            continue
        run_dirs.append((ts, rd))
    run_dirs.sort(key=lambda x: x[0], reverse=True)
    return run_dirs


def _pick_result_file_in_run_dir(run_dir: str, case_name: str):
    cands = [
        os.path.join(run_dir, f"{case_name}_AIhydroWeight.ins"),
        os.path.join(run_dir, f"{case_name}_AIhydroWeight.res"),
        os.path.join(run_dir, f"{case_name}_AIhydro.res"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return ""


def _pick_pred_summary_in_run_dir(run_dir: str, case_name: str):
    cands = [
        os.path.join(run_dir, f"{case_name}_AI_hydro_pred.json"),
        os.path.join(run_dir, f"{case_name}_AIhydro_pred.json"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return ""


def _load_pred_h_total(pred_summary_path: str):
    with open(pred_summary_path, "r", encoding="utf-8", errors="ignore") as f:
        obj = json.load(f)
    if "pred_h_total" in obj:
        return int(obj["pred_h_total"])
    raise ValueError(f"pred_h_total not found in {pred_summary_path}")


def find_case_files(
    case_dir: str,
    origin_ins: str = "",
    ai_file: str = "",
    any_timestamp_pass: bool = False,
):
    case_dir = os.path.abspath(case_dir)
    case_name = os.path.basename(case_dir.rstrip("\\/"))

    if origin_ins:
        origin_ins_path = os.path.abspath(origin_ins)
    else:
        default_ins = os.path.join(case_dir, f"{case_name}.ins")
        if os.path.isfile(default_ins):
            origin_ins_path = default_ins
        else:
            root_ins = [
                os.path.join(case_dir, x)
                for x in os.listdir(case_dir)
                if x.lower().endswith(".ins") and os.path.isfile(os.path.join(case_dir, x))
            ]
            if not root_ins:
                raise FileNotFoundError(f"No initial .ins found under {case_dir}")
            origin_ins_path = sorted(root_ins)[0]

    if ai_file:
        ai_path = os.path.abspath(ai_file)
        ai_dir = os.path.dirname(ai_path)
        pred_path = _pick_pred_summary_in_run_dir(ai_dir, case_name)
        file_items = [{"ai_file": ai_path, "pred_summary": pred_path}]
    else:
        run_dirs = _find_timestamp_run_dirs(case_dir, case_name)
        if not run_dirs:
            raise FileNotFoundError(
                f"No timestamp run subdirectory found under {case_dir} "
                f"(expected: YYYY-MM-DD_HH-MM-SS_{case_name})"
            )

        if any_timestamp_pass:
            file_items = []
            for _, rd in run_dirs:
                p = _pick_result_file_in_run_dir(rd, case_name)
                if p:
                    file_items.append(
                        {
                            "ai_file": p,
                            "pred_summary": _pick_pred_summary_in_run_dir(rd, case_name),
                        }
                    )
            if not file_items:
                raise ResultFileNotFoundError(
                    f"No {case_name}_AIhydroWeight.ins/.res or {case_name}_AIhydro.res found "
                    f"in any timestamp run dir under: {case_dir}"
                )
        else:
            latest_run_dir = run_dirs[0][1]
            ai_file_path = _pick_result_file_in_run_dir(latest_run_dir, case_name)
            if not ai_file_path:
                raise ResultFileNotFoundError(
                    f"No {case_name}_AIhydroWeight.ins/.res or {case_name}_AIhydro.res found "
                    f"in latest run dir: {latest_run_dir}"
                )
            file_items = [
                {
                    "ai_file": ai_file_path,
                    "pred_summary": _pick_pred_summary_in_run_dir(latest_run_dir, case_name),
                }
            ]

    if not os.path.isfile(origin_ins_path):
        raise FileNotFoundError(f"Initial ins not found: {origin_ins_path}")
    for it in file_items:
        p = it["ai_file"]
        if not os.path.isfile(p):
            raise FileNotFoundError(f"result file not found: {p}")
    return origin_ins_path, file_items


def _build_ratio_metrics(
    base_sfac,
    base_unit,
    ai_sfac,
    ai_counter,
    allow_ratio_tol: float,
    ignore_h: bool,
):
    all_elems = list(base_sfac)
    for e in ai_sfac:
        if e not in all_elems:
            all_elems.append(e)

    if ignore_h:
        elems = [e for e in all_elems if e.upper() != "H"]
    else:
        elems = list(all_elems)

    base_counts = [base_unit[base_sfac.index(e)] if e in base_sfac else 0 for e in elems]
    ai_counts = [int(ai_counter.get(e, 0)) for e in elems]

    base_total = sum(base_counts)
    ai_total = sum(ai_counts)
    base_ratio = [x / base_total if base_total > 0 else 0.0 for x in base_counts]
    ai_ratio = [x / ai_total if ai_total > 0 else 0.0 for x in ai_counts]
    ratio_diff = [abs(a - b) for a, b in zip(ai_ratio, base_ratio)]
    max_diff = max(ratio_diff) if ratio_diff else 0.0

    exact_match = ratio_match_exact(base_counts, ai_counts)
    tol_match = max_diff <= float(allow_ratio_tol)
    is_match = exact_match if allow_ratio_tol == 0 else tol_match

    return {
        "elems": elems,
        "base_counts": base_counts,
        "ai_counts": ai_counts,
        "base_total": base_total,
        "ai_total": ai_total,
        "base_ratio": base_ratio,
        "ai_ratio": ai_ratio,
        "ratio_diff": ratio_diff,
        "max_diff": max_diff,
        "exact_match": exact_match,
        "tol_match": tol_match,
        "is_match": is_match,
    }


def analyze_case(
    case_dir: str,
    origin_ins: str = "",
    ai_file: str = "",
    allow_ratio_tol: float = 0.0,
    any_timestamp_pass: bool = False,
):
    origin_ins, file_items = find_case_files(
        case_dir=case_dir,
        origin_ins=origin_ins,
        ai_file=ai_file,
        any_timestamp_pass=any_timestamp_pass,
    )
    base_sfac, base_unit = parse_initial_ins(origin_ins)
    run_results = []
    for it in file_items:
        ai_file = it["ai_file"]
        pred_summary = it.get("pred_summary", "")
        ai_sfac, ai_counter = parse_aihydroweight_atoms(ai_file)
        final_h_total = int(ai_counter.get("H", 0))

        pred_h_total = None
        pred_error = ""
        if pred_summary:
            try:
                pred_h_total = _load_pred_h_total(pred_summary)
            except Exception as e:
                pred_error = str(e)

        # WITH_H now prefers hydrogen count from hydro prediction stage.
        ai_counter_with_h = Counter(ai_counter)
        if pred_h_total is not None:
            ai_counter_with_h["H"] = int(pred_h_total)

        with_h = _build_ratio_metrics(
            base_sfac=base_sfac,
            base_unit=base_unit,
            ai_sfac=ai_sfac,
            ai_counter=ai_counter_with_h,
            allow_ratio_tol=allow_ratio_tol,
            ignore_h=False,
        )
        no_h = _build_ratio_metrics(
            base_sfac=base_sfac,
            base_unit=base_unit,
            ai_sfac=ai_sfac,
            ai_counter=ai_counter,
            allow_ratio_tol=allow_ratio_tol,
            ignore_h=True,
        )
        run_results.append(
            {
                "ai_file": ai_file,
                "pred_summary": pred_summary,
                "pred_h_total": pred_h_total,
                "pred_error": pred_error,
                "final_h_total": final_h_total,
                "h_mismatch": (pred_h_total is not None and final_h_total != int(pred_h_total)),
                "with_h": with_h,
                "no_h": no_h,
            }
        )

    mismatch_runs = []
    for r in run_results:
        if r["h_mismatch"]:
            mismatch_runs.append(
                {
                    "ai_file": r["ai_file"],
                    "pred_summary": r["pred_summary"],
                    "pred_h_total": r["pred_h_total"],
                    "final_h_total": r["final_h_total"],
                }
            )

    # default display base: latest candidate (first in list)
    display = run_results[0]
    if any_timestamp_pass and len(run_results) > 1:
        # for "any pass" mode, choose a passing run for display when available.
        for r in run_results:
            if r["with_h"]["is_match"]:
                display_with_h_run = r
                break
        else:
            display_with_h_run = display

        for r in run_results:
            if r["no_h"]["is_match"]:
                display_no_h_run = r
                break
        else:
            display_no_h_run = display

        with_h_final = dict(display_with_h_run["with_h"])
        with_h_final["is_match"] = any(r["with_h"]["is_match"] for r in run_results)
        with_h_final["source_ai_file"] = display_with_h_run["ai_file"]
        with_h_final["pred_h_total"] = display_with_h_run["pred_h_total"]
        with_h_final["final_h_total"] = display_with_h_run["final_h_total"]
        with_h_final["h_mismatch"] = display_with_h_run["h_mismatch"]

        no_h_final = dict(display_no_h_run["no_h"])
        no_h_final["is_match"] = any(r["no_h"]["is_match"] for r in run_results)
        no_h_final["source_ai_file"] = display_no_h_run["ai_file"]
        no_h_final["pred_h_total"] = display_no_h_run["pred_h_total"]
        no_h_final["final_h_total"] = display_no_h_run["final_h_total"]
        no_h_final["h_mismatch"] = display_no_h_run["h_mismatch"]
    else:
        with_h_final = dict(display["with_h"])
        with_h_final["source_ai_file"] = display["ai_file"]
        with_h_final["pred_h_total"] = display["pred_h_total"]
        with_h_final["final_h_total"] = display["final_h_total"]
        with_h_final["h_mismatch"] = display["h_mismatch"]
        no_h_final = dict(display["no_h"])
        no_h_final["source_ai_file"] = display["ai_file"]
        no_h_final["pred_h_total"] = display["pred_h_total"]
        no_h_final["final_h_total"] = display["final_h_total"]
        no_h_final["h_mismatch"] = display["h_mismatch"]

    return {
        "case_dir": os.path.abspath(case_dir),
        "case_name": os.path.basename(os.path.abspath(case_dir).rstrip("\\/")),
        "origin_ins": origin_ins,
        "ai_file": display["ai_file"],
        "ai_files_checked": [r["ai_file"] for r in run_results],
        "mismatch_runs": mismatch_runs,
        "h_stats": {
            "runs_total": len(run_results),
            "runs_with_pred": sum(1 for r in run_results if r["pred_h_total"] is not None),
            "runs_mismatch": sum(1 for r in run_results if r["h_mismatch"]),
            "any_mismatch": any(r["h_mismatch"] for r in run_results),
        },
        "with_h": with_h_final,
        "no_h": no_h_final,
    }


def print_case_report(ret: dict, allow_ratio_tol: float):
    def _print_metrics(title: str, m: dict):
        print(f"[{title}]")
        print("Source result:", m.get("source_ai_file", ret["ai_file"]))
        if m.get("pred_h_total") is not None:
            print(
                f"Hydrogen compare (pred vs final): {m['pred_h_total']} vs {m['final_h_total']} "
                f"| mismatch={m.get('h_mismatch')}"
            )
        else:
            print("Hydrogen compare (pred vs final): pred summary not found")
        print("Element order:", " ".join(m["elems"]))
        print(
            "Initial UNIT counts:",
            " ".join(str(x) for x in m["base_counts"]),
            f"(total={m['base_total']})",
        )
        print(
            "AI counts:",
            " ".join(str(x) for x in m["ai_counts"]),
            f"(total={m['ai_total']})",
        )
        print("Ratios (Initial -> AI):")
        for e, br, ar, d in zip(m["elems"], m["base_ratio"], m["ai_ratio"], m["ratio_diff"]):
            print(f"  {e}: {br:.6f} -> {ar:.6f}  (diff={d:.6f})")
        print("Exact ratio match:", "YES" if m["exact_match"] else "NO")
        if allow_ratio_tol != 0:
            print(
                f"Tolerance ratio match (tol={allow_ratio_tol}):",
                "YES" if m["tol_match"] else "NO",
            )
        print("Final verdict:", "CONSISTENT" if m["is_match"] else "INCONSISTENT")
        print()

    print("Initial INS:", ret["origin_ins"])
    if len(ret.get("ai_files_checked", [])) > 1:
        print("Result files checked:", len(ret["ai_files_checked"]))
    else:
        print("Result file:", ret["ai_file"])
    hs = ret.get("h_stats", {})
    if hs:
        print(
            f"Hydrogen mismatch stats (runs): {hs.get('runs_mismatch', 0)}/"
            f"{hs.get('runs_with_pred', 0)} with prediction summary"
        )
    print()
    _print_metrics("WITH_H", ret["with_h"])
    _print_metrics("NO_H", ret["no_h"])


def main():
    parser = argparse.ArgumentParser(
        description="Compare AIhydroWeight atom ratio vs initial INS SFAC/UNIT ratio."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--case_dir",
        type=str,
        help="Single case directory, e.g. inference_code/2021244",
    )
    group.add_argument(
        "--root_dir",
        type=str,
        help="Batch root directory: iterate all immediate subdirectories as cases",
    )
    parser.add_argument("--origin_ins", type=str, default="", help="Optional explicit initial .ins path")
    parser.add_argument(
        "--ai_file",
        type=str,
        default="",
        help="Optional explicit AIhydroWeight .ins/.res path",
    )
    parser.add_argument(
        "--allow_ratio_tol",
        type=float,
        default=0.0,
        help="Allowed max absolute ratio diff (default 0 for strict exact ratio)",
    )
    parser.add_argument(
        "--any_timestamp_pass",
        action="store_true",
        help="Check all timestamp run dirs; if any one is consistent, mark the case as consistent.",
    )
    parser.add_argument(
        "--mismatch_txt",
        type=str,
        default="",
        help="Optional output txt path for pred_h != final_h cases in batch mode "
             "(default: <root_dir>/pred_h_mismatch_cases.txt).",
    )
    parser.add_argument(
        "--with_h_consistent_txt",
        type=str,
        default="",
        help="Optional output txt path for cases where with_h is consistent in batch mode "
             "(default: <root_dir>/with_h_consistent_cases.txt).",
    )
    args = parser.parse_args()

    if args.case_dir:
        ret = analyze_case(
            case_dir=args.case_dir,
            origin_ins=args.origin_ins,
            ai_file=args.ai_file,
            allow_ratio_tol=args.allow_ratio_tol,
            any_timestamp_pass=args.any_timestamp_pass,
        )
        print_case_report(ret, allow_ratio_tol=args.allow_ratio_tol)
        return

    if args.origin_ins or args.ai_file:
        raise ValueError("--origin_ins/--ai_file only support single-case mode (--case_dir).")

    root_dir = os.path.abspath(args.root_dir)
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"root_dir not found: {root_dir}")

    case_dirs = [
        os.path.join(root_dir, d)
        for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    if not case_dirs:
        raise FileNotFoundError(f"No subdirectories under root_dir: {root_dir}")

    total = 0
    consistent_with_h = 0
    inconsistent_with_h = 0
    consistent_no_h = 0
    inconsistent_no_h = 0
    pred_h_cases = 0
    pred_h_mismatch_cases = 0
    mismatch_rows = []
    with_h_consistent_rows = []
    skipped = 0
    errors = 0

    print("Batch root:", root_dir)
    print()

    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir.rstrip("\\/"))
        total += 1

        # Skip folders that are not case dirs (no root-level .ins).
        case_ins = os.path.join(case_dir, f"{case_name}.ins")
        has_any_ins = any(
            fn.lower().endswith(".ins")
            for fn in os.listdir(case_dir)
            if os.path.isfile(os.path.join(case_dir, fn))
        )
        if (not os.path.isfile(case_ins)) and (not has_any_ins):
            print(f"[SKIP] {case_name} (no initial .ins in case root)")
            skipped += 1
            continue

        try:
            ret = analyze_case(
                case_dir=case_dir,
                allow_ratio_tol=args.allow_ratio_tol,
                any_timestamp_pass=args.any_timestamp_pass,
            )
            if ret["with_h"]["is_match"]:
                consistent_with_h += 1
                verdict_with_h = "CONSISTENT"
            else:
                inconsistent_with_h += 1
                verdict_with_h = "INCONSISTENT"

            if ret["no_h"]["is_match"]:
                consistent_no_h += 1
                verdict_no_h = "CONSISTENT"
            else:
                inconsistent_no_h += 1
                verdict_no_h = "INCONSISTENT"

            if ret["with_h"]["is_match"]:
                with_h_consistent_rows.append(
                    {
                        "case_name": ret["case_name"],
                        "with_h": verdict_with_h,
                        "no_h": verdict_no_h,
                        "max_diff_with_h": ret["with_h"]["max_diff"],
                        "max_diff_no_h": ret["no_h"]["max_diff"],
                        "pred_h_total": ret["with_h"].get("pred_h_total"),
                        "final_h_total": ret["with_h"].get("final_h_total"),
                        "h_mismatch": ret["with_h"].get("h_mismatch"),
                        "ai_file": ret["with_h"].get("source_ai_file", ret["ai_file"]),
                    }
                )

            h_stats = ret.get("h_stats", {})
            if h_stats.get("runs_with_pred", 0) > 0:
                pred_h_cases += 1
                if h_stats.get("any_mismatch", False):
                    pred_h_mismatch_cases += 1
                    for item in ret.get("mismatch_runs", []):
                        mismatch_rows.append(
                            {
                                "case_name": ret["case_name"],
                                "pred_h_total": item.get("pred_h_total"),
                                "final_h_total": item.get("final_h_total"),
                                "ai_file": item.get("ai_file", ""),
                                "pred_summary": item.get("pred_summary", ""),
                            }
                        )

            print(
                f"[DONE] {case_name}  with_h={verdict_with_h}  no_h={verdict_no_h}  "
                f"max_diff_with_h={ret['with_h']['max_diff']:.6f}  "
                f"max_diff_no_h={ret['no_h']['max_diff']:.6f}  "
                f"h_mismatch_runs={h_stats.get('runs_mismatch', 0)}/{h_stats.get('runs_with_pred', 0)}  "
                f"ai={ret['ai_file']}"
            )
        except ResultFileNotFoundError as e:
            skipped += 1
            print(f"[SKIP] {case_name} ({e})")
        except Exception as e:
            errors += 1
            print(f"[ERR ] {case_name}  {e}")

    print()
    print("Batch summary:")
    print(f"  total        : {total}")
    print(f"  with_h consistent   : {consistent_with_h}")
    print(f"  with_h inconsistent : {inconsistent_with_h}")
    print(f"  no_h consistent     : {consistent_no_h}")
    print(f"  no_h inconsistent   : {inconsistent_no_h}")
    print(f"  cases with pred_h summary : {pred_h_cases}")
    print(f"  cases pred_h != final_h   : {pred_h_mismatch_cases}")
    print(f"  skipped      : {skipped}")
    print(f"  errors       : {errors}")

    mismatch_txt = args.mismatch_txt.strip()
    if not mismatch_txt:
        mismatch_txt = os.path.join(root_dir, "pred_h_mismatch_cases.txt")
    with open(mismatch_txt, "w", encoding="utf-8") as f:
        f.write("case_name\tpred_h_total\tfinal_h_total\tai_file\tpred_summary\n")
        for row in mismatch_rows:
            f.write(
                f"{row['case_name']}\t{row['pred_h_total']}\t{row['final_h_total']}\t"
                f"{row['ai_file']}\t{row['pred_summary']}\n"
            )
    print(f"  mismatch txt : {os.path.abspath(mismatch_txt)}")

    with_h_consistent_txt = args.with_h_consistent_txt.strip()
    if not with_h_consistent_txt:
        with_h_consistent_txt = os.path.join(root_dir, "with_h_consistent_cases.txt")
    with open(with_h_consistent_txt, "w", encoding="utf-8") as f:
        f.write(
            "case_name\twith_h\tno_h\tmax_diff_with_h\tmax_diff_no_h\t"
            "pred_h_total\tfinal_h_total\th_mismatch\tai_file\n"
        )
        for row in with_h_consistent_rows:
            f.write(
                f"{row['case_name']}\t{row['with_h']}\t{row['no_h']}\t"
                f"{row['max_diff_with_h']:.6f}\t{row['max_diff_no_h']:.6f}\t"
                f"{row['pred_h_total']}\t{row['final_h_total']}\t{row['h_mismatch']}\t"
                f"{row['ai_file']}\n"
            )
    print(f"  with_h consistent txt : {os.path.abspath(with_h_consistent_txt)}")


if __name__ == "__main__":
    main()
