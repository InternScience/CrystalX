"""Path helpers for CrystalX inference workflows."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def strip_optional_suffix(name: str, suffix: str) -> str:
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def stem_from_pt_path(pt_path: str) -> str:
    stem = Path(pt_path).stem
    return stem[6:] if stem.startswith("equiv_") else stem


def load_excluded_stems(txt_path: str) -> set[str]:
    stems = set()
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as file_obj:
        for line in file_obj:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            stem = Path(token).stem
            if stem.startswith("equiv_"):
                stem = stem[6:]
            stems.add(stem)
    return stems


def locate_source_case_files(stem: str, search_roots: list[str] | tuple[str, ...]):
    for root in search_roots:
        case_dir = Path(root) / stem
        res_candidate = case_dir / f"{stem}_a.res"
        hkl_candidate = case_dir / f"{stem}_a.hkl"
        if res_candidate.exists() and hkl_candidate.exists():
            return {
                "case_dir": str(case_dir),
                "res": str(res_candidate),
                "hkl": str(hkl_candidate),
            }
    return None


def resolve_refined_case_files(refined_dir: str, fname0: str):
    stem_full = Path(fname0).stem
    stem_short = stem_from_pt_path(fname0)
    candidates = [Path(refined_dir) / stem_full, Path(refined_dir) / stem_short]
    for case_dir in candidates:
        case_name = case_dir.name
        files = {
            "case_dir": str(case_dir),
            "case_name": case_name,
            "hkl": str(case_dir / f"{case_name}_AI.hkl"),
            "ins": str(case_dir / f"{case_name}_AI.ins"),
            "lst": str(case_dir / f"{case_name}_AI.lst"),
            "res": str(case_dir / f"{case_name}_AI.res"),
        }
        if os.path.exists(files["ins"]) and os.path.exists(files["lst"]):
            return files
    return None


@dataclass(frozen=True)
class HeavyPredictionPaths:
    res_input_path: Path
    output_dir: Path
    output_stem: str
    hkl_input_path: Path | None

    @classmethod
    def from_values(
        cls,
        work_dir: str = "",
        fname: str = "",
        res_path: str = "",
        hkl_path: str = "",
    ):
        if res_path:
            res_input_path = Path(res_path)
            output_dir = Path(work_dir) if work_dir else res_input_path.parent
            base_stem = strip_optional_suffix(strip_optional_suffix(res_input_path.stem, "_a"), "_AI")
            output_stem = fname if fname else base_stem
            hkl_input_path = Path(hkl_path) if hkl_path else None
            return cls(
                res_input_path=res_input_path,
                output_dir=output_dir,
                output_stem=output_stem,
                hkl_input_path=hkl_input_path,
            )

        if not fname or not work_dir:
            raise ValueError("Heavy prediction requires --res_path or both --fname and --work_dir.")

        resolved_work_dir = Path(work_dir)
        legacy_hkl_path = resolved_work_dir / f"{fname}_a.hkl"
        return cls(
            res_input_path=resolved_work_dir / f"{fname}_a.res",
            output_dir=resolved_work_dir,
            output_stem=fname,
            hkl_input_path=legacy_hkl_path if legacy_hkl_path.exists() else None,
        )

    @property
    def output_ins_path(self) -> Path:
        return self.output_dir / f"{self.output_stem}_AI.ins"

    @property
    def output_hkl_path(self) -> Path:
        return self.output_dir / f"{self.output_stem}_AI.hkl"

    @property
    def topk_path(self) -> Path:
        return self.output_dir / f"{self.output_stem}_AI_topk.json"


@dataclass(frozen=True)
class BondInputPaths:
    work_dir: Path
    fname: str

    @classmethod
    def from_values(cls, work_dir: str, fname: str):
        return cls(work_dir=Path(work_dir), fname=fname)

    @property
    def source_hkl_path(self) -> Path:
        return self.work_dir / f"{self.fname}.hkl"

    @property
    def source_res_path(self) -> Path:
        return self.work_dir / f"{self.fname}.res"

    @property
    def output_hkl_path(self) -> Path:
        return self.work_dir / f"{self.fname}Bond.hkl"

    @property
    def output_ins_path(self) -> Path:
        return self.work_dir / f"{self.fname}Bond.ins"


@dataclass(frozen=True)
class HydroPredictionPaths:
    res_file_path: Path
    ins_file_path: Path
    output_dir: Path
    output_stem: str
    template_stem: str
    hkl_input_path: Path | None

    @classmethod
    def from_values(
        cls,
        work_dir: str = "",
        fname: str = "",
        res_path: str = "",
        hkl_path: str = "",
    ):
        if res_path:
            res_input_path = Path(res_path)
            output_dir = Path(work_dir) if work_dir else res_input_path.parent
            output_stem = fname if fname else res_input_path.stem
            hkl_input_path = Path(hkl_path) if hkl_path else None
            return cls(
                res_file_path=res_input_path,
                ins_file_path=res_input_path,
                output_dir=output_dir,
                output_stem=output_stem,
                template_stem=res_input_path.stem,
                hkl_input_path=hkl_input_path,
            )

        if not fname or not work_dir:
            raise ValueError("Hydro prediction requires --res_path or both --fname and --work_dir.")

        resolved_work_dir = Path(work_dir)
        legacy_hkl_path = resolved_work_dir / f"{fname}.hkl"
        return cls(
            res_file_path=resolved_work_dir / f"{fname}.res",
            ins_file_path=resolved_work_dir / f"{fname}.ins",
            output_dir=resolved_work_dir,
            output_stem=fname,
            template_stem=fname,
            hkl_input_path=legacy_hkl_path if legacy_hkl_path.exists() else None,
        )

    @property
    def lst_candidates(self) -> list[Path]:
        return [
            self.res_file_path.parent / f"{self.template_stem}Bond.lst",
            self.res_file_path.parent / f"{self.template_stem}.lst",
        ]

    @property
    def output_ins_path(self) -> Path:
        return self.output_dir / f"{self.output_stem}hydro.ins"

    @property
    def output_hkl_path(self) -> Path:
        return self.output_dir / f"{self.output_stem}hydro.hkl"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / f"{self.output_stem}_hydro_pred.json"

    @property
    def topk_path(self) -> Path:
        return self.output_dir / f"{self.output_stem}hydro_topk.json"


@dataclass(frozen=True)
class WeightRefinePaths:
    work_dir: Path
    fname: str

    @classmethod
    def from_values(cls, work_dir: str, fname: str):
        return cls(work_dir=Path(work_dir), fname=fname)

    @property
    def source_hkl_path(self) -> Path:
        return self.work_dir / f"{self.fname}.hkl"

    @property
    def source_res_path(self) -> Path:
        return self.work_dir / f"{self.fname}.res"

    @property
    def fallback_res_path(self) -> Path:
        return self.work_dir / f"{strip_optional_suffix(self.fname, 'hydro')}.res"

    @property
    def output_hkl_path(self) -> Path:
        return self.work_dir / f"{self.fname}Weight.hkl"

    @property
    def output_ins_path(self) -> Path:
        return self.work_dir / f"{self.fname}Weight.ins"


@dataclass(frozen=True)
class FinalOutputPaths:
    work_dir: Path
    fname: str

    @classmethod
    def from_values(cls, work_dir: str, fname: str):
        return cls(work_dir=Path(work_dir), fname=fname)

    @property
    def source_res_path(self) -> Path:
        return self.work_dir / f"{self.fname}.res"

    @property
    def fallback_res_path(self) -> Path:
        return self.work_dir / f"{strip_optional_suffix(self.fname, 'Weight')}.res"

    @property
    def source_hkl_path(self) -> Path:
        return self.work_dir / f"{self.fname}.hkl"

    @property
    def source_cif_path(self) -> Path:
        return self.work_dir / f"{self.fname}.cif"

    @property
    def source_chk_path(self) -> Path:
        return self.work_dir / f"{self.fname}.chk"

    @property
    def final_ins_path(self) -> Path:
        return self.work_dir / f"{self.fname}Final.ins"

    @property
    def final_hkl_path(self) -> Path:
        return self.work_dir / f"{self.fname}Final.hkl"

    @property
    def final_cif_path(self) -> Path:
        return self.work_dir / f"{self.fname}Final.cif"

    @property
    def final_xyz_path(self) -> Path:
        return self.work_dir / f"{self.fname}Final.xyz"

    @property
    def final_gjf_path(self) -> Path:
        return self.work_dir / f"{self.fname}Final.gjf"

    @property
    def final_metrics_path(self) -> Path:
        return self.work_dir / f"{self.fname}FinalMetrics.json"

    @property
    def final_zip_path(self) -> Path:
        return self.work_dir / f"{self.fname}Final.zip"

    @property
    def final_chk_path(self) -> Path:
        return self.work_dir / f"{self.fname}Final_checkcif.chk"
