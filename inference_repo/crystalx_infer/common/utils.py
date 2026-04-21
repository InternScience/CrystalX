"""Backward-compatible re-export layer for legacy inference imports."""

from scipy.spatial.distance import cdist

from crystalx_infer.common.hydrogen import HFIX_HCOUNT, build_hfix_summary, gen_hfix_ins
from crystalx_infer.common.shelx import (
    check_sfac,
    copy_file,
    extract_non_numeric_prefix,
    get_R2,
    get_bond,
    get_equiv_pos2,
    load_shelxt,
    load_shelxt_final,
    read_checkcif,
    update_shelxt,
    update_shelxt_final,
    update_shelxt_hydro,
    update_shelxt_weight,
)

__all__ = [
    "HFIX_HCOUNT",
    "build_hfix_summary",
    "cdist",
    "check_sfac",
    "copy_file",
    "extract_non_numeric_prefix",
    "gen_hfix_ins",
    "get_R2",
    "get_bond",
    "get_equiv_pos2",
    "load_shelxt",
    "load_shelxt_final",
    "read_checkcif",
    "update_shelxt",
    "update_shelxt_final",
    "update_shelxt_hydro",
    "update_shelxt_weight",
]
