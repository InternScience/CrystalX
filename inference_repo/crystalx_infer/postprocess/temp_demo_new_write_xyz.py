from iotbx.data_manager import DataManager    # Load in the DataManager
from iotbx.shelx import crystal_symmetry_from_ins
from cctbx.xray.structure import structure
from collections import Counter
import os
from crystalx_infer.common.utils import *
import numpy as np
import torch
import argparse
import json
import zipfile
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='adjust ShelxL weight')
    parser.add_argument('--fname', type=str, default='sample2_AIhydroWeight', help='file name')
    parser.add_argument('--work_dir', type=str, default='work_dir', help='work directory')
    args = parser.parse_args()

    fname = args.fname
    work_dir = args.work_dir

    fname = args.fname
    qpeak_res_name = f'{work_dir}/{fname}.res'
    hkl_file_path = f'{work_dir}/{fname}.hkl'
    cif_file_path = f'{work_dir}/{fname}.cif'
    chk_file_path = f'{work_dir}/{fname}.chk'
    final_qpeak_res_name = f'{work_dir}/{fname}Final.ins'
    new_hkl_file_path = f'{work_dir}/{fname}Final.hkl'
    new_cif_file_path = f'{work_dir}/{fname}Final.cif'
    xyz_save_path = f'{work_dir}/{fname}Final.xyz'
    gjf_save_path = f'{work_dir}/{fname}Final.gjf'
    metrics_save_path = f'{work_dir}/{fname}FinalMetrics.json'
    zip_save_path = f'{work_dir}/{fname}Final.zip'
    new_chk_file_path = f'{work_dir}/{fname}Final_checkcif.chk'

    if os.path.getsize(qpeak_res_name) == 0:
        qpeak_res_name = f'{work_dir}/{fname[:-6]}.res'
        is_structure_valid = False
        is_quality_valid = False
        zip_lst = [final_qpeak_res_name, new_hkl_file_path, xyz_save_path, gjf_save_path]
    else:
        os.rename(chk_file_path, new_chk_file_path)
        is_structure_valid, is_quality_valid = read_checkcif(new_chk_file_path)
        zip_lst = [final_qpeak_res_name, new_hkl_file_path, new_cif_file_path, xyz_save_path, gjf_save_path, new_chk_file_path]

    real_frac, shelxt_pred = load_shelxt_final(qpeak_res_name, begin_flag = 'FVAR')
    shelxt_pred = [item.capitalize() for item in shelxt_pred]
    print(shelxt_pred)

    atom_num = len(shelxt_pred)

    sfac = dict(Counter(shelxt_pred))
    mol_formular = ''
    for k, v in sfac.items():
        e = k+str(v)
        mol_formular += e

    update_shelxt_final(qpeak_res_name, final_qpeak_res_name, sfac, no_given_sfac = True)
    copy_file(hkl_file_path, new_hkl_file_path)
    copy_file(cif_file_path, new_cif_file_path)
    
    sym = crystal_symmetry_from_ins.extract_from(qpeak_res_name)
    structure = structure(crystal_symmetry=sym)
    uc = structure.crystal_symmetry().unit_cell()
    real_cart = [list(uc.orthogonalize(_point)) for _point in real_frac]

    with open(xyz_save_path, 'w') as file:
        file.write(f'{str(atom_num)}\n')
        file.write(f'{mol_formular}\n')
        for atom, coord in zip(shelxt_pred, real_cart):
            one_line = f'{atom} {str(coord[0])} {str(coord[1])} {str(coord[2])}\n'
            file.write(one_line)

    with open(gjf_save_path, 'w') as file:
        file.write(f'%chk=11.chk\n')
        file.write(f'#opt freq pm6\n')
        file.write(f'\n')
        file.write(f'Title {mol_formular}\n')
        file.write(f'\n')
        file.write(f'0 1\n')
        for atom, coord in zip(shelxt_pred, real_cart):
            one_line = f'{atom} {str(coord[0])} {str(coord[1])} {str(coord[2])}\n'
            file.write(one_line)
    
    with zipfile.ZipFile(zip_save_path, 'w') as zipf:
        for file in zip_lst:
            zipf.write(file)
    

    wr2, r1, goof, q1 = get_R2(qpeak_res_name)

    fitting_metrics = {
        'wr2':wr2,
        'r1':r1,
        'goof':goof,
        'q1':q1,
        'formular':mol_formular
        }
    # if wr2 < 0.35:
    #     is_valid = True
    # else:
    #     is_valid = False
    fitting_metrics['is_structure_valid'] = is_structure_valid
    fitting_metrics['is_quality_valid'] = is_quality_valid
    if is_structure_valid:
        structure_mes = 'Stucture valid!'
    else:
        structure_mes = 'Slight structural flaws, possibly due to disorder or the ambiguity of hydrogen atom positions.'
        # hydro_mes = ' Adding hydrogen atoms to free water is not currently supported.'
        # structure_mes += hydro_mes
    quality_mes = ''
    if is_quality_valid:
        quality_mes = 'Reflections valid!'
    else:
        quality_mes = 'May need reflection correction'
    # fitting_metrics['mes'] = structure_mes
    fitting_metrics['structure_mes'] = structure_mes
    fitting_metrics['quality_mes'] = quality_mes
        
    # if is_quality_valid:

    with open(metrics_save_path, 'w') as file:
        json.dump(fitting_metrics, file)








