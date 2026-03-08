import os
import numpy as np
import re
from itertools import combinations, permutations, product
from scipy.spatial.distance import cdist
import torch
from sympy import symbols, Poly, simplify
from collections import Counter

def are_polynomials_equal(poly_list1, poly_list2):
    # 定义变量
    x, y, z = symbols('x y z')
    total_degree = 0
    for poly1, poly2 in zip(poly_list1, poly_list2):
        # 将字符串转换为 SymPy 的多项式对象
        poly_obj1 = Poly(poly1, x, y, z)
        poly_obj2 = Poly(poly2, x, y, z)
        # 判断两个多项式相加是否变量是否都被消去了
        sum_result = simplify(poly_obj1 + poly_obj2)
        total_degree += sum_result.as_poly().total_degree()
    return total_degree == 0 or total_degree == 3

def count_unique_polynomial_lists(poly_lists):
    unique_lists = []
    for i, poly_list1 in enumerate(poly_lists):
        is_unique = True
        for j, poly_list2 in enumerate(unique_lists):
            if are_polynomials_equal(poly_list1, poly_list2):
                is_unique = False
                break
        if is_unique:
            unique_lists.append(poly_list1)
    return unique_lists



def load_shelxt(res_file_path, begin_flag = 'PLAN', end_flag = 'HKLF', is_atom_name = False):
    qpeak = []
    coord = []
    shelxt_pred = []
    qvalue_list = []
    isotropy_list = []
    with open(res_file_path, 'r') as res_file:
        start_reading = False
        for line in res_file:
            if start_reading:
                if end_flag in line:
                    break
                qpeak.append(line.split())
            if begin_flag in line:
                start_reading = True
                continue
    for i in range(0, len(qpeak), 1):
        first_line = qpeak[i]
        if is_atom_name:
            atom_name = first_line[0]
        else:
            atom_name = extract_non_numeric_prefix(first_line[0])
        shelxt_pred.append(atom_name)
        x = float(first_line[2])
        y = float(first_line[3])
        z = float(first_line[4])
        coord.append([x,y,z])
        qvalue_list.append(float(first_line[-1]))
        isotropy_list.append(float(first_line[-2]))
    return coord, shelxt_pred, qvalue_list, isotropy_list

def load_shelxt_final(res_file_path, begin_flag = 'PLAN', end_flag = 'HKLF', is_atom_name = False, is_hydro = True):
    qpeak = []
    coord = []
    shelxt_pred = []
    with open(res_file_path, 'r') as res_file:
        start_reading = False
        for line in res_file:
            if start_reading:
                if end_flag in line:
                    break
                if 'AFIX' in line:
                    continue
                first_str = line.split()[0]
                if any(char.isalpha() for char in first_str):
                    qpeak.append(line.split())
            if begin_flag in line:
                start_reading = True
                continue
    for i in range(0, len(qpeak), 1):
        first_line = qpeak[i]
        if is_atom_name:
            atom_name = first_line[0]
        else:
            atom_name = extract_non_numeric_prefix(first_line[0])
        if not is_hydro:
            if atom_name == 'H':
                continue
        shelxt_pred.append(atom_name)
        x = float(first_line[2])
        y = float(first_line[3])
        z = float(first_line[4])
        coord.append([x,y,z])
    return coord, shelxt_pred

def update_shelxt(res_file_path, new_res_file_path, atom_list, no_sfac = False, refine_round = 10):
    fname = os.path.basename(new_res_file_path)
    fname = os.path.splitext(fname)[0]
    sfac = {}
    p = 0
    atom_cnt = {}
    atom_list = [item.capitalize() for item in atom_list]
    if no_sfac:
        sfac_line = 'SFAC '
        unit_line = 'UNIT '
        pred_sfac = dict(Counter(atom_list))
        for k, v in pred_sfac.items():
            sfac_line += f'{k} '
            unit_line += f'{str(v)} '
        sfac_line += 'H\n'
        unit_line += '37\n'
    with open(res_file_path, 'r') as res_file, open(new_res_file_path, 'w') as new_res_file:
        start_reading = False
        for line in res_file:
            if not start_reading:
                if 'TITL' in line:
                    if line.split()[0] == 'TITL':
                        line = f'TITL {fname}\n'
                if 'SFAC' in line:
                    if no_sfac:
                        line = sfac_line
                    sfac_atom = line.split()[1:]
                    for i in range(len(sfac_atom)):
                        sfac[sfac_atom[i].capitalize()] = i + 1
                if 'UNIT' in line:
                    if no_sfac:
                        line = unit_line
                if 'L.S.' in line:
                    line = f'L.S. {str(refine_round)}\n'
                if 'LIST' in line:
                    line = f'LIST 4\n'
                new_res_file.write(line)
            if start_reading:
                if 'HKLF' in line:
                    new_res_file.write(line)
                    start_reading = False
                else:
                    atom_pos = line.split()
                    atom = atom_list[p].capitalize()
                    if atom not in atom_cnt:
                        atom_cnt[atom] = 1
                    else:
                        atom_cnt[atom] += 1
                    atom_name = atom + str(atom_cnt[atom])
                    atom_pos[0] = atom_name
                    atom_pos[1] = str(sfac[atom])
                    atom_pos = ' '.join(atom_pos)
                    atom_pos += '\n'
                    p += 1
                    new_res_file.write(atom_pos)
            if 'PLAN' in line:
                new_res_file.write('ANIS\n')
                start_reading = True

def update_shelxt_weight(res_file_path, new_res_file_path, is_weight = True, is_acta = True, re_afix = False):
    if is_weight:
        with open(res_file_path, 'r') as res_file:
            start_reading = False
            for line in res_file:
                if start_reading:
                    if 'WGHT' in line:
                        weight_line = line
                if 'END' in line:
                    start_reading = True
    with open(res_file_path, 'r') as res_file, open(new_res_file_path, 'w') as new_res_file:
        for line in res_file:
            if re_afix:
                if 'AFIX' in line:
                    if atom_name == 'N':
                        line = 'AFIX 3\n'
                else:
                    line_info = line.split()
                    if len(line_info) > 0:
                        if any(char.isalpha() for char in line_info[0]):
                            atom_name = extract_non_numeric_prefix(line_info[0])
            if 'WGHT' in line:
                if is_weight:
                    if is_acta:
                        new_res_file.write('ACTA\n')
                    line = weight_line
            if 'END' in line:
                new_res_file.write('END')
                break
            new_res_file.write(line)

def update_shelxt_final(res_file_path, new_res_file_path, sfac, no_given_sfac = False):
    with open(res_file_path, 'r') as res_file, open(new_res_file_path, 'w') as new_res_file:
        for line in res_file:
            if 'ZERR' in line:
                if no_given_sfac:
                    zerr = 1.00
                    sp_line = line.split()
                    sp_line[1] = str(zerr)
                    line = ' '.join(sp_line)
                    line += '\n'
                else:
                    zerr = float(line.split()[1])
            if 'UNIT' in line:
                sp_line = line.split()
                cell_hydro_num = int(sfac['H'] * zerr)
                sp_line[-1] = str(cell_hydro_num)
                line = ' '.join(sp_line)
                line += '\n'
            if 'LIST 6' in line:
                line = 'LIST 4\n'
            new_res_file.write(line)

def read_checkcif(chk_path):
    ln_max = 10
    # with open(chk_path, 'r') as chk_file:
    #     for line in chk_file:
    #         if 'ALERT_Level_B' in line:
    #             ln_max = 10
    #             break
    is_structure_valid = True
    is_quality_valid = True
    with open(chk_path, 'r') as chk_file:
        line_cnt = 0
        for line in chk_file:
            if '#=======================' in line:
                line_cnt += 1
            if line_cnt == ln_max:
                break
            if 'ALERT_2_B' in line or 'ALERT_2_A' in line:
                print(line)
                is_structure_valid = False
            if 'ALERT_3_B' in line or 'ALERT_3_A' in line:
                is_quality_valid = False
    return is_structure_valid, is_quality_valid

def read_checkcif123(chk_path):
    is_structure_valid = True
    is_quality_valid = True
    with open(chk_path, 'r') as chk_file:
        for line in chk_file:
            if 'ALERT_2_B' in line or 'ALERT_2_A' in line:
                # print(line)
                is_structure_valid = False
                break
    return is_structure_valid, is_quality_valid


def get_bond(lst_file_path):
    mol_graph = {}
    with open(lst_file_path, 'r') as lst_file:
        start_reading = False
        for line in lst_file:
            if 'connectivity table' in line:
                start_reading = True
                continue
            if start_reading:
                if '-' in line:
                    k = line.split()[0]
                    if 'no bonds found' in line:
                        v = []
                    else:
                        v = line.split()[2:]
                        # v = [item for item in v if '$' not in item]
                    mol_graph[k] = v
                if 'Reflections' in line or 'Operators for generating equivalent atoms' in line:
                    break
    return mol_graph

def get_bond_in_order(lst_file_path):
    bond_num = []
    with open(lst_file_path, 'r') as lst_file:
        start_reading = False
        for line in lst_file:
            if 'connectivity table' in line:
                start_reading = True
                continue
            if start_reading:
                if '-' in line:
                    k = line.split()[0]
                    if 'no bonds found' in line:
                        v = []
                    else:
                        v = line.split()[2:]
                        v = [item for item in v if '$' not in item]

                    bond_num.append(len(v))
                if 'Reflections' in line or 'Operators for generating equivalent atoms' in line:
                    break
    return bond_num

def hydro_num_calibrating(ins_file_path, mol_graph, hydro_num):
    _, atom_name, _, _ = load_shelxt(ins_file_path, begin_flag='ANIS', atom_name=True)
    calibrated_hydro_num = []
    for atom, h in zip(atom_name, hydro_num):
        h = h.item()
        if h == 0 or h == 4:
            calibrated_hydro_num.append(h)
            continue
        try:
            connect_atom = mol_graph[atom]
        except Exception as e:
            print(ins_file_path)
        atom_type = extract_non_numeric_prefix(atom)
        if atom_type == 'C':
            if len(connect_atom) == 3:
                calibrated_hydro_num.append(1)
            elif len(connect_atom) == 2:
                if h > 2:
                    calibrated_hydro_num.append(2)
                else:
                    calibrated_hydro_num.append(h)
            elif len(connect_atom) == 1:
                if h > 3:
                    calibrated_hydro_num.append(3)
                else:
                    calibrated_hydro_num.append(h)
            else:
                calibrated_hydro_num.append(h)
        elif atom_type == 'N':
            if len(connect_atom) == 3:
                calibrated_hydro_num.append(1)
            elif len(connect_atom) == 2:
                if h > 2:
                    calibrated_hydro_num.append(2)
                else:
                    calibrated_hydro_num.append(h)
            elif len(connect_atom) == 1:
                if h < 2:
                    calibrated_hydro_num.append(2)
                else:
                    calibrated_hydro_num.append(h)
            else:
                calibrated_hydro_num.append(h)
        else:
            calibrated_hydro_num.append(h)
        
    return torch.tensor(calibrated_hydro_num)



def gen_hfix_ins(ins_file_path, mol_graph, hydro_num):
    _, atom_name, _, _ = load_shelxt(ins_file_path, begin_flag='ANIS', is_atom_name=True)
    hydro_type = {}
    hydro_type['HFIX 13'] = []
    hydro_type['HFIX 23'] = []
    hydro_type['HFIX 137'] = []
    hydro_type['HFIX 153'] = []
    hydro_type['HFIX 163'] = []
    hydro_type['HFIX 43'] = []
    hydro_type['HFIX 93'] = []
    hydro_type['HFIX 147'] = []
    for atom, h in zip(atom_name, hydro_num):
        h = h.item()
        if h == 0 or h == 4:
            continue
        connect_atom = mol_graph[atom]
        atom_type = extract_non_numeric_prefix(atom)
        if atom_type == 'C':
            if h == 1:
                if len(connect_atom) == 2:
                    hydro_type['HFIX 43'].append(atom)
                elif len(connect_atom) == 1:
                    hydro_type['HFIX 163'].append(atom)
                elif len(connect_atom) == 3:
                    hydro_type['HFIX 13'].append(atom)

            if h == 2:
                if len(connect_atom) == 1:
                    hydro_type['HFIX 93'].append(atom)
                elif len(connect_atom) == 2:
                    hydro_type['HFIX 23'].append(atom)
            if h == 3: 
                if len(connect_atom) == 1:
                    hydro_type['HFIX 137'].append(atom)
        if atom_type == 'N':
            if h == 1:
                if len(connect_atom) == 2:
                    hydro_type['HFIX 43'].append(atom)
                if len(connect_atom) == 3:
                    hydro_type['HFIX 13'].append(atom)
            if h == 2:
                if len(connect_atom) == 1:
                    hydro_type['HFIX 93'].append(atom)
                if len(connect_atom) == 2:
                    hydro_type['HFIX 23'].append(atom)
            if h == 3:
                if len(connect_atom) < 2:
                    hydro_type['HFIX 137'].append(atom)
                
        if atom_type == 'O':
            if h == 1:
                if len(connect_atom) == 1:
                    hydro_type['HFIX 147'].append(atom)
            # if h == 2:
            #     hydro_type['HFIX 23'].append(atom)
        if atom_type == 'B':
            if h == 1:
                hydro_type['HFIX 153'].append(atom)
    hfix_ins = []
    for k, v in hydro_type.items():
        if len(v) == 0:
            continue
        ins = ''
        v = [v[i:i+10] for i in range(0, len(v), 10)]
        for item in v:
            ins = k + ' ' + ' '.join(item) + '\n'
            hfix_ins.append(ins)
    return hfix_ins

def invalid_hydro(ins_file_path, mol_graph, hydro_num):
    _, atom_name, _, _ = load_shelxt(ins_file_path, begin_flag='ANIS', is_atom_name=True)
    flag = False
    for atom, h in zip(atom_name, hydro_num):
        h = h.item()
        if h == 0:
            continue
        connect_atom = mol_graph[atom]
        atom_type = extract_non_numeric_prefix(atom)
        if atom_type == 'O':
            if h == 3:
                print(ins_file_path)
            # if len(connect_atom) != 1:
            #     flag = True
            if h > 1:
                flag = True
        if atom_type == 'N':
            if h == 1:
                if len(connect_atom) == 1:
                    flag = True
            if h == 2:
                if len(connect_atom) > 2:
                    flag = True
            if h == 3:
                if len(connect_atom) > 1:
                    flag = True
            if h == 4:
                flag = True
        if atom_type == 'C':
            if h == 1:
                if len(connect_atom) > 3:
                    flag = True
            if h == 2:
                if len(connect_atom) > 2:
                    flag = True
            if h == 3:
                if len(connect_atom) > 1:
                    flag = True
    return flag

def oxygen_calibrate(ins_file_path, mol_graph, hydro_num):
    _, atom_name, _, _ = load_shelxt(ins_file_path, begin_flag='ANIS', is_atom_name=True)
    new_hydro_num = []
    for atom, h in zip(atom_name, hydro_num):
        h = h.item()
        connect_atom = mol_graph[atom]
        atom_type = extract_non_numeric_prefix(atom)
        if atom_type == 'O':
            if len(connect_atom) == 0:
                new_hydro_num.append(h)
            elif len(connect_atom) == 1:
                if h > 1:
                    new_hydro_num.append(1)
                else:
                    new_hydro_num.append(h)
            else:
                new_hydro_num.append(h)
        else:
            new_hydro_num.append(h)
    return torch.tensor(new_hydro_num)

def sfac_purturb(y):
    sfac = {}
    sfac[6] = 0
    sfac[7] = 0
    sfac[8] = 0
    for item in y:
        if item not in sfac.keys():
            sfac[item] = 0
        sfac[item] += 1
    ny = []
    sy = []
    for k in sorted(sfac.keys(), reverse = True):
        if k > 8:
            ny += [k] * sfac[k]
        if k < 6:
            sy += [k] * sfac[k]
    all_y = []
    nums = [-2,-1,0,1,2]
    # nums = [0]
    perms = product(nums, repeat=3)
    for perm in perms:
        ky = []
        if sum(perm) == 0:
            c = perm[0] + sfac[6]
            n = perm[1] + sfac[7]
            o = perm[2] + sfac[8]
            if c > -1 and n > -1 and o > -1:
                ky += [8] * o
                ky += [7] * n
                ky += [6] * c
                all_y.append(ny + ky + sy)
    return all_y

def one_atom_purturb(y):
    real_num = len(y)
    refine_z = [y]
    for i in range(real_num):
        if y[i] in [7,16]:
            new_z = [item for item in y]
            new_z[i] += 1
            refine_z.append(new_z)
            new_z = [item for item in y]
            new_z[i] -= 1
            refine_z.append(new_z)
        if y[i] in [8,17]:
            new_z = [item for item in y]
            new_z[i] -= 1
            refine_z.append(new_z)
            if y[i] in [8]:
                new_z = [item for item in y]
                new_z[i] -= 2
                refine_z.append(new_z)
        if y[i] in [6,15]:
            new_z = [item for item in y]
            new_z[i] += 1
            refine_z.append(new_z)
            if y[i] in [6]:
                new_z = [item for item in y]
                new_z[i] += 2
                refine_z.append(new_z)
    return refine_z

def one_atom_purturb_hydro(y, z):
    real_num = len(y)
    refine_z = [y]
    for i in range(real_num):
        if z[i] not in [6,7,8]:
            continue
        if y[i] in [1,2]:
            new_z = [item for item in y]
            new_z[i] += 1
            refine_z.append(new_z)
            new_z = [item for item in y]
            new_z[i] -= 1
            refine_z.append(new_z)
        # if y[i] in [3]:
        #     new_z = [item for item in y]
        #     new_z[i] -= 1
        #     refine_z.append(new_z)
        #     new_z = [item for item in y]
        #     new_z[i] -= 2
        #     refine_z.append(new_z)
        # if y[i] in [0]:
        #     new_z = [item for item in y]
        #     new_z[i] += 1
        #     refine_z.append(new_z)
        #     new_z = [item for item in y]
        #     new_z[i] += 2
        #     refine_z.append(new_z)
    return refine_z

def two_atom_purturb(y, sample_num = 5):
    real_num = len(y)
    indices = [random.sample(real_num, 2) for i in range(sample_num)]
    refine_z = []
    for indice in indices:
        if y[i] in [7,16]:
            new_z = [item for item in y]
            new_z[i] += 1
            refine_z.append(new_z)
            new_z = [item for item in y]
            new_z[i] -= 1
            refine_z.append(new_z)
        if y[i] in [8,17]:
            new_z = [item for item in y]
            new_z[i] -= 1
            refine_z.append(new_z)
        if y[i] in [6,15]:
            new_z = [item for item in y]
            new_z[i] += 1
            refine_z.append(new_z)
    return refine_z

def n_atom_purturb_hydro(y, n = 2, ratio=1):
    real_num = len(y)
    refine_z = []
    all_comb = list(combinations(list(range(real_num)),n))
    all_comb = random.sample(all_comb, ratio)
    for comb_id in all_comb:
        new_z = [item for item in y]
        new_z[comb_id[0]] += 1
        new_z[comb_id[1]] += 1
        if not any(x < 0 for x in new_z):
            refine_z.append(new_z)
        new_z = [item for item in y]
        new_z[comb_id[0]] -= 1
        new_z[comb_id[1]] += 1
        if not any(x < 0 for x in new_z):
            refine_z.append(new_z)
        new_z = [item for item in y]
        new_z[comb_id[0]] += 1
        new_z[comb_id[1]] -= 1
        if not any(x < 0 for x in new_z):
            refine_z.append(new_z)
        new_z = [item for item in y]
        new_z[comb_id[0]] -= 1
        new_z[comb_id[1]] -= 1
        if not any(x < 0 for x in new_z):
            refine_z.append(new_z)
    return refine_z

def hydro_purturb(y):
    real_num = len(y)
    refine_z = []
    all_comb = list(combinations(list(range(real_num)),2))
    for comb_id in all_comb:
        new_z = [item for item in y]
        new_z[comb_id[0]] += 1
        new_z[comb_id[1]] += 1
        if not any(x < 0 for x in new_z):
            refine_z.append(new_z)
        new_z = [item for item in y]
        new_z[comb_id[0]] -= 1
        new_z[comb_id[1]] += 1
        if not any(x < 0 for x in new_z):
            refine_z.append(new_z)
        new_z = [item for item in y]
        new_z[comb_id[0]] += 1
        new_z[comb_id[1]] -= 1
        if not any(x < 0 for x in new_z):
            refine_z.append(new_z)
        new_z = [item for item in y]
        new_z[comb_id[0]] -= 1
        new_z[comb_id[1]] -= 1
        if not any(x < 0 for x in new_z):
            refine_z.append(new_z)
    return refine_z

    

def update_shelxt_hydro(res_file_path, new_res_file_path, hfix_ins):
    with open(res_file_path, 'r') as res_file, open(new_res_file_path, 'w') as new_res_file:
        start_reading = False
        for line in res_file:
            if 'BOND' in line:
                new_res_file.write('\n')
                for hfix in hfix_ins:
                    new_res_file.write(hfix)
                new_res_file.write('BOND $H\n')
                new_res_file.write('CONF\n')
                new_res_file.write('HTAB\n')
            elif 'END' in line:
                new_res_file.write(line)
                break
            else:
                new_res_file.write(line)

def get_R2(res_file_path):
    with open(res_file_path, 'r') as res_file:
        start_reading = False
        for line in res_file:
            if 'REM wR2' in line:
                start_reading = True
                wr2 = float(line.split()[3][:-1])
                goof = float(line.split()[8][:-1])
            if 'REM R1' in line:
                r1 = float(line.split()[3])
            if start_reading and 'Q1' in line.split():
                q1 = float(line.split()[7])
    return wr2, r1, goof, q1

def get_oxygen_Q(lst_file_path, ins_file_path, mol_graph, hydro_num):
    Q_atom = {}

    _, atom_name_lst, _, _ = load_shelxt(ins_file_path, begin_flag='ANIS', is_atom_name=True)
    hydro_num_dict = dict(zip(atom_name_lst, hydro_num))

    with open(lst_file_path, 'r') as lst_file:
        start_reading = False
        for line in lst_file:
            if 'Fourier peaks appended to .res file' in line:
                start_reading = True
                continue
            if 'Shortest distances between peaks (including symmetry equivalents)' in line:
                break
            if start_reading and 'Q' in line:
                q_atom_name = line.split()[9]
                if q_atom_name not in atom_name_lst:
                    continue
                h_num = hydro_num_dict[q_atom_name]
                # bond_num = len(mol_graph[atom_name])
                atom_type = extract_non_numeric_prefix(q_atom_name)
                if atom_type == 'O' and h_num == 2:
                    qdist = float(line.split()[8])
                    if qdist < 0.7 and qdist < 1.2:
                        x = float(line.split()[2])
                        y = float(line.split()[3])
                        z = float(line.split()[4])
                        if q_atom_name not in Q_atom.keys():
                            Q_atom[q_atom_name] = []
                        Q_atom[q_atom_name].append([x,y,z])
    return Q_atom



def copy_file(res_file_path, new_res_file_path):
    with open(res_file_path, 'r') as res_file, open(new_res_file_path, 'w') as new_res_file:
        for line in res_file:
            new_res_file.write(line)

def gt_info(res_file_path, new_res_file_path, structure):
    sfac = {}
    anis_dict = {}
    atom_dict = {}
    with open(res_file_path, 'r') as res_file:
        for line in res_file:
            if 'SFAC' in line:
                sfac_atom = line.split()[1:]
                for i in range(len(sfac_atom)):
                    sfac[sfac_atom[i].capitalize()] = str(i + 1)
                break

    atom_type = list(structure.as_cif_block()['_atom_site_type_symbol'])
    atom_label = list(structure.as_cif_block()['_atom_site_label'])

    x = list(structure.as_cif_block()['_atom_site_fract_x'])
    y = list(structure.as_cif_block()['_atom_site_fract_y'])
    z = list(structure.as_cif_block()['_atom_site_fract_z'])

    iso = list(structure.as_cif_block()['_atom_site_U_iso_or_equiv'])

    anis_label = list(structure.as_cif_block()['_atom_site_aniso_label'])
    anis11 = list(structure.as_cif_block()['_atom_site_aniso_U_11'])
    anis22 = list(structure.as_cif_block()['_atom_site_aniso_U_22'])
    anis33 = list(structure.as_cif_block()['_atom_site_aniso_U_33'])
    anis12 = list(structure.as_cif_block()['_atom_site_aniso_U_12'])
    anis13 = list(structure.as_cif_block()['_atom_site_aniso_U_13'])
    anis23 = list(structure.as_cif_block()['_atom_site_aniso_U_23'])

    for i in range(len(anis_label)):
        atom = anis_label[i]
        anis_dict[atom] = ''
        anis_dict[atom] += (anis11[i] + ' ')
        anis_dict[atom] += (anis22[i] + ' ')
        anis_dict[atom] += '=\n         '
        anis_dict[atom] += (anis33[i] + ' ')
        anis_dict[atom] += (anis23[i] + ' ')
        anis_dict[atom] += (anis13[i] + ' ')
        anis_dict[atom] += anis12[i]

    for i in range(len(atom_label)):
        atom = atom_label[i]
        atom_dict[atom] = ''
        atom_dict[atom] += (atom + ' ')
        atom_dict[atom] += (sfac[atom_type[i].capitalize()] + ' ')
        atom_dict[atom] += (x[i] + ' ')
        atom_dict[atom] += (y[i] + ' ')
        atom_dict[atom] += (z[i] + ' ')
        atom_dict[atom] += (str(11.00) + ' ')
        if atom not in anis_dict.keys():
            atom_dict[atom] += (iso[i])
        else:
            atom_dict[atom] += anis_dict[atom]
        atom_dict[atom] += '\n'

    with open(res_file_path, 'r') as res_file, open(new_res_file_path, 'w') as new_res_file:
        for line in res_file:
            if 'PLAN' not in line:
                new_res_file.write(line)
            else:
                new_res_file.write(line)
                for k in atom_dict.keys():
                    new_res_file.write(atom_dict[k])
                new_res_file.write('HKLF 4\n')
                new_res_file.write('END')
                break


SYMMDICT = {}
SYMMDICT['P'] = 1
SYMMDICT['I'] = 2
SYMMDICT['R'] = 3
SYMMDICT['F'] = 4
SYMMDICT['A'] = 5
SYMMDICT['B'] = 6
SYMMDICT['C'] = 7
def cif2ins(out_path, structure, hydro_info = True):
    wavelength = structure.wavelength
    sfac_dict = structure.unit_cell_content()
    sfac = {}
    anis_dict = {}
    atom_dict = {}
    p = 1
    for k in sfac_dict.keys():
        sfac[k.capitalize()] = str(p)
        p += 1
    cell_params = structure.crystal_symmetry().as_cif_block()
    a = cell_params['_cell.length_a']
    b = cell_params['_cell.length_b']
    c = cell_params['_cell.length_c']
    zerr = 'ZERR 1.0 0.0, 0.0, 0.0, 0.0, 0.0, 0.0'
    alpha = cell_params['_cell.angle_alpha']
    beta = cell_params['_cell.angle_beta']
    gamma = cell_params['_cell.angle_gamma']
    operation_xyz = cell_params['_space_group_symop.operation_xyz']
    spag = cell_params['_space_group.name_H-M_alt']
    bravis = spag[0]
    is_centric = structure.crystal_symmetry().space_group().is_centric()

    atom_type = list(structure.as_cif_block()['_atom_site_type_symbol'])
    atom_label = list(structure.as_cif_block()['_atom_site_label'])

    print(structure.as_cif_block())

    x = list(structure.as_cif_block()['_atom_site_fract_x'])
    y = list(structure.as_cif_block()['_atom_site_fract_y'])
    z = list(structure.as_cif_block()['_atom_site_fract_z'])

    iso = list(structure.as_cif_block()['_atom_site_U_iso_or_equiv'])

    anis_label = list(structure.as_cif_block()['_atom_site_aniso_label'])
    anis11 = list(structure.as_cif_block()['_atom_site_aniso_U_11'])
    anis22 = list(structure.as_cif_block()['_atom_site_aniso_U_22'])
    anis33 = list(structure.as_cif_block()['_atom_site_aniso_U_33'])
    anis12 = list(structure.as_cif_block()['_atom_site_aniso_U_12'])
    anis13 = list(structure.as_cif_block()['_atom_site_aniso_U_13'])
    anis23 = list(structure.as_cif_block()['_atom_site_aniso_U_23'])

    for i in range(len(anis_label)):
        atom = anis_label[i]
        anis_dict[atom] = ''
        anis_dict[atom] += (anis11[i] + ' ')
        anis_dict[atom] += (anis22[i] + ' ')
        anis_dict[atom] += '=\n         '
        anis_dict[atom] += (anis33[i] + ' ')
        anis_dict[atom] += (anis23[i] + ' ')
        anis_dict[atom] += (anis13[i] + ' ')
        anis_dict[atom] += anis12[i]

    for i in range(len(atom_label)):
        atom = atom_label[i]
        attype = atom_type[i].capitalize()
        if not hydro_info:
            if attype == 'H':
                continue
        atom_dict[atom] = ''
        atom_dict[atom] += (atom + ' ')
        atom_dict[atom] += (sfac[attype] + ' ')
        atom_dict[atom] += (x[i] + ' ')
        atom_dict[atom] += (y[i] + ' ')
        atom_dict[atom] += (z[i] + ' ')
        atom_dict[atom] += (str(11.00) + ' ')
        if atom not in anis_dict.keys():
            atom_dict[atom] += (iso[i])
        else:
            atom_dict[atom] += anis_dict[atom]
        atom_dict[atom] += '\n'
    
    with open(out_path, 'w') as file:
        file.write("TITL\n")
        cell_ins = [str(wavelength),a,b,c,alpha,beta,gamma]
        cell_ins = ' '.join(cell_ins)
        file.write("CELL "+cell_ins+"\n")
        file.write(zerr+"\n")
        if is_centric:
            latt_ins = SYMMDICT[bravis]
        else:
            latt_ins = -1 * SYMMDICT[bravis]
        file.write("LATT "+str(latt_ins)+"\n")

        xyz_list = []
        for item in operation_xyz:
            if are_polynomials_equal(item.split(','),'x,y,z'.split(',')) or are_polynomials_equal(item.split(','),'-x,-y,-z'.split(',')):
                continue
            xyz_list.append(item.split(','))
        filtered_xyz_list = count_unique_polynomial_lists(xyz_list)
        for item in filtered_xyz_list:
            item = ','.join(item)
            file.write("SYMM "+item+"\n")
        atomlist = []
        numlist = []
        for k,v in sfac_dict.items():
            atomlist.append(k)
            numlist.append(str(int(v)))
        atomins = ' '.join(atomlist)
        numins = ' '.join(numlist)
        file.write("SFAC "+atomins+"\n")
        file.write("UNIT "+numins+"\n")
        file.write("L.S. 0\n")
        file.write("BOND\n")
        file.write("LIST 6\n")
        file.write("FMAP 2\n")
        file.write("PLAN 20\n")
        for k in atom_dict.keys():
            file.write(atom_dict[k])
        file.write("HKLF 4\n")
        file.write("END ")

def hkl2hkl(in_path, out_path):
    from iotbx.reflection_file_reader import any_reflection_file
    hkl_file = any_reflection_file(in_path)
    miller = hkl_file.as_miller_arrays()
    real_f = miller[1]
    with open(out_path, 'w') as f:
        real_f.export_as_shelx_hklf(f)




def cif2ins_no_atom(out_path, structure, wavel=None):
    if wavel:
        wavelength = wavel
    else:
        wavelength = structure.wavelength
    sfac_dict = structure.unit_cell_content()
    sfac = {}
    anis_dict = {}
    atom_dict = {}
    p = 1
    for k in sfac_dict.keys():
        sfac[k.capitalize()] = str(p)
        p += 1
    cell_params = structure.crystal_symmetry().as_cif_block()
    a = cell_params['_cell.length_a']
    b = cell_params['_cell.length_b']
    c = cell_params['_cell.length_c']
    zerr = 'ZERR 1.0 0.0, 0.0, 0.0, 0.0, 0.0, 0.0'
    alpha = cell_params['_cell.angle_alpha']
    beta = cell_params['_cell.angle_beta']
    gamma = cell_params['_cell.angle_gamma']
    operation_xyz = cell_params['_space_group_symop.operation_xyz']
    spag = cell_params['_space_group.name_H-M_alt']
    bravis = spag[0]
    is_centric = structure.crystal_symmetry().space_group().is_centric()
    
    with open(out_path, 'w') as file:
        file.write("TITL\n")
        cell_ins = [str(wavelength),a,b,c,alpha,beta,gamma]
        cell_ins = ' '.join(cell_ins)
        file.write("CELL "+cell_ins+"\n")
        file.write(zerr+"\n")
        if is_centric:
            latt_ins = SYMMDICT[bravis]
        else:
            latt_ins = -1 * SYMMDICT[bravis]
        file.write("LATT "+str(latt_ins)+"\n")

        xyz_list = []
        for item in operation_xyz:
            if are_polynomials_equal(item.split(','),'x,y,z'.split(',')) or are_polynomials_equal(item.split(','),'-x,-y,-z'.split(',')):
                continue
            xyz_list.append(item.split(','))
        filtered_xyz_list = count_unique_polynomial_lists(xyz_list)
        for item in filtered_xyz_list:
            item = ','.join(item)
            file.write("SYMM "+item+"\n")
        atomlist = []
        numlist = []
        for k,v in sfac_dict.items():
            atomlist.append(k)
            numlist.append(str(int(v)))
        atomins = ' '.join(atomlist)
        numins = ' '.join(numlist)
        file.write("SFAC "+atomins+"\n")
        file.write("UNIT "+numins+"\n")
        file.write("HKLF 4\n")
        file.write("END ")


def change_to_cart(points, model):
    uc = model.crystal_symmetry().unit_cell()
    cart_points = [list(uc.orthogonalize(_point)) for _point in points]
    cart_points = np.array(cart_points)
    return cart_points

def change_to_frac(points, model):
    uc = model.crystal_symmetry().unit_cell()
    cart_points = [list(uc.fractionalize(_point)) for _point in points]
    cart_points = np.array(cart_points)
    return cart_points

def remove_nan(points, label):
    nan_indices = np.any(np.isnan(points), axis=1)
    return points[~nan_indices], label[~nan_indices]

def extract_non_numeric_prefix(string):
    match = re.match(r'^[a-zA-Z]+', string)
    return match.group() if match else ''

def perturbe(points, mean = 0, std = 1, scale = 1):
    return points + scale*np.random.normal(mean, std, points.shape)

def kruskal_algorithm(distance_matrix):
    num_vertices = len(distance_matrix)
    proxy_matrix = 10000*np.ones((num_vertices, num_vertices))
    for i in range(num_vertices):
        for j in range(num_vertices):
            if distance_matrix[i][j] < 2:
                proxy_matrix[i][j] = distance_matrix[i][j]
    edges = []
    all_edges = [(i, j, proxy_matrix[i][j]) for i, j in combinations(range(num_vertices), 2)]
    all_edges.sort(key=lambda x: x[2])
    
    parent = [i for i in range(num_vertices)]
    
    def find_set(v):
        if parent[v] == v:
            return v
        return find_set(parent[v])
    
    def union_sets(u, v):
        root_u = find_set(u)
        root_v = find_set(v)
        parent[root_u] = root_v
    
    for edge in all_edges:
        u, v, weight = edge
        if find_set(u) != find_set(v):
            edges.append((u, v))
            union_sets(u, v)
    
    return edges

def organize_edges_by_vertices(edges, distances):
    organized_edges = {}
    for edge in edges:
        u, v = edge
        if u not in organized_edges:
            organized_edges[u] = []
        if v not in organized_edges:
            organized_edges[v] = []

        organized_edges[u].append(int(distances[u,v] * 100))
        organized_edges[v].append(int(distances[v,u] * 100))

    return organized_edges

def get_dist(points):
    num_points = points.shape[0]
    distances = np.zeros((num_points, num_points)) 
    for i in range(num_points):
        for j in range(num_points):
            point_dist = np.linalg.norm(points[i] - points[j])
            distances[i, j] = point_dist
    return distances

def get_connection(matrix, threshold):
    return (matrix < threshold).astype(int)

def is_disorder(model):
    occs = model.get_atoms().extract_occ()
    occs = [occ for occ in occs]
    for num in occs:
        if num < 1:
            return True
    return False

def is_hydro_special(res_file_path, begin_flag = 'PLAN', end_flag = 'HKLF', is_atom_name = False, is_hydro = True):
    qpeak = []
    coord = []
    shelxt_pred = []
    with open(res_file_path, 'r') as res_file:
        start_reading = False
        for line in res_file:
            if start_reading:
                if end_flag in line:
                    break
                if 'AFIX' in line:
                    continue
                first_str = line.split()[0]
                if any(char.isalpha() for char in first_str):
                    qpeak.append(line.split())
            if begin_flag in line:
                start_reading = True
                continue
    for i in range(0, len(qpeak), 1):
        first_line = qpeak[i]
        if is_atom_name:
            atom_name = first_line[0]
        else:
            atom_name = extract_non_numeric_prefix(first_line[0])
        if not is_hydro:
            if atom_name == 'H':
                continue
        shelxt_pred.append(atom_name)
        x = float(first_line[2])
        y = float(first_line[3])
        z = float(first_line[4])
        if atom_name == 'H':
            if x % 0.25 == 0 or y % 0.25 == 0 or z % 0.25 == 0:
                return True
    return False
def get_gt_points(model):
    uc = model.crystal_symmetry().unit_cell()
    sites_cart = model.get_sites_cart()  
    atom_names = model.get_atoms().extract_element()
    occs = model.get_atoms().extract_occ()
    label = [atom for atom in atom_names]
    points = [list(atsite) for atsite in sites_cart]
    occs = [occ for occ in occs]
    main_label = []
    main_points = []
    main_occ = []
    for _l, _p, _o in zip(label, points, occs):
        if _l.strip() != 'H':
            main_label.append(_l.strip())
            main_points.append(_p)
            main_occ.append(_o)
    frac_points = [list(uc.fractionalize(_point)) for _point in main_points]
    return frac_points, main_label, main_occ

def get_all_gt_points(model):
    uc = model.crystal_symmetry().unit_cell()
    sites_cart = model.get_sites_cart()  
    atom_names = model.get_atoms().extract_element()
    label = [atom for atom in atom_names]
    points = [list(atsite) for atsite in sites_cart]
    main_label = []
    main_points = []
    hydro_label = []
    hydro_points = []
    for _l, _p in zip(label, points):
        if _l.strip() != 'H':
            main_label.append(_l.strip())
            main_points.append(_p)
        else:
            hydro_label.append(_l.strip())
            hydro_points.append(_p)

    main_points = np.array(main_points)
    hydro_num = [0] * main_points.shape[0]
    min_dist = -1
    min_index = -1
    if len(hydro_points) > 0:
        hydro_points = np.array(hydro_points)
        distances = cdist(hydro_points, main_points)
        min_dist = np.min(distances, axis = 1)
        min_index = np.argmin(distances, axis = 1)
        for item in min_index:
            hydro_num[item] += 1
    frac_points = [list(uc.fractionalize(_point)) for _point in main_points]
    hydro_points = [list(uc.fractionalize(_point)) for _point in hydro_points]
    return frac_points, hydro_points, main_label, min_dist, min_index, hydro_num



def valid_hydro_num(main_label, hydro_num):
    is_valid = True
    for atom, hydro in zip(main_label, hydro_num):
        if atom == 'C':
            if hydro > 3:
                is_valid = False
            break
    return is_valid

def get_equiv_pos(ideal_frac, gt, structure, model):
    ideal_cart = change_to_cart(ideal_frac, model)
    equiv_ideal_frac = []
    equiv_gt = []
    is_equiv = True
    equiv_cnt = 0
    for i in range(len(gt)):
        atom_site = ideal_frac[i]
        equiv_sites = structure.sym_equiv_sites(atom_site.tolist()).coordinates()
        equiv_sites = [list(site) for site in equiv_sites]
        equiv_ideal_frac += equiv_sites
        equiv_gt += [gt[i]] * len(equiv_sites)
    equiv_ideal_frac = np.array(equiv_ideal_frac)
    if equiv_ideal_frac.shape[0] < 1:
        is_equiv = False
    
    if is_equiv:
        expand_equiv_ideal_frac = []
        expand_equiv_gt = []
        from itertools import product
        t = [-2,-1,0,1,2]
        t = list(product(t, repeat=3))
        for op in t:
            expand_equiv_ideal_frac.append(equiv_ideal_frac + op)
            expand_equiv_gt += equiv_gt
        expand_equiv_ideal_frac = np.concatenate(expand_equiv_ideal_frac, axis=0)
        expand_equiv_gt = np.array(expand_equiv_gt)
        
        # equiv_ideal_cart = change_to_cart(equiv_ideal_frac, model)
        equiv_ideal_cart = change_to_cart(expand_equiv_ideal_frac, model)
        distances = cdist(ideal_cart, equiv_ideal_cart)
        min_dist = np.min(distances, axis = 1)
        min_index = np.argmin(distances, axis = 1)
        if np.min(min_dist) < 3:
            min_index = np.where(distances < 3)[1]
            min_index = [item for item in min_index if np.min(cdist(equiv_ideal_cart[[item]], ideal_cart)) > 0.2]
            if len(min_index) > 0:
                equiv_cnt += 1
            ideal_cart = np.concatenate([ideal_cart, equiv_ideal_cart[min_index]], axis=0)
            gt = gt + expand_equiv_gt[min_index].tolist()
            # equiv_cnt += 1
            # print(np.min(min_dist))
            # print(equiv_cnt)
            # print(equiv_ideal_cart[min_index])
    return ideal_cart, gt

def get_equiv_pos2(ideal_frac, gt, structure, radius = 3):
    uc = structure.crystal_symmetry().unit_cell()
    ideal_cart = [list(uc.orthogonalize(_point)) for _point in ideal_frac]
    ideal_cart = np.array(ideal_cart)
    equiv_ideal_frac = []
    equiv_gt = []
    is_equiv = True
    equiv_cnt = 0
    for i in range(len(gt)):
        atom_site = ideal_frac[i]
        equiv_sites = structure.sym_equiv_sites(atom_site.tolist()).coordinates()
        equiv_sites = [list(site) for site in equiv_sites]
        equiv_ideal_frac += equiv_sites
        equiv_gt += [gt[i]] * len(equiv_sites)
    equiv_ideal_frac = np.array(equiv_ideal_frac)
    if equiv_ideal_frac.shape[0] < 1:
        is_equiv = False
    
    if is_equiv:
        expand_equiv_ideal_frac = []
        expand_equiv_gt = []
        from itertools import product
        t = [-2,-1,0,1,2]
        t = list(product(t, repeat=3))
        for op in t:
            expand_equiv_ideal_frac.append(equiv_ideal_frac + op)
            expand_equiv_gt += equiv_gt
        expand_equiv_ideal_frac = np.concatenate(expand_equiv_ideal_frac, axis=0)
        expand_equiv_gt = np.array(expand_equiv_gt)
        
        # equiv_ideal_cart = change_to_cart(equiv_ideal_frac, model)
        equiv_ideal_cart = [list(uc.orthogonalize(_point)) for _point in expand_equiv_ideal_frac]
        equiv_ideal_cart = np.array(equiv_ideal_cart)
        distances = cdist(ideal_cart, equiv_ideal_cart)
        min_dist = np.min(distances, axis = 1)
        min_index = np.argmin(distances, axis = 1)
        if np.min(min_dist) < radius:
            min_index = np.where(distances < radius)[1]
            min_index = [item for item in min_index if np.min(cdist(equiv_ideal_cart[[item]], ideal_cart)) > 0.2]
            if len(min_index) > 0:
                equiv_cnt += 1
            ideal_cart = np.concatenate([ideal_cart, equiv_ideal_cart[min_index]], axis=0)
            gt = gt + expand_equiv_gt[min_index].tolist()
            # equiv_cnt += 1
            # print(np.min(min_dist))
            # print(equiv_cnt)
            # print(equiv_ideal_cart[min_index])
    return ideal_cart, gt

def plot_corr(pred='all_fitting_metrics.csv', gt='all_fitting_metrics_gt.csv'):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    gt = pd.read_csv(gt)
    gt.columns = ['id','gt_wr2', 'gt_r1', 'gt_goof', 'gt_q1']
    pred = pd.read_csv(pred)
    pred.columns = ['id','pred_wr2', 'pred_r1', 'pred_goof', 'pred_q1', 'correct_flag']
    merge_pred = pd.merge(gt, pred, on='id', how='inner')
    sns.scatterplot(x='pred_wr2', y='gt_wr2', data=merge_pred)
    plt.title('wr2')
    plt.savefig('wr2_corr.png')






if __name__ == "__main__":
    distance_matrix = np.array([
        [0, 2, 0, 6, 0],
        [2, 0, 3, 8, 5],
        [0, 3, 0, 0, 7],
        [6, 8, 0, 0, 9],
        [0, 5, 7, 9, 0]
    ])

    from iotbx.data_manager import DataManager
    dm = DataManager()                            
    dm.set_overwrite(True)  
    pdb_dir = 'all_pdb_data'

    # import iotbx.cif
    # cif_name = 'final_main_correct/2240237/2240237.cif'
    # fname = '2240237'
    # structure = iotbx.cif.reader(file_path=cif_name).build_crystal_structures()[fname]
    # cif2ins('test.ins', structure)

    path = 'final_main_correct/2218802/2218802_gtWeight.res'
    print(is_hydro_special(path, begin_flag='FVAR'))




    path = 'final_main_correct/4087950/4087950_AI.lst'
    print(get_Q(path))

    # structure = iotbx.cif.reader(file_path=cif_name).build_crystal_structures()[fname]
    # print(structure.unit_cell_content())
    # out_path = f'work_dir/{fname}.ins'
    # cif2ins(out_path, structure)
    
    # res_file_path = 'all_refined_10/2019836/2019836_a.res'
    # new_res_file_path = 'all_refined_10/2019836/2019836_gt.ins'
    # gt_info(res_file_path, new_res_file_path, structure)

    # model = dm.get_model(pdb_name)  
    # print(list(model.get_atoms().extract_b()))
    # frac_points, main_label = get_gt_points(model)
    # print(main_label)

    # res_file_path = 'work_dir/2021070503_AIhydroWeight.res'
    # # hkl_file_path = 'all_xrd_dataset_test/2201868/2201868_a.hkl'
    # # new_res_file_path = 'all_xrd_dataset_test/2201868/2201868_AI.ins'
    # # new_hkl_file_path = 'all_xrd_dataset_test/2201868/2201868_AI.hkl'
    # # copy_file(hkl_file_path, new_hkl_file_path)
    # coord, shelxt_pred = load_shelxt_final(res_file_path, begin_flag = 'FVAR')
    # print(coord)
    # print(shelxt_pred)
    # update_shelxt(res_file_path, new_res_file_path, shelxt_pred)

    # print(get_bond('all_refined_10/2232471/2232471_AIhydro.lst'))
    # wr2, r1, goof, q1 = get_R2('all_refined_10/2019836/2019836_AIhydroWeight.res')
    # print((wr2, r1, goof, q1))
    # y = [3,2,47, 35, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6]
    # print(sfac_purturb(y))

    


    # result_kruskal = kruskal_algorithm(distance_matrix)
    # print(result_kruskal)

