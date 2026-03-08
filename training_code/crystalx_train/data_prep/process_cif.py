import os
import iotbx.cif
from sympy import symbols, Poly, simplify

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

idx = 1
SYMMDICT = {}
SYMMDICT['P'] = 1
SYMMDICT['I'] = 2
SYMMDICT['R'] = 3
SYMMDICT['F'] = 4
SYMMDICT['A'] = 5
SYMMDICT['B'] = 6
SYMMDICT['C'] = 7

data_dir = 'real_xrd_dataset'
file_list = [f.name for f in os.scandir(data_dir) if f.is_dir()]
for fname in file_list:
    path_name = f'real_xrd_dataset/{fname}/{fname}.cif'
    file_path = f'real_xrd_dataset/{fname}/{fname}.ins'
    try:
        structure = iotbx.cif.reader(file_path=path_name).build_crystal_structures()[fname]
    except iotbx.cif.CifParserError:
        continue
    wavelength = structure.wavelength
    sfac_dict = structure.unit_cell_content()
    cell_params = structure.crystal_symmetry().as_cif_block()
    a = cell_params['_cell.length_a']
    b = cell_params['_cell.length_b']
    c = cell_params['_cell.length_c']
    alpha = cell_params['_cell.angle_alpha']
    beta = cell_params['_cell.angle_beta']
    gamma = cell_params['_cell.angle_gamma']
    operation_xyz = cell_params['_space_group_symop.operation_xyz']
    spag = cell_params['_space_group.name_H-M_alt']
    bravis = spag[0]
    is_centric = structure.crystal_symmetry().space_group().is_centric()

    with open(file_path, 'w') as file:
        file.write("TITL\n")
        cell_ins = [str(wavelength),a,b,c,alpha,beta,gamma]
        cell_ins = ' '.join(cell_ins)
        file.write("CELL "+cell_ins+"\n")
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
        file.write("LIST 2\n")
        #file.write("PLAN 500\n")
        file.write("TREF 10000\n")
        file.write("HKLF 4\n")
        file.write("END ")
