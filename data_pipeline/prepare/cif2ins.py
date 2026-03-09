import shutil
from pathlib import Path

import iotbx.cif
from sympy import Poly, simplify, symbols
from tqdm import tqdm


def are_polynomials_equal(poly_list1, poly_list2):
    x, y, z = symbols("x y z")
    total_degree = 0
    for poly1, poly2 in zip(poly_list1, poly_list2):
        poly_obj1 = Poly(poly1, x, y, z)
        poly_obj2 = Poly(poly2, x, y, z)
        sum_result = simplify(poly_obj1 + poly_obj2)
        total_degree += sum_result.as_poly().total_degree()
    return total_degree == 0 or total_degree == 3


def count_unique_polynomial_lists(poly_lists):
    unique_lists = []
    for poly_list1 in poly_lists:
        is_unique = True
        for poly_list2 in unique_lists:
            if are_polynomials_equal(poly_list1, poly_list2):
                is_unique = False
                break
        if is_unique:
            unique_lists.append(poly_list1)
    return unique_lists


SYMMDICT = {"P": 1, "I": 2, "R": 3, "F": 4, "A": 5, "B": 6, "C": 7}

cif_dir = Path("all_cif")
out_root = Path("all_shelx_file")
copy_cif_into_folder = True

out_root.mkdir(parents=True, exist_ok=True)

cif_files = sorted(cif_dir.glob("*.cif"))
failed = []

for cif_path in tqdm(cif_files, desc="Converting CIF -> INS", unit="file"):
    basename = cif_path.stem
    sample_dir = out_root / basename
    sample_dir.mkdir(parents=True, exist_ok=True)

    if copy_cif_into_folder:
        shutil.copy2(cif_path, sample_dir / cif_path.name)

    ins_path = sample_dir / f"{basename}.ins"

    try:
        structures = iotbx.cif.reader(file_path=str(cif_path)).build_crystal_structures()
        structure = structures.get(basename) or next(iter(structures.values()))

        wavelength = structure.wavelength
        sfac_dict = structure.unit_cell_content()
        cell_params = structure.crystal_symmetry().as_cif_block()

        a = cell_params["_cell.length_a"]
        b = cell_params["_cell.length_b"]
        c = cell_params["_cell.length_c"]
        alpha = cell_params["_cell.angle_alpha"]
        beta = cell_params["_cell.angle_beta"]
        gamma = cell_params["_cell.angle_gamma"]

        operation_xyz = cell_params["_space_group_symop.operation_xyz"]
        spag = cell_params["_space_group.name_H-M_alt"]
        bravis = spag[0]
        is_centric = structure.crystal_symmetry().space_group().is_centric()

        with open(ins_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("TITL\n")

            cell_ins = " ".join(
                [str(wavelength), str(a), str(b), str(c), str(alpha), str(beta), str(gamma)]
            )
            file_obj.write("CELL " + cell_ins + "\n")

            latt_ins = SYMMDICT.get(bravis, 1)
            if not is_centric:
                latt_ins = -latt_ins
            file_obj.write("LATT " + str(latt_ins) + "\n")

            xyz_list = []
            for item in operation_xyz:
                if are_polynomials_equal(item.split(","), "x,y,z".split(",")) or are_polynomials_equal(
                    item.split(","), "-x,-y,-z".split(",")
                ):
                    continue
                xyz_list.append(item.split(","))

            filtered_xyz_list = count_unique_polynomial_lists(xyz_list)
            for item in filtered_xyz_list:
                file_obj.write("SYMM " + ",".join(item) + "\n")

            atomlist = []
            numlist = []
            for key, value in sfac_dict.items():
                atomlist.append(key)
                numlist.append(str(int(value)))

            file_obj.write("SFAC " + " ".join(atomlist) + "\n")
            file_obj.write("UNIT " + " ".join(numlist) + "\n")
            file_obj.write("HKLF 4\n")
            file_obj.write("END ")
    except Exception as exc:
        failed.append((str(cif_path), repr(exc)))

print(f"\nDone. Total: {len(cif_files)}, Failed: {len(failed)}")
if failed:
    print("Failed files (first 10):")
    for path, err in failed[:10]:
        print(" -", path, err)
