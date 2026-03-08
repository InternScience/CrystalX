from iotbx.reflection_file_reader import any_reflection_file
data_dir = 'real_xrd_dataset'
file_list = [f.name for f in os.scandir(data_dir) if f.is_dir()]
for fname in file_list:
    path_name = f'real_xrd_dataset/{fname}/{fname}.hkl'
    file_path = f'real_xrd_dataset/{fname}/{fname}.hkl'
    hkl_file = any_reflection_file(path_name)
    miller = hkl_file.as_miller_arrays()
    real_f = miller[1]
    with open(file_path, 'w') as f:
        real_f.export_as_shelx_hklf(f)
