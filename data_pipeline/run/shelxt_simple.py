from pathlib import Path
import random
import subprocess

from tqdm import tqdm


root_dir = Path("all_test_shelx")
INPUT_EXT = ".ins"
RES_SUFFIX = "_a.res"
executable = Path("./shelxt").resolve()

todo = []

for subdir in sorted(root_dir.iterdir()):
    if not subdir.is_dir():
        continue

    cod_id = subdir.name
    ins_path = subdir / f"{cod_id}{INPUT_EXT}"
    res_path = subdir / f"{cod_id}{RES_SUFFIX}"

    if ins_path.is_file() and not res_path.exists():
        todo.append(str(subdir / cod_id))

print("Pending cases:", len(todo))

random.seed(6910)
random.shuffle(todo)

for prefix in tqdm(todo, desc="Running SHELXT"):
    subprocess.run([str(executable), prefix], check=False)
