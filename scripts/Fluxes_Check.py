import glob

import numpy as np

files = glob.glob("data/fluxes_bls/*.npz")

print(len(files))

bad = 0

bad_files = []
for f in files:
    try:
        with np.load(f) as d:
            if d["global_view"].shape[0] != 2001 or np.isnan(d["global_view"]).any() or d["local_view"].shape[0] != 201:
                bad += 1
                bad_files.append(f)
    except Exception as e: # noqa: BLE001
        bad += 1
        bad_files.append(f)
        print(f"Error occurred while processing file {f}: {e}")

#for f in bad_files:
#    os.remove(f)

print(f"{len(bad_files)} bad files")