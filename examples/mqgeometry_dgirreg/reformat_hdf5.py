import h5py
import numpy as np
from tqdm import tqdm
from helmholtz_spectra.utils import convert_to_batched_hdf5


old_h5_path = "neumann.evec.h5"
new_h5_path = "neumann.evec.batched.h5"
convert_to_batched_hdf5(old_h5_path, new_h5_path)
print(f"Converted {old_h5_path} to batched format in {new_h5_path}")

old_h5_path = "dirichlet.evec.h5"
new_h5_path = "dirichlet.evec.batched.h5"
convert_to_batched_hdf5(old_h5_path, new_h5_path)
print(f"Converted {old_h5_path} to batched format in {new_h5_path}")