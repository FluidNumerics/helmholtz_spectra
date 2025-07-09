import h5py
import numpy as np
from tqdm import tqdm

def convert_to_batched_hdf5(old_h5_path, new_h5_path, batch_size=256):
    with h5py.File(old_h5_path, "r") as f_in, h5py.File(new_h5_path, "w") as f_out:
        _keys = [k for k in f_in.keys() if k.startswith("Xr")]
        tag = "_".join(_keys[0].split("_")[1:])
        keys = [f"Xr{k}_{tag}" for k in range(len(_keys))]

        n_modes = len(keys)
        n_points = f_in[keys[0]].shape[0]

        dset = f_out.create_dataset(
            "eigenmodes",
            shape=(n_modes, n_points),
            dtype=f_in[keys[0]].dtype,
            chunks=(64, n_points),
            compression="gzip",
            compression_opts=4
        )

        print(f"Converting {old_h5_path} to batched format with {n_modes} modes and {n_points} points per mode.")
        # preallocate numpy array
        batch_data = np.empty((batch_size, n_points), dtype=f_in[keys[0]].dtype)
        for i in tqdm(range(0, n_modes, batch_size)):
            ubound = min(i + batch_size, n_modes)
            n_modes_in_batch = ubound - i
            batch_keys = keys[i:ubound]
            for j, k in enumerate(batch_keys):
                batch_data[j, :] = f_in[k][:]
            # Write all at once
            dset[i:ubound, :] = batch_data[0:n_modes_in_batch, :]


def main():
    old_h5_path = "neumann.evec.h5"
    new_h5_path = "neumann.evec.batched.h5"
    convert_to_batched_hdf5(old_h5_path, new_h5_path)
    print(f"Converted {old_h5_path} to batched format in {new_h5_path}")

    old_h5_path = "dirichlet.evec.h5"
    new_h5_path = "dirichlet.evec.batched.h5"
    convert_to_batched_hdf5(old_h5_path, new_h5_path)
    print(f"Converted {old_h5_path} to batched format in {new_h5_path}")

if __name__ == "__main__":
    main()
