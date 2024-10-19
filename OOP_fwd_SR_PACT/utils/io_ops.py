import os
from datetime import datetime

import h5py


def save_data_h5(folder_name, pressure_data, recon_params):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(folder_name, timestamp)
    os.makedirs(output_folder, exist_ok=True)

    # Save pressure data
    with h5py.File(os.path.join(output_folder, "pressure_data.h5"), "w") as f:
        f.create_dataset("pressure_data", data=pressure_data, compression="gzip")

    # Save reconstructed images
    with h5py.File(os.path.join(output_folder, "reconstructed_images.h5"), "w") as f:
        for attr_name, attr_value in vars(recon_params).items():
            dataset_name = f"recon_{attr_name[0]}_{attr_name[1]}_{attr_name[2]}_{attr_name[3]}"
            f.create_dataset(dataset_name, data=attr_value, compression="gzip")

    print(f"Data saved to {output_folder}")
