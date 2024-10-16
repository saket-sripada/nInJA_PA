import os
from datetime import datetime

import h5py


def save_data_h5(folder_name, pressure_data, reconstructed_images):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(folder_name, timestamp)
    os.makedirs(output_folder, exist_ok=True)

    # Save pressure data
    with h5py.File(os.path.join(output_folder, "pressure_data.h5"), "w") as f:
        f.create_dataset("pressure_data", data=pressure_data, compression="gzip")

    # Save reconstructed images
    with h5py.File(os.path.join(output_folder, "reconstructed_images.h5"), "w") as f:
        for key, value in reconstructed_images.items():
            dataset_name = f"recon_{key[0]}_{key[1]}_{key[2]}_{key[3]}"
            f.create_dataset(dataset_name, data=value, compression="gzip")

    print(f"Data saved to {output_folder}")
