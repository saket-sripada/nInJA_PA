import os
from datetime import datetime

import h5py
from tqdm import tqdm


def save_data_h5(reconstructed_images, superposition_pressure, filename=""):
    # drive.mount('/content/drive')
    # save_path = '/content/drive/My Drive/' + filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"./outputs/{timestamp}/"
    os.makedirs(folder_name, exist_ok=True)
    save_path = folder_name + filename + "recos.h5"

    with h5py.File(save_path, "w") as f:
        for key, value in tqdm(reconstructed_images.items(), desc="Saving recon imgs"):
            dataset_name = f"sphere_{key[0]}_sir_{key[1]}_bf_{key[2]}_fnumber_{key[3]}_apod_{key[4]}"
            f.create_dataset(dataset_name, data=value, compression="gzip")

    save_path = folder_name + filename + "superpos.h5"
    with h5py.File(save_path, "w") as f:
        for key, value in tqdm(superposition_pressure.items(), desc="Saving superposition pressure"):
            dataset_name = f"sphere_{key[0]}_sir_{key[1]}"
            f.create_dataset(dataset_name, data=value, compression="gzip")

    print(f"Data saved to {save_path}")


def load_reconstructed_images(filename):
    file_path = filename

    reconstructed_images = {}
    superposition_data = {}

    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            # Parse the key
            parts = key.split("_")
            print(parts)
            radius = float(parts[1])
            is_sir = parts[3] == "True"
            # bf = parts[5]
            bf = parts[9]
            fnumber = float(parts[5])
            apod = parts[7]

            # Load the dataset
            data = f[key][()]

            # Store in the appropriate dictionary
            if bf in ["DAS", "DMAS"]:
                reconstructed_images[(radius, is_sir, bf, fnumber, apod)] = data
            else:
                superposition_data[(radius, is_sir)] = data

    print(f"Data loaded from {file_path}")
    return reconstructed_images, superposition_data
