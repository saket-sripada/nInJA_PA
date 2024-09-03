import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fwd_model.io import load_data_h5

label_font_size = 21


def visualize_comparison(
    folder_name,
    superposition_data,
    reconstructed_images,
    sphere_radii,
    x_positions,
    z_positions,
    L,
    a,
    b,
):
    fig, axs = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle("Comparison of Analytical and Reconstructed Images", fontsize=label_font_size)

    xamg, zamg = np.meshgrid(x_positions, z_positions)

    for i, sphere_radius in enumerate(sphere_radii):
        # Scatter plot of sphere positions
        for xGT, zGT in tqdm(zip(xamg.flatten(), zamg.flatten()), total=xamg.size):
            axs[i, 0].scatter(xGT, zGT, c="orange", s=np.pi * (sphere_radius**2) * 4.2e1)
        axs[i, 0].set_xlim(-L / 2, L / 2)
        axs[i, 0].set_ylim(L, 0)
        axs[i, 0].set_title(f"Spheres (r={sphere_radius:.3f})")
        axs[i, 0].set_xlabel("Lateral position (mm)")
        axs[i, 0].set_ylabel("Axial position (mm)")

        # Analytical pressure field (point-like transducer)
        im1 = axs[i, 1].imshow(superposition_data[(sphere_radius, False)].T, cmap="gray", aspect="auto")
        axs[i, 1].set_title("Analytical Pressure Data (Point-like Transducer)", fontsize=label_font_size)
        plt.colorbar(im1, ax=axs[i, 1], label="Pressure")

        # Analytical pressure field (finite numerical aperture)
        im2 = axs[i, 2].imshow(superposition_data[(sphere_radius, True)].T, cmap="gray", aspect="auto")
        axs[i, 2].set_title(f"Analytical Pressure Data, NA_Tx = {a}mm(W), {b}mm(H)", fontsize=label_font_size)
        plt.colorbar(im2, ax=axs[i, 2], label="Pressure")

        # Reconstructed image (DAS)
        das_key = (sphere_radius, True, "DAS", 0, "None")
        im3 = axs[i, 3].imshow(reconstructed_images[das_key], cmap="hot", aspect="auto", extent=[-L / 2, L / 2, L, 0])
        axs[i, 3].set_title(f"Recon : DAS using f#{0}, apo = None", fontsize=label_font_size)
        axs[i, 3].set_xlabel("Lateral position (mm)", fontsize=label_font_size)
        axs[i, 3].set_ylabel("Axial position (mm)", fontsize=label_font_size)
        plt.colorbar(im3, ax=axs[i, 3], label="Intensity")

        # Reconstructed image (DMAS)
        dmas_key = (sphere_radius, True, "DMAS", 0, "None")
        im4 = axs[i, 4].imshow(reconstructed_images[dmas_key], cmap="hot", aspect="auto", extent=[-L / 2, L / 2, L, 0])
        axs[i, 4].set_title("DMAS Reconstruction")
        axs[i, 4].set_xlabel("Lateral position (mm)", fontsize=label_font_size)
        axs[i, 4].set_ylabel("Axial position (mm)", fontsize=label_font_size)
        plt.colorbar(im4, ax=axs[i, 4], label="Intensity")

    plt.tight_layout()
    plt.show()

    # save the figure to folder_name
    fig.savefig(folder_name + "comparison.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize the comparison of analytical and reconstructed images")
    parser.add_argument("--folder", type=str, default="", help="Folder name containing the data")
    # Add more arguments as needed
    return parser.parse_args()


def main():
    args = parse_args()

    # folder_name = "./outputs/20240902_191309/"  # Replace with your folder_name
    folder_name = args.folder

    # If folder name is empty, take the latest folder from outputs/
    if folder_name == "":
        folder_name = max(glob.glob("./outputs/*"), key=os.path.getctime)

    # Add trailing slash if not present
    if folder_name[-1] != "/":
        folder_name += "/"

    reconstructed_images, superposition_data, a, b = load_data_h5(folder_name)

    # plotting params
    # sphere_radii = [1.542, 0.42, 0.21]  # Your sphere radii
    # Get unique sphere radii
    sphere_radii = sorted(set(key[0] for key in reconstructed_images.keys()))

    # Assume L is the maximum depth in your images
    L = reconstructed_images[list(reconstructed_images.keys())[0]].shape[1]
    delL = L / 3  # 3 = num_spheres along axis of transducer

    x_positions = np.arange(-L / 2 + delL / 2, L / 2, delL)
    z_positions = np.arange(L / 4, 4 * L / 5, L / 4)  # 10mm to 31mm at 10mm steps

    visualize_comparison(
        folder_name,
        superposition_data,
        reconstructed_images,
        sphere_radii,
        x_positions,
        z_positions,
        L,
        a,
        b,
    )


if __name__ == "__main__":
    main()
