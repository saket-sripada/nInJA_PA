import argparse
import glob
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class Visualizer:
    def __init__(self, phantom_config, aperture):
        self.phantom_config = phantom_config
        self.aperture = aperture
        filename=""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_name = f"./outputs/{timestamp}/"
        os.makedirs(self.folder_name, exist_ok=True)
        #save_path = self.folder_name + filename
        """
        self.args = self.parse_args()

        # folder_name = "./outputs/20240902_191309/"  # Replace with your folder_name
        self.folder_name = self.args.folder

        # If folder name is empty, take the latest folder from outputs/
        if self.folder_name == "":
            self.folder_name = max(glob.glob("./outputs/*"), key=os.path.getctime)
        # Add trailing slash if not present
        if self.folder_name[-1] != "/":
            self.folder_name += "/"
        """


    def parse_args(self):
        parser = argparse.ArgumentParser(description="Visualize the comparison of analytical and reconstructed images")
        parser.add_argument("--folder", type=str, default="", help="Folder name containing the data")
        # Add more arguments as needed
        return parser.parse_args()

    def plot_GT_RF_BF(
            self,
            pressure_data,
            reconstructed_image            
    ):
        label_font_size = 21
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Ground Truth phantom, RF Pressure Traces, Beamformed Image", fontsize=label_font_size)

        radii = [sphere.radius for sphere in self.phantom_config.spheres]
        x_values = [sphere.position[0] for sphere in self.phantom_config.spheres]
        z_values = [sphere.position[2] for sphere in self.phantom_config.spheres]

        xamg, zamg = np.meshgrid(x_values, z_values)
        for i, sphere_radius in enumerate(radii):
            # Scatter plot of sphere positions
            for xGT, zGT in tqdm(zip(xamg.flatten(), zamg.flatten()), total=xamg.size, desc="Plotting spheres"):
                axs[0].scatter(xGT, zGT, c="orange", s=np.pi * (sphere_radius**2) * 4.2e1)
            axs[0].set_xlim(-self.aperture.L / 2, self.aperture.L / 2)
            axs[0].set_ylim(self.aperture.L, 0)
            axs[0].set_title(f"Sphere_radius ={sphere_radius:.3f}")
            axs[0].set_xlabel("Lateral position (mm)")
            axs[0].set_ylabel("Axial position (mm)")

        im2 = axs[1].imshow(pressure_data.T, cmap="gray", aspect="auto")
        axs[1].set_title(f"Analytical Pressure Data, NA_Tx = {self.aperture.a}mm(W), {self.aperture.b}mm(H)")
        plt.colorbar(im2, ax=axs[1], label="Pressure")

        # Reconstructed image (DMAS)
        im4 = axs[2].imshow(reconstructed_image, cmap="hot", aspect="auto",
            extent=[-self.aperture.L / 2, self.aperture.L / 2, self.aperture.L, 0])
        axs[2].set_title(f"DMAS reconstruction of phantom")
        axs[2].set_xlabel("Lateral position (mm)", fontsize=label_font_size)
        axs[2].set_ylabel("Axial position (mm)", fontsize=label_font_size)
        plt.colorbar(im4, ax=axs[2], label="Intensity")

        plt.tight_layout()
        plt.show()

        # save the figure to folder_name
        fig.savefig(self.folder_name + "locsGT_RBF.png")
        #fig.savefig("./outputs/test/locsGT_RBF.png")

    
    
    # below needs lots of fixes
    def plot_comparison(
        self,
        superposition_data,
        reconstructed_images,
    ):

        #'''
        label_font_size = 42
        radii = [sphere.radius for sphere in self.phantom_config.spheres]
        fig, axs = plt.subplots(len(radii), 4, figsize=(25, 15))
        fig.suptitle("Comparison of Analytical and Reconstructed Images", fontsize=label_font_size)

        x_values = [sphere.position[0] for sphere in self.phantom_config.spheres]
        z_values = [sphere.position[2] for sphere in self.phantom_config.spheres]

        xamg, zamg = np.meshgrid(x_values, z_values)

        for i, sphere_radius in enumerate(radii):
            # Scatter plot of sphere positions
            for xGT, zGT in tqdm(zip(xamg.flatten(), zamg.flatten()), total=xamg.size, desc="Plotting spheres"):
                axs[i, 0].scatter(xGT, zGT, c="orange", s=np.pi * (sphere_radius**2) * 4.2e1)
            axs[i, 0].set_xlim(-self.aperture.L / 2, self.aperture.L / 2)
            axs[i, 0].set_ylim(self.aperture.L, 0)
            axs[i, 0].set_title(f"Sphere_radius ={sphere_radius:.3f}")
            axs[i, 0].set_xlabel("Lateral position (mm)")
            axs[i, 0].set_ylabel("Axial position (mm)")

            """
            # Analytical pressure field (point-like transducer)
            im1 = axs[i, 1].imshow(superposition_data[sphere_radius].T,
              cmap="gray", aspect="auto")
            axs[i, 1].set_title("Analytical Pressure Data (Point-like Transducer)")
            plt.colorbar(im1, ax=axs[i, 1], label="Pressure")
            """

            # Analytical pressure field (finite numerical aperture)
            im2 = axs[i, 1].imshow(superposition_data[sphere_radius].T, cmap="gray", aspect="auto")
            axs[i, 1].set_title(f"Analytical Pressure Data, NA_Tx = {self.aperture.a}mm(W), {self.aperture.b}mm(H)")
            plt.colorbar(im2, ax=axs[i, 1], label="Pressure")

            # Reconstructed image (DAS)
            das_key = (sphere_radius, True, "DAS", 0, "None")
            im3 = axs[i, 2].imshow(
                reconstructed_images[das_key],
                cmap="hot",
                aspect="auto",
                extent=[-self.aperture.L / 2, self.aperture.L / 2, self.aperture.L, 0],
            )
            axs[i, 2].set_title(f"Recon : DAS using f#{0}, apo = None")
            axs[i, 2].set_xlabel("Lateral position (mm)", fontsize=label_font_size)
            axs[i, 2].set_ylabel("Axial position (mm)", fontsize=label_font_size)
            plt.colorbar(im3, ax=axs[i, 2], label="Intensity")

            # Reconstructed image (DMAS)
            dmas_key = (sphere_radius, True, "DMAS", 0, "None")
            im4 = axs[i, 3].imshow(
                reconstructed_images[dmas_key],
                cmap="hot",
                aspect="auto",
                extent=[-self.aperture.L / 2, self.aperture.L / 2, self.aperture.L, 0],
            )
            axs[i, 3].set_title(f"Recon : DMAS using f#{0}, apo = None")
            axs[i, 3].set_xlabel("Lateral position (mm)", fontsize=label_font_size)
            axs[i, 3].set_ylabel("Axial position (mm)", fontsize=label_font_size)
            plt.colorbar(im4, ax=axs[i, 3], label="Intensity")

        plt.tight_layout()
        plt.show()

        # save the figure to folder_name
        fig.savefig(self.folder_name + "comparison.png")

        unique_keys = set(reconstructed_images.keys())
        unique_radius = sorted(set(key[0] for key in unique_keys))
        unique_is_sir = sorted(set(key[1] for key in unique_keys))
        unique_bf = sorted(set(key[2] for key in unique_keys))
        unique_fnum = sorted(set(key[3] for key in unique_keys))
        unique_apod_mode = sorted(set(key[4] for key in unique_keys))
        #'''

        #'''
        for radius in unique_radius:
            for is_sir in unique_is_sir:
                for bf in unique_bf:
                    plt.figure(figsize=(42, 21))
                    i = 0
                    for apod_mode in unique_apod_mode:
                        for fnum in unique_fnum:
                            i += 1
                            key = (radius, bf, fnum, apod_mode, is_sir)

                            if key not in reconstructed_images:
                                continue

                            value = reconstructed_images[key]

                            plt.subplot(len(unique_apod_mode), len(unique_fnum), i)
                            plt.imshow(
                                value,
                                aspect="auto",
                                cmap="hot",
                                extent=[-self.aperture.L / 2, self.aperture.L / 2, self.aperture.L, 0],
                            )
                            plt.colorbar(label="Intensity")
                            plt.title(f"f# {fnum} , apod : {apod_mode}", fontsize=label_font_size)
                            plt.xlabel("Lateral position (mm)", fontsize=label_font_size)
                            plt.ylabel("Axial position (mm)", fontsize=label_font_size)
                            if is_sir:
                                plt.suptitle(
                                    f"Center Freq = 8MHz +/-3dB, BF: {bf}, radius: {radius}mm, NumAperture_Transducer = {a}mm(W), {b}mm(H)",
                                    fontsize=2 * label_font_size,
                                )
                            """
                            else:
                                plt.suptitle(
                                    f"Center Freq = 8MHz +/-3dB, , BF: {bf},
                                      radius: {radius}mm, point-Transducer",
                                    fontsize=2 * label_font_size,
                                )
                                """

                    plt.tight_layout()
                    plt.show()

                    # save the figure to folder_name with appropriate name
                    plt.savefig(self.folder_name + f"radius_{radius}_sir_{is_sir}_bf_{bf}.png")
                    #'''
