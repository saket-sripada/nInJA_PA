import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import nakagami


class AcousticNoise:
    def __init__(self, snr_db, speckle_std, noise_type):
        self.snr_db = snr_db
        self.speckle_std = speckle_std
        self.noise_type = noise_type

    def add_rf_noise(self, pressure_data):
        depth_indices = np.arange(pressure_data.shape[0])
        """Add depth-dependent noise to RF pressure data"""
        # Depth-dependent SNR scaling
        depth_factor = 1 + 0.1 * (depth_indices / depth_indices.max())
        base_signal_power = np.mean(pressure_data**2)
        # Two-component Gaussian mixture for more realistic noise
        noise_power = base_signal_power / (10 ** (self.snr_db / 10))

        if self.noise_type == "gaussian_mixture":
            noise1 = np.random.normal(0, np.sqrt(0.7 * noise_power), pressure_data.shape)
            noise2 = np.random.normal(0, np.sqrt(0.3 * noise_power), pressure_data.shape)
            noise = (noise1 + noise2) * depth_factor[:, np.newaxis]

        elif self.noise_type == "nakagami":
            # Nakagami noise model for tissue-like scattering
            shape_m = 1.0  # Shape parameter
            noise = nakagami.rvs(shape_m, scale=np.sqrt(noise_power), size=pressure_data.shape)

        return pressure_data + noise

    def add_post_beamforming_noise(self, reconstructed_image):
        """Add realistic post-beamforming noise"""
        # Combine speckle and electronic noise
        m = 1 / self.speckle_std
        speckle = nakagami.rvs(m, scale=1.0, size=reconstructed_image.shape)

        # Add depth-dependent electronic noise
        z_positions = np.arange(reconstructed_image.shape[0])
        depth_factor = 1 + 0.05 * (z_positions / z_positions.max())
        electronic_noise = np.random.normal(0, 0.01, reconstructed_image.shape)
        electronic_noise *= depth_factor[:, np.newaxis]

        return reconstructed_image * speckle + electronic_noise

    def apply_system_effects(self, image_data):
        """Apply realistic system effects"""
        # Apply frequency-dependent attenuation
        freq_att = np.exp(-0.2 * np.arange(image_data.shape[0])[:, np.newaxis] / image_data.shape[0])

        # Add lateral resolution degradation with depth
        sigma = 0.2 + 0.8 * np.arange(image_data.shape[0])[:, np.newaxis] / image_data.shape[0]
        for i in range(image_data.shape[0]):
            image_data[i, :] = gaussian_filter1d(image_data[i, :], sigma[i, 0])

        return image_data * freq_att
