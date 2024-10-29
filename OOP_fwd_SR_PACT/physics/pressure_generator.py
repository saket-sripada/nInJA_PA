import numpy as np
import torch
from tqdm import tqdm
import cupy as cp  # for GPU acceleration


class PressureDataGenerator:
    """
    Main class for generating pressure data.
    """

    def __init__(self, phantom_config, aperture, PhysicsParameters, acoustic_noise):
        self.phantom_config = phantom_config
        self.aperture = aperture
        self.PhysicsParameters = PhysicsParameters
        self.noise_model = acoustic_noise
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Original implementations of bipolar and pressure_gen without bells/whistles

    def bipolar_f(self, frequencies, epsilon):
        # calculate bipolar frequency domain signal.
        ctft = np.zeros_like(frequencies, dtype=complex)
        nonzero_freqs = frequencies != 0
        freq_nonzero = frequencies[nonzero_freqs]
        ctft[nonzero_freqs] = (
            -1j
            * self.PhysicsParameters.c0
            / freq_nonzero
            * (
                (epsilon / self.PhysicsParameters.c0)
                * np.cos(2 * np.pi * freq_nonzero * epsilon / self.PhysicsParameters.c0)
                - (1 / (2 * np.pi * freq_nonzero))
                * np.sin(2 * np.pi * freq_nonzero * epsilon / self.PhysicsParameters.c0)
            )
        )
        return ctft

    def generate_pressure_data(self):
        # Generate pressure data for the entire phantom configuration.
        upsampling_ratio = 4
        padded_len = upsampling_ratio * self.PhysicsParameters.N_samples
        freqs = np.fft.fftfreq(padded_len, d=1 / self.aperture.sampling_rate)
        positive_freq_indices = np.where(freqs > 0)[0]
        positive_freqs = freqs[positive_freq_indices]
        min_freq = np.min(freqs)

        superposition_data = np.zeros((self.aperture.N_elements, self.PhysicsParameters.N_samples))

        # Process spheres
        for sphere in tqdm(self.phantom_config.all_sources(), desc="Processing spheres"):
            bipolar_f_values = self.bipolar_f(positive_freqs, sphere.radius)
            bipolar_min_freq = self.bipolar_f(min_freq, sphere.radius)

            for i_meas in range(self.aperture.N_elements):
                hSIR_f_values = self.aperture.hSIR_f(positive_freqs, sphere.position, i_meas)
                hEIR_f_values = self.aperture.hEIR(positive_freqs)
                combined_freq_domain = bipolar_f_values * hEIR_f_values * hSIR_f_values

                complete_combined_values = np.concatenate(
                    (
                        [0],  # For zero frequency
                        combined_freq_domain,
                        np.array([bipolar_min_freq * self.aperture.hSIR_f(min_freq, sphere.position, i_meas)])
                        * self.aperture.hEIR(min_freq),
                        np.conj(combined_freq_domain[::-1]),
                    )
                )

                time_domain_combined_signal = np.fft.ifft(complete_combined_values)

                superposition_data[i_meas, :] += (
                    np.real(time_domain_combined_signal[: self.PhysicsParameters.N_samples])
                    * self.aperture.sampling_rate
                    * sphere.intensity
                )

        if self.noise_model:
            noisy_pressure_data = self.noise_model.add_rf_noise(superposition_data)

        return noisy_pressure_data

    def bipolar_f_cupy(self, frequencies, epsilon):
        frequencies = cp.asarray(frequencies)
        epsilon = cp.asarray(epsilon)

        ctft = cp.zeros_like(frequencies, dtype=cp.complex64)
        nonzero_mask = frequencies != 0
        freq_nonzero = frequencies[nonzero_mask]
        phase = 2 * cp.pi * freq_nonzero * epsilon / self.PhysicsParameters.c0

        ctft[nonzero_mask] = (-1j * self.PhysicsParameters.c0 / freq_nonzero) * (
            ( epsilon / self.PhysicsParameters.c0 ) * cp.cos(phase) - (
                1 / (2 * cp.pi * freq_nonzero)) * cp.sin(phase) 
                )
        return ctft
    
    def generate_pressure_cuvec(self):
        upsampling_ratio = 4
        padded_len = upsampling_ratio * self.PhysicsParameters.N_samples
        freqs = cp.fft.fftfreq(padded_len, d=1 / self.aperture.sampling_rate)
        positive_freq_indices = cp.where(freqs > 0)[0]
        positive_freqs = freqs[positive_freq_indices]
        min_freq = cp.min(freqs)

        superposition_data = cp.zeros((self.aperture.N_elements, self.PhysicsParameters.N_samples))

        for sphere in self.phantom_config.all_sources():
            bipolar_f_values = self.bipolar_f_cupy(positive_freqs, sphere.radius)
            bipolar_min_freq = self.bipolar_f_cupy(min_freq, sphere.radius)

            for i_meas in range(self.aperture.N_elements):
                hSIR_f_values = self.aperture.hSIR_f(positive_freqs, sphere.position, i_meas)
                hEIR_f_values = self.aperture.hEIR(positive_freqs)
                combined_freq_domain = bipolar_f_values * hEIR_f_values * hSIR_f_values

                complete_combined_values = cp.concatenate(
                    (
                        cp.array([0], dtype=cp.complex64),
                        combined_freq_domain,
                        cp.array([bipolar_min_freq * self.aperture.hSIR_f(min_freq, sphere.position, i_meas)])
                        * self.aperture.hEIR(min_freq),
                        cp.conj(combined_freq_domain[::-1]),
                    )
                )

                time_domain_combined_signal = cp.fft.ifft(complete_combined_values)

                superposition_data[i_meas, :] += (
                    cp.real(time_domain_combined_signal[: self.PhysicsParameters.N_samples])
                    * self.aperture.sampling_rate
                    * sphere.intensity
                )

        if self.noise_model:
            noisy_pressure_data = self.noise_model.add_rf_noise(superposition_data)

        return noisy_pressure_data.cpu().numpy()
