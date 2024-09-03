import numpy as np
from tqdm import tqdm

from config import (
    L,
    N_elements,
    N_samples,
    a,
    b,
    c0,
    dt,
    sampling_rate,
    x_positions,
    y_positions,
    z_positions,
)
from fwd_model.frequency_domain import bipolar_f, hEIR, hSIR_f


def generate_pressure_data(sphere_radius, is_sir):
    superposition_data = {}
    # This is the main loop that implements the frequency domain image model calculations in Section 2.2
    upsampling_ratio = 4
    padded_len = upsampling_ratio * N_samples
    freqs = np.fft.fftfreq(padded_len, d=1 / sampling_rate)
    positive_freq_indices = np.where(freqs > 0)[0]
    positive_freqs = freqs[positive_freq_indices]

    # Pre-computation of the bipolar frequency signal
    # This corresponds to the frequency domain calculations in Section 2.2
    bipolar_f_values = bipolar_f(positive_freqs, sphere_radius, c0)
    min_freq = np.min(freqs)
    bipolar_min_freq = bipolar_f(min_freq, sphere_radius, c0)

    superposition_data[(sphere_radius, is_sir)] = np.zeros((N_elements, N_samples))
    xamg, yamg, zamg = np.meshgrid(x_positions, y_positions, z_positions)
    for x, y, z in tqdm(zip(xamg.flatten(), yamg.flatten(), zamg.flatten()), total=xamg.size):
        sphere_positions = np.array((x, y, z))
        analytical_data = np.zeros((N_elements, N_samples))
        for i_meas in range(N_elements):
            # Multiplication with SIR in frequency domain
            hSIR_f_values = hSIR_f(positive_freqs, a, b, c0, sphere_positions, i_meas, is_sir)
            hEIR_f_values = hEIR(positive_freqs, center_freq=8e6, bandwidth=3e6)
            combined_freq_domain = bipolar_f_values * hEIR_f_values * hSIR_f_values

            # Fill frequency spectrum
            complete_combined_values = np.concatenate(
                (
                    [0],  # For zero frequency
                    combined_freq_domain,
                    np.array([bipolar_min_freq * hSIR_f(min_freq, a, b, c0, sphere_positions, i_meas, is_sir)])
                    * hEIR(np.array([min_freq])),
                    np.conj(combined_freq_domain[::-1]),
                )
            )

            # Perform the inverse FFT to transform to the time domain
            time_domain_combined_signal = np.fft.ifft(complete_combined_values)
            analytical_data[i_meas, :] += np.real(time_domain_combined_signal[:N_samples]) / dt / a / b
        superposition_data[(sphere_radius, is_sir)] += analytical_data

    return superposition_data[(sphere_radius, is_sir)]
