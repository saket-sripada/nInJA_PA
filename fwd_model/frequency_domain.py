import numpy as np

from fwd_model.utils import r0, x_i


# Frequency domain of bipolar signal
# This is based on Equation (9) in the paper
def bipolar_f(frequencies, epsilon, c0):
    ctft = np.zeros_like(frequencies, dtype=complex)
    nonzero_freqs = frequencies != 0
    freq_nonzero = frequencies[nonzero_freqs]
    ctft[nonzero_freqs] = (
        -1j
        * c0
        / freq_nonzero
        * (
            (epsilon / c0) * np.cos(2 * np.pi * freq_nonzero * epsilon / c0)
            - (1 / (2 * np.pi * freq_nonzero)) * np.sin(2 * np.pi * freq_nonzero * epsilon / c0)
        )
    )
    return ctft


# Frequency domain of SIR blurring kernel for rectangular linear array
# This is based on Equation (10) in the paper, adapted for rectangular geometry
def hSIR_f(frequencies, a, b, c0, sphere_positions, i, is_sir):

    rn = sphere_positions
    xn, yn, zn = rn[0], rn[1], rn[2]

    # Compute x_nq and y_nq for rectangular geometry
    # This replaces the spherical coordinate calculations in the original paper
    x_nq = xn - x_i(i)
    y_nq = yn

    # Distance term
    r_qn = np.linalg.norm(rn - r0(i))

    # Compute the SIR in the frequency domain
    # This is the far-field approximation from Equation (10)
    sinc_x = np.sinc(a * frequencies * x_nq / (c0 * r_qn))
    sinc_y = np.sinc(b * frequencies * y_nq / (c0 * r_qn))
    if is_sir:
        h_qs = a * b * np.exp(-1j * 2 * np.pi * frequencies * r_qn / c0) / (2 * np.pi * r_qn) * sinc_x * sinc_y
    else:
        h_qs = a * b * np.exp(-1j * 2 * np.pi * frequencies * r_qn / c0) / (2 * np.pi * r_qn)
    return h_qs


def hEIR(frequencies, center_freq=8e6, bandwidth=3e6):
    # kwave uses a finite impulse repsonse tht is a gaussian in freq domain
    # learn to write math as code bro.. please learn spend time
    # read kwave documentation, simpa docs, ipasc docs, pacfish docs?
    """
    Generate the electrical impulse response in the frequency domain.

    Parameters:
    - frequencies: Array of frequencies
    - center_freq: Center frequency of the Gaussian (default 8 MHz)
    - bandwidth: Bandwidth (±3 dB) of the Gaussian (default 3 MHz)

    Returns:
    - EIR frequency response
    """
    # Create Gaussian frequency response
    sigma = bandwidth / (2 * np.sqrt(2 * np.log(2)))  # Convert bandwidth to standard deviation
    gaussian_response = np.exp(-((frequencies - center_freq) ** 2) / (2 * sigma**2))
    # Normalize the response
    gaussian_response /= np.max(gaussian_response)
    return gaussian_response
