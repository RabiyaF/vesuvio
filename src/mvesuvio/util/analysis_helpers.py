import numpy as np
from mantid.kernel import logger


def pseudo_voigt(x, sigma, gamma):
    """Convolution between Gaussian with std sigma and Lorentzian with HWHM gamma"""
    fg, fl = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0)), 2.0 * gamma
    f = 0.5346 * fl + np.sqrt(0.2166 * fl**2 + fg**2)
    eta = 1.36603 * fl / f - 0.47719 * (fl / f) ** 2 + 0.11116 * (fl / f) ** 3
    sigma_v, gamma_v = f / (2.0 * np.sqrt(2.0 * np.log(2.0))), f / 2.0
    pseudo_voigt = eta * lorentzian(x, gamma_v) + (1.0 - eta) * gaussian(x, sigma_v)
    return pseudo_voigt


def gaussian(x, sigma):
    """Gaussian centered at zero"""
    gauss = np.exp(-(x**2) / 2 / sigma**2)
    gauss /= np.sqrt(2.0 * np.pi) * sigma
    return gauss


def lorentzian(x, gamma):
    """Lorentzian centered at zero"""
    return gamma / np.pi / (x**2 + gamma**2)


def numerical_third_derivative(x, y):
    k6 = (-y[:, 12:] + y[:, :-12]) * 1
    k5 = (+y[:, 11:-1] - y[:, 1:-11]) * 24
    k4 = (-y[:, 10:-2] + y[:, 2:-10]) * 192
    k3 = (+y[:, 9:-3] - y[:, 3:-9]) * 488
    k2 = (+y[:, 8:-4] - y[:, 4:-8]) * 387
    k1 = (-y[:, 7:-5] + y[:, 5:-7]) * 1584

    dev = k1 + k2 + k3 + k4 + k5 + k6
    dev /= np.power(x[:, 7:-5] - x[:, 6:-6], 3)
    dev /= 12**3
    return dev


def load_resolution(instrument_params):
    """Resolution of parameters to propagate into TOF resolution
    Output: matrix with each parameter in each column"""
    spectra = instrument_params[:, 0]
    L = len(spectra)
    # For spec no below 135, back scattering detectors, mode is double difference
    # For spec no 135 or above, front scattering detectors, mode is single difference
    dE1 = np.where(spectra < 135, 88.7, 73)  # meV, STD
    dE1_lorz = np.where(spectra < 135, 40.3, 24)  # meV, HFHM
    dTOF = np.repeat(0.37, L)  # us
    dTheta = np.repeat(0.016, L)  # rad
    dL0 = np.repeat(0.021, L)  # meters
    dL1 = np.repeat(0.023, L)  # meters

    resolutionPars = np.vstack((dE1, dTOF, dTheta, dL0, dL1, dE1_lorz)).transpose()
    return resolutionPars


def load_instrument_params(ip_file, spectrum_list):
    first_spec = min(spectrum_list)
    last_spec = max(spectrum_list)
    data = np.loadtxt(ip_file, dtype=str)[1:].astype(float)
    spectra = data[:, 0]

    select_rows = np.where((spectra >= first_spec) & (spectra <= last_spec))
    return data[select_rows]


def extend_range_of_array(arr, n_columns):
    arr = arr.copy()
    left_extend = arr[:, :n_columns] + (arr[:, 0] - arr[:, n_columns]).reshape(-1, 1)
    right_extend = arr[:, -n_columns:] + (arr[:, -1] - arr[:, -n_columns - 1]).reshape(-1, 1)
    return np.concatenate([left_extend, arr, right_extend], axis=-1)


def make_gamma_correction_input_string(masses, mean_widths, mean_intensity_ratios):
    profiles = ""
    for mass, width, intensity in zip(masses, mean_widths, mean_intensity_ratios):
        profiles += "name=GaussianComptonProfile,Mass=" + str(mass) + ",Width=" + str(width) + ",Intensity=" + str(intensity) + ";"
    logger.notice("\nThe sample properties for Gamma Correction are:\n\n" + str(profiles).replace(";", "\n\n").replace(",", "\n"))
    return profiles


def make_multiple_scattering_input_string(masses, meanWidths, meanIntensityRatios):
    atomic_properties_list = np.vstack([masses, meanIntensityRatios, meanWidths]).transpose().flatten().tolist()
    logger.notice(
        "\nSample properties for multiple scattering correction:\n\n"
        + "mass   intensity   width\n"
        + str(np.array(atomic_properties_list).reshape(-1, 3)).replace("[", "").replace("]", "")
        + "\n"
    )
    return atomic_properties_list
