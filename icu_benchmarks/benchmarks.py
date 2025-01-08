import numpy as np


def severinghaus_spo2_to_po2_from_table(spo2):
    """pO2 imputation from SpO2 using Table 1 of Serveringhaus (1979)."""
    # fmt: off
    po2_grid = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 175, 200, 225, 250, 300, 400, 500])
    spo2_grid = np.array([0.60, 1.19, 2.56, 4.37, 6.68, 9.58, 12.96, 16.89, 21.40, 26.50, 32.12, 37.60, 43.14, 48.27, 53.16, 57.54, 61.69, 65.16, 68.63, 71.94, 74.69, 77.29, 79.55, 81.71, 83.52, 85.08, 86.59, 87.70, 88.93, 89.95, 90.85, 92.73, 94.06, 95.10, 95.84, 96.42, 96.88, 97.25, 97.49, 97.91, 98.21, 98.44, 98.62, 98.77, 99.03, 99.20, 99.32, 99.41, 99.53, 99.65, 99.72])
    # fmt: on
    return np.interp(spo2, spo2_grid / 100, po2_grid)


def severinghaus_spo2_to_po2(spo2, temp=None, ph=None):
    """
    pO2 imputation from SpO2 of Severinghaus (1979) and Ellis.

    In their original paper "Simple, accurate equations for human blood O2 dissociation
    computations", Severinghaus only gives the equation

    S = [23,000 * (PO2^3 + 150 * PO2)]^{-1} + 1]^{-1}

    and Table 1 (see below) with SpO2 values and corresponding PO2 values.

    In "Determination of POD from saturation" Roger K. Ellis yields an inversion of the
    above equation as

    PO2 = { 11,700 / (1 / S - 1) + [50^3 + (11,700 / (1 / S - 1))^2]^{1/2} }^{1/3} +
            { 11,700 / (1 / S - 1) - [50^3 + (11,700 / (1 / S - 1))^2]^{1/2} }^{1/3}

    which is the equation used here. That paper by Ellis also includes a reply by
    Severinghaus with the ph and temperature correction factors.

    Severinghaus (1979): http://www.nickalls.org/dick/papers/anes/JWSrevised2007.pdf
    Ellis: https://journals.physiology.org/doi/epdf/10.1152/jappl.1989.67.2.902

    Notably, the applying Severinghaus' equation to the PO2 values below does not yield
    the table from Severinghaus paper. Especially for larger values of spo2, the results
    are off.

    The same holds in the opposite direction. E.g., PO2(0.9972) ~ 202.

    Table 1:
    PO2 %Sat PO2 %Sat PO2 %Sat
    1 0.60 34 65.16 80 95.84
    2 1.19 36 68.63 85 96.42
    4 2.56 38 71.94 90 96.88
    6 4.37 40 74.69 95 97.25
    8 6.68 42 77.29 100 97.49
    10 9.58 44 79.55 110 97.91
    12 12.96 46 81.71 120 98.21
    14 16.89 48 83.52 130 98.44
    16 21.40 50 85.08 140 98.62
    18 26.50 52 86.59 150 98.77
    20 32.12 54 87.70 175 99.03
    22 37.60 56 88.93 200 99.20
    24 43.14 58 89.95 225 99.32
    26 48.27 60 90.85 250 99.41
    28 53.16 65 92.73 300 99.53
    30 57.54 70 94.06 400 99.65
    32 61.69 75 95.10 500 99.72
    """
    inner = spo2 * 11_700 / (1 - spo2)
    po2 = np.pow(inner + np.sqrt(125_000 + inner**2), 1 / 3) - np.pow(
        -inner + np.sqrt(125_000 + inner**2), 1 / 3
    )
    if temp is not None:
        factor = 0.058 * (1 / (0.243 * np.pow(0.01 * po2, 3.88) + 1)) + 0.013
        po2 *= np.exp((temp - 37) * factor)
    if ph is not None:
        factor = np.pow(po2 / 26.6, 0.184) - 2.2
        po2 *= np.exp((ph - 7.4) * factor)
    return po2


def severinghaus_po2_to_spo2(po2):
    """SpO2 imputation from pO2 using the equation of Severinghaus (1979)."""
    return 1 / (23_400 / (po2**3 + 150 * po2) + 1)
