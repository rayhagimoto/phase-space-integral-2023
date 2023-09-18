import numpy as np

# Conversion factor from MeV^5 to ergs / cm^3 / s
CONVERSION_FACTOR = (3.156565344122207e-48) ** -1

# Default parameters
# These imply beta_F_mu is approximately 0.836788
DEFAULT_VALUES = dict(
    T=0.0861733,  # MeV (=10^9 K)
    ma=0.511,  # MeV (electron)
    mb=0.8 * 938.27208816,  # MeV (proton)
    m1=106.0,  # MeV (muon)
    pFa=193.0,  # MeV (electron)
    pFb=226.0,  # MeV (proton)
    pF1=162.0,  # MeV (muon)
)


def get_input_params(
    T,
    ma,
    mb,
    m1,
    pFa=None,
    pFb=None,
    pF1=None,
    beta_F_mu=None,
):
    m2 = mb  # (proton)

    if pFa and pFb and pF1:
        if beta_F_mu:
            import warnings

            warnings.warn(
                "both (pFa, pFb, pF1) and beta_F_mu inputs have been detected. Prioritising (pFa, pFb, pF1) and ignoring beta_F_mu."
            )
        mu_a = np.sqrt(pFa**2 + ma**2)
        mu_b = np.sqrt(pFb**2 + mb**2)
        mu_1 = np.sqrt(pF1**2 + m1**2)
        mu_2 = mu_b
    elif beta_F_mu:
        pF1 = m1 * beta_F_mu / np.sqrt(1 - beta_F_mu**2)
        mu_1 = np.sqrt(pF1**2 + m1**2)
        mu_a = mu_1
        pFa = np.sqrt(mu_a**2 - ma**2)
        pFb = (pFa**3 + pF1**3) ** (1 / 3)
        mu_b = np.sqrt(pFb**2 + mb**2)
        mu_2 = mu_b
    else:
        raise Exception(
            "Either pFa, pFb, and pF1 must be supplied or beta_F_mu must be supplied. One of these conditions has not been met."
        )
    return T, ma, mb, m1, m2, pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2
