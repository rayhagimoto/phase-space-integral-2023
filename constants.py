import numpy as np

# Masses in MeV
me = 0.511
mp = 0.8 * 938.27208816
mu = 106.0
mn = 940.6

# Conversion factor from MeV^5 to ergs / cm^3 / s
CONVERSION_FACTOR = (3.156565344122207e-48) ** -1

# Default parameters
DEFAULT_VALUES = dict(
    T=0.0861733,  # MeV (=10^9 K)
    process="ep->upa",
    m3=0,
    beta_F_mu=0.836788,
)


def _check_process(process):
    assert len(process) == len("ep->upa")
    _in, _out = process.split("->")


def get_fermi_params(
    process,
    beta_F_mu,
):
    """Get Fermi momenta using some assumptions such as charge conservation."""
    _check_process(process)
    ma, mb, m1, m2 = get_masses(process)

    # ---- l p -> l' p a ----- #
    if process == "ep->upa":
        pF1 = m1 * beta_F_mu / (1 - beta_F_mu**2) ** 0.5
        mu_1 = np.sqrt(pF1**2 + m1**2)
        mu_a = mu_1
        pFa = np.sqrt(mu_a**2 - ma**2)
        pFb = (pFa**3 + pF1**3) ** (1 / 3)
        mu_b = np.sqrt(pFb**2 + mb**2)
        mu_2 = mu_b
    if process == "up->epa":
        pFa = ma * beta_F_mu / (1 - beta_F_mu**2) ** 0.5
        mu_a = (pFa**2 + ma**2) ** 0.5
        mu_1 = mu_a
        pF1 = (mu_1**2 - m1**2) ** 0.5
        pFb = (pFa**3 + pF1**3) ** (1 / 3)
        mu_b = (pFb**2 + mb**2) ** 0.5
        mu_2 = mu_b

    # ---- l l -> l' l a ----- #
    if process == "ee->uea":
        pF1 = m1 * beta_F_mu / (1 - beta_F_mu**2) ** 0.5
        mu_1 = (pF1**2 + m1**2) ** 0.5
        mu_a = mu_1
        pFa = (mu_a**2 - ma**2) ** 0.5
        pFb = pFa
        mu_b = mu_a
        mu_2 = mu_b
    if process == "uu->eua":
        pFa = ma * beta_F_mu / (1 - beta_F_mu**2) ** 0.5
        mu_a = (pFa**2 + ma**2) ** 0.5
        mu_1 = mu_a
        pF1 = (mu_1**2 - m1**2) ** 0.5
        pFb = pFa
        mu_b = mu_a
        mu_2 = mu_b

    # ---- l l' -> l' l' a ----- #
    if process == "eu->uua":
        pF1 = m1 * beta_F_mu / (1 - beta_F_mu**2) ** 0.5
        mu_1 = (pF1**2 + m1**2) ** 0.5
        mu_a = mu_1
        pFa = (mu_a**2 - ma**2) ** 0.5
        pFb = pF1
        mu_b = mu_1
        mu_2 = mu_b
    if process == "ue->eea":
        pFa = ma * beta_F_mu / (1 - beta_F_mu**2) ** 0.5
        mu_a = (pFa**2 + ma**2) ** 0.5
        mu_1 = mu_a
        pF1 = (mu_a**2 - ma**2) ** 0.5
        pFb = pF1
        mu_b = mu_1
        mu_2 = mu_b

    return pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2


# Processes
def get_masses(process):
    """
    Parameters
    ----------
    process : str
        must be one of the following:
        lp->l'pa
        ll->l'la
        ll'->l'l'a

        where l is either e or u and l' is always 'the other one'. E.g. if l is e then l' is u.
    """
    masses = {
        "e": me,
        "p": mp,
        "u": mu,
        "n": mn,
    }
    _check_process(process)
    _in, _out = process.split("->")

    ma = masses[_in[0]]
    mb = masses[_in[1]]

    m1 = masses[_out[0]]
    m2 = masses[_out[1]]

    return ma, mb, m1, m2
