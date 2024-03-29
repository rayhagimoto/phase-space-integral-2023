import numpy as np

# Masses in MeV
me = 0.511 # MeV
mp = 0.8 * 938.27208816 # MeV
mu = 106.0 # MeV 
mn = 940.6 # MeV

# Fine structure constant
alpha_EM = 1 / 137.0

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
    assert len(_in) == 2
    assert len(_out) == 3


def get_fermi_params(
    process,
    beta_F_mu,
):
    """Get Fermi momenta using some assumptions such as charge conservation."""
    _check_process(process)
    ma, mb, m1, m2 = get_masses(process)

    pFu = mu * beta_F_mu / (1 - beta_F_mu**2) ** 0.5
    EFu = (pFu**2 + mu**2) ** 0.5
    EFe = EFu
    pFe = (EFe**2 - me**2) ** 0.5
    pFp = (pFu**3 + pFe**3) ** (1 / 3)
    EFp = (pFp**2 + mp**2) ** 0.5

    # ---- l p -> l' p a ----- #
    if process == "ep->upa":
        pF1 = m1 * beta_F_mu / (1 - beta_F_mu**2) ** 0.5  # muon
        mu_1 = np.sqrt(pF1**2 + m1**2)  # muon chem. pot.
        mu_a = mu_1  # muon chem. pot. = electron chem. pot.
        pFa = np.sqrt(mu_a**2 - ma**2)  # electron Fermi momentum
        pFb = (pFa**3 + pF1**3) ** (
            1 / 3
        )  # charge conservation --> proton Fermi momentum
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

    return pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2, pFe, pFu, pFp, EFe, EFu, EFp


def get_fermi_params_delta(process, beta_F_mu, Delta, T):
    """
    This function should be commented out most of the time. Its only purpose
    is to make sure that EFe and EFu are different.
    """
    _check_process(process)

    assert process in ["ep->upa", "up->epa"]
    ma, mb, m1, m2 = get_masses(process)

    pFu = mu * beta_F_mu / (1 - beta_F_mu**2) ** 0.5
    EFu = (pFu**2 + mu**2) ** 0.5
    EFe = EFu + Delta * T
    pFe = (EFe**2 - me**2) ** 0.5
    pFp = (pFu**3 + pFe**3) ** (1 / 3)
    EFp = (pFp**2 + mp**2) ** 0.5

    # ---- l p -> l' p a ----- #
    if process == "ep->upa":
        pF1 = pFu
        mu_1 = EFu
        mu_a = EFe
        pFa = EFe  # electron Fermi momentum
        pFb = pFp
        mu_b = EFp
        mu_2 = EFp
    if process == "up->epa":
        pFa = pFu
        mu_a = EFu
        mu_1 = EFe
        pF1 = pFe
        pFb = pFp
        mu_b = EFp
        mu_2 = EFp

    return pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2, pFe, pFu, pFp, EFe, EFu, EFp


# Processes
def get_masses(process):
    """
    Parameters
    ----------
    process : str
        must be one of the following:
        ep->upa
        up->epa
        ee->uea
        uu->eua
        eu->uua
        ue->eea
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


class Parameters:
    """Class that contains methods for getting .csv header, filename, directory.

    List of all relevant variables:
    - T
    - beta_F_mu
    - n
    - m3
    - neval
    - process

    The filename will be of the form
      "{process}/emissivity-vs-{dep}-var1={var1}-var2={var2}-var3={var3}-var4={var4}.csv"
    where var1, var2, var3, var4 are the variables in the list above which are
    not the dependent variable, dep.

    Floats will be formatted using f"{float_value:.3e}", and ints are formatted as f"{int_value}".


    Parameters
    ----------
    dep : str
        Name of the dependent parameter. Must be one of "T", "beta_F_mu", "n"
        or "m3".
    params : dict
        A dictionary containing the values of all the variables. params.keys()
        must contain "T", "beta_F_mu", "n", and "m3". The values should be
        either floats or ints.
    """

    def __init__(self, dep: str, params: dict):
        variables = ["T", "beta_F_mu", "n", "m3", "neval", "process"]
        self.Delta = None
        if "Delta" in params.keys():
            variables.append("Delta")
            self.Delta = params["Delta"]
        assert dep in variables
        for _ in variables:
            assert _ in params.keys()

        self.params = params
        self.dep = dep
        self.process = params["process"]
        self.T = params["T"]
        self.beta_F_mu = params["beta_F_mu"]
        self.neval = params["neval"]
        self.m3 = params["m3"]
        self.n = params["n"]

    @property
    def filename(self):
        fn = f"{'-to-'.join(self.process.split('->'))}/emissivity-vs-{self.dep}"

        for x in ["T", "beta_F_mu", "n", "m3", "neval"]:
            if x != self.dep:
                if isinstance(self.params[x], int):
                    fn += f"-{x}={self.params[x]}"
                else:
                    fn += f"-{x}={self.params[x]:.3e}"

        # suffix that helps describe different variations of the integrand. 
        fn += "-kTF_sq.csv" # denotes integral was run with a version of the code that accounts for plasma effects.

        return fn

    @property
    def header(self):
        res = self.params_str
        if self.dep != "T":
            res += f"\nColumns:{self.dep},emissivity (ergs/cm^3/s),error"
        else:
            res += f"\nColumns:T / 10^9 K,emissivity (ergs/cm^3/s),error"
        return res

    @property
    def params_str(self):
        (
            pFa,
            pFb,
            pF1,
            mu_a,
            mu_b,
            mu_1,
            mu_2,
            pFe,
            pFu,
            pFp,
            EFe,
            EFu,
            EFp,
        ) = get_fermi_params(self.process, self.beta_F_mu)

        if self.Delta is not None:
            (
                pFa,
                pFb,
                pF1,
                mu_a,
                mu_b,
                mu_1,
                mu_2,
                pFe,
                pFu,
                pFp,
                EFe,
                EFu,
                EFp,
            ) = get_fermi_params_delta(self.process, self.beta_F_mu, self.Delta, self.T)

        ma, mb, m1, m2 = get_masses(self.process)

        params_str = (
            f"ma = {ma}"
            f"\nmb = {mb}"
            f"\nm1 = {m1}"
            f"\nm2 = {m2}"
            f"\npFa = {pFa}"
            f"\npFb = {pFb}"
            f"\npF1 = {pF1}"
            f"\npF2 = {np.sqrt(mu_2**2 - m2**2)}"
            f"\nmu_a = {mu_a}"
            f"\nmu_b = {mu_b}"
            f"\nmu_1 = {mu_1}"
            f"\nmu_2 = {mu_2}"
        )

        for key in [
            "T",
            "beta_F_mu",
            "n",
            "m3",
            "neval",
        ]:
            if key != self.dep:
                params_str += f"\n{key} = {self.params[key]}"

        return params_str


# beta_F_mu vals used by Hong Yi.
HY_beta_F_mu_vals = np.array(
    [
        1 / 100,
        1 / 50,
        3 / 100,
        1 / 25,
        1 / 20,
        1 / 10,
        3 / 20,
        1 / 5,
        1 / 4,
        3 / 10,
        33 / 100,
        17 / 50,
        7 / 20,
        9 / 25,
        37 / 100,
        2 / 5,
        9 / 20,
        12 / 25,
        51 / 100,
        27 / 50,
        57 / 100,
        3 / 5,
        31 / 50,
        16 / 25,
        33 / 50,
        17 / 25,
        7 / 10,
        18 / 25,
        37 / 50,
        19 / 25,
        39 / 50,
        4 / 5,
        41 / 50,
        21 / 25,
        43 / 50,
        22 / 25,
        9 / 10,
    ],
    dtype=float,
)
