"""MAIN ROUTINE
Uses vegas Monte Carlo integrator to estimate axion emissivity integral from
neutron star (NS).
"""

import vegas
import numpy as np
import logging, sys
import time
from pathlib import Path


from integrand import Integrand
from constants import get_input_params, CONVERSION_FACTOR

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)


def save_file(
    filepath,
    header,
    data,
):
    WRITE_HEADER = not Path(filepath).is_file()

    # Append result to file
    with open(filepath, "ba") as f:
        if WRITE_HEADER:
            np.savetxt(
                f,
                data,
                delimiter=",",
                header=header,
            )
        else:
            np.savetxt(
                f,
                data,
                delimiter=",",
            )


def main(beta_F_mu, directory, filename):
    n = 10
    print(f"mu +/- {n} * T\n")
    # beta_F_mu = 0.836788
    # T = 0.0861733 # MeV
    T, ma, mb, m1, m2, pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2 = get_input_params(
        T=0.0861733,  # MeV
        ma=0.511,  # MeV (electron)
        mb=0.8 * 938.27208816,  # MeV (proton)
        m1=106.0,  # MeV (muon)
        beta_F_mu=beta_F_mu,
        # pFa=100,  # MeV (electron)
        # pFb=226.0,  # MeV (proton)
        # pF1=162.0,  # MeV (muon)
    )

    # initialise Monte Carlo integrator object
    integ = vegas.Integrator(
        [
            [max(ma, mu_a - n * T), mu_a + n * T],
            [max(mb, mu_b - n * T), mu_b + n * T],
            [max(m1, mu_1 - n * T), mu_1 + n * T],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [0, 2 * np.pi],
            [0, 2 * np.pi],
            [0, 2 * np.pi],
        ],
        nproc=16,
    )

    neval = 10**7

    params_str = (
        f"T = {T}\n"
        f"ma = {ma}\n"
        f"mb = {mb}\n"
        f"m1 = {m1}\n"
        f"m2 = {m2}\n"
        f"pFa = {pFa}\n"
        f"pFb = {pFb}\n"
        f"pF1 = {pF1}\n"
        f"pF2 = {np.sqrt(mu_2**2 - m2**2)}\n"
        f"mu_a = {mu_a}\n"
        f"mu_b = {mu_b}\n"
        f"mu_1 = {mu_1}\n"
        f"mu_2 = {mu_2}\n"
        f"neval = {neval}"
    )

    print(f"beta_F_mu = {beta_F_mu}")
    print(f"Parameters\n----------\n{params_str}")

    # ------------------- COMPUTE THE INTEGRAL ------------------- #
    t1 = time.perf_counter()

    # For constant matrix element, set conversion factor to 1 for convenience.
    # For full matrix element set conversion factor to  CONVERSION_FACTOR.
    f = Integrand(
        ma, mb, m1, m2, mu_a, mu_b, mu_1, mu_2, T, conversion_factor=CONVERSION_FACTOR
    )
    # step 1 -- adapt to f; discard results
    integ(f, nitn=10, neval=neval, alpha=0.1)

    # step 2 -- integ has adapted to f; keep results
    result = integ(f, nitn=20, neval=neval, alpha=False)
    print(result.summary())
    print(f"result = {result}    Q = {result.Q:.2f}")

    t2 = time.perf_counter()
    print(f"Time elapsed: {t2 - t1:.2f}")
    # --------------------- end of main part --------------------- #

    # # -------------------------- DATA IO -------------------------- #
    # filename = filename + f"-neval={neval}-9-22-2023.csv"
    # filepath = directory + "/" + filename
    # HEADER = params_str + "\n" + f"Columns:n,emissivity (ergs/cm^3/s),error"
    # data = np.array([[beta_F_mu, result.mean, result.sdev]])

    # save_file(filepath, header=HEADER, data=data)


if __name__ == "__main__":
    T0 = 0.0861733  # MeV
    FILE_DIRECTORY = "./results/ep-to-upa-FSA"
    Path(FILE_DIRECTORY).mkdir(exist_ok=True)
    FILE_NAME = "beta_F_mu-vs-emissivity"
    beta_F_mu = 0.836788
    # for beta_F_mu in np.linspace(0.01, 0.05, 10):
    main(beta_F_mu, FILE_DIRECTORY, FILE_NAME)
