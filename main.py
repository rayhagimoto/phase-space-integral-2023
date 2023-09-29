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
from constants import get_fermi_params, get_masses, CONVERSION_FACTOR, DEFAULT_VALUES

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


def calc_emissivity(
    process, beta_F_mu, T, m3, directory=None, filename=None, save=False
):
    n = 10

    ma, mb, m1, m2 = get_masses(process)
    pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2 = get_fermi_params(
        process,
        beta_F_mu,
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
        f"m3 = {m3}\n"
        f"pFa = {pFa}\n"
        f"pFb = {pFb}\n"
        f"pF1 = {pF1}\n"
        f"pF2 = {np.sqrt(mu_2**2 - m2**2)}\n"
        f"mu_a = {mu_a}\n"
        f"mu_b = {mu_b}\n"
        f"mu_1 = {mu_1}\n"
        f"mu_2 = {mu_2}\n"
        # f"beta_F_mu = {beta_F_mu}\n"
        f"neval = {neval}\n"
        f"n = {n}"
    )

    print(f"beta_F_mu = {beta_F_mu}")
    print(f"Parameters\n----------\n{params_str}")

    # ------------------- COMPUTE THE INTEGRAL ------------------- #
    t1 = time.perf_counter()

    # For constant matrix element, set conversion factor to 1 for convenience.
    # For full matrix element set conversion factor to  CONVERSION_FACTOR.
    f = Integrand(
        process,
        beta_F_mu,
        m3,
        T,
        CONVERSION_FACTOR,
    )

    # step 1 -- adapt to f; discard results
    integ(f, nitn=20, neval=neval, alpha=0.3)

    # step 2 -- integ has adapted to f; keep results
    result = integ(f, nitn=20, neval=neval, alpha=False)
    print(result.summary())
    print(f"result = {result}    Q = {result.Q:.2f}")

    t2 = time.perf_counter()
    print(f"Time elapsed: {t2 - t1:.2f}")
    # --------------------- end of main part --------------------- #

    # -------------------------- DATA IO -------------------------- #
    if save:
        from datetime import date

        today = date.today().strftime("%d-%m-%Y")
        filename = filename + f"-neval={neval}-1.csv"
        filepath = directory + "/" + filename
        HEADER = params_str + "\n" + f"Columns:beta_F_mu,emissivity (ergs/cm^3/s),error"
        data = np.array([[beta_F_mu, result.mean, result.sdev]])

        save_file(filepath, header=HEADER, data=data)
        print(f"file updated at {filepath}")

    return result


def main():
    m3 = 0
    process = DEFAULT_VALUES["process"]
    beta_F_mu = DEFAULT_VALUES["beta_F_mu"]
    T0 = DEFAULT_VALUES["T"]  # MeV
    process_lists = [
        ["ue->eea"],
        # ["ep->upa", "up->epa", "ee->uea", "ue->eea"],
        ["uu->eua", "eu->uua"],
    ]

    beta_F_mu_vals_list = [
        np.hstack(
            (
                np.linspace(0.0, 0.1, 5),
                np.linspace(0.1, 0.85, 6)[1:],
                np.linspace(0.85, 0.95, 6)[1:],
            )
        ),
        np.hstack((np.linspace(0.35, 0.4, 5), np.linspace(0.4, 0.95, 11)[1:])),
    ]

    for process_list, beta_F_mu_vals in zip(process_lists, beta_F_mu_vals_list):
        for process in process_list:
            print(f"\n--------\nStarting {process}\n--------")
            for beta_F_mu in beta_F_mu_vals:
                if m3 > 0:
                    FILE_DIRECTORY = (
                        f"./results/massive-axion/{'-to-'.join(process.split('->'))}"
                    )
                else:
                    FILE_DIRECTORY = f"./results/{'-to-'.join(process.split('->'))}"
                FILE_NAME = "beta_F_mu-vs-emissivity"

                Path(FILE_DIRECTORY).mkdir(exist_ok=True, parents=True)

                # for T in np.logspace(-3, 4, 50) * T0:
                res = calc_emissivity(
                    process, beta_F_mu, T0, m3, FILE_DIRECTORY, FILE_NAME, save=True
                )


if __name__ == "__main__":
    main()
