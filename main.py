"""MAIN ROUTINE
Uses vegas Monte Carlo integrator to estimate axion emissivity integral from
neutron star (NS).
"""

import vegas
import numpy as np
import logging, sys
import time
from pathlib import Path
import os


from integrand import Integrand
from constants import (
    get_fermi_params,
    get_masses,
    CONVERSION_FACTOR,
    DEFAULT_VALUES,
    Parameters,
)

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

T0 = DEFAULT_VALUES["T"]  # MeV


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


def calc_emissivity(dep, params, directory=None, save=False, **kwargs):
    parameters = Parameters(dep, params)
    process = params["process"]
    beta_F_mu = params["beta_F_mu"]
    m3 = params["m3"]
    T = params["T"]
    n = params["n"]
    neval = params["neval"]

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

    print(f"{dep} = {params[dep]}")
    print(f"Parameters\n----------\n{parameters.params_str}")

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
    integ(f, nitn=10, neval=neval, alpha=0.1)

    # step 2 -- integ has adapted to f; keep results
    result = integ(f, nitn=10, neval=neval, alpha=False)
    print(result.summary())
    print(f"result = {result}    Q = {result.Q:.2f}")

    t2 = time.perf_counter()
    print(f"Time elapsed: {t2 - t1:.2f}")
    # --------------------- end of main part --------------------- #

    # -------------------------- DATA IO -------------------------- #
    if save:
        filepath = Path(directory) / parameters.filename
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)

        HEADER = parameters.header
        if dep != "T":
            data = np.array([[params[dep], result.mean, result.sdev]])
        else:
            data = np.array([[T / T0, result.mean, result.sdev]])

        save_file(filepath, header=HEADER, data=data)
        print(f"file updated at {str(filepath)}")

    return result


def main():
    m3 = 0
    T = 0.1 * T0
    n = 10
    neval = 2*10**7
    dep = "beta_F_mu"

    for process in [
        "ep->upa",
        "up->epa",
        "ee->uea",
        "ue->eea",
        "eu->uua",
        "uu->eua",
    ]:
        if process.split("->")[0] in ["ep", "up", "ue", "ee"]:
            beta_F_mu_vals = np.linspace(0.05, 0.95, 5)
        else:
            beta_F_mu_vals = np.linspace(0.30, 0.95, 5)

        for beta_F_mu in beta_F_mu_vals:
            print(f"\n--------\nStarting {process}\n--------")

            params = {
                "process": process,
                "beta_F_mu": beta_F_mu,
                "T": T,
                "m3": m3,
                "n": n,
                "neval": neval,
            }

            if m3 > 0:
                RESULTS_DIRECTORY = f"./results/massive-axion/"
            else:
                RESULTS_DIRECTORY = f"./results/"

            res = calc_emissivity(
                dep,
                params,
                directory=RESULTS_DIRECTORY,
                save=True,
            )


if __name__ == "__main__":
    main()
