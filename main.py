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
from multiprocessing import cpu_count


from integrand import Integrand
from constants import (
    get_fermi_params,
    get_masses,
    CONVERSION_FACTOR,
    DEFAULT_VALUES,
    Parameters,
    HY_beta_F_mu_vals,
)

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

T0 = DEFAULT_VALUES["T"]  # MeV


def save_file(
    filepath,
    header,
    data,
):
    def get_header(fn):
        with open(fn, "r") as f:
            res = "\n".join(
                [x.split("# ")[1].strip() for x in f.readlines() if "#" in x]
            )
            print(res)
        return res

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

    if not WRITE_HEADER:
        HEADER = get_header(filepath)

        # sort results by first column
        tmp = np.loadtxt(filepath, delimiter=",")
        tmp = tmp[tmp[:, 0].argsort()]
        np.savetxt(filepath, tmp, delimiter=",", header=HEADER)


def calc_emissivity(dep, params, directory=None, save=False, **kwargs):
    parameters = Parameters(dep, params)
    process = params["process"]
    beta_F_mu = params["beta_F_mu"]
    m3 = params["m3"]
    T = params["T"]
    n = params["n"]
    neval = params["neval"]

    ma, mb, m1, m2 = get_masses(process)
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
    ) = get_fermi_params(
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
        nproc=cpu_count(),
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
    integ(f, nitn=10, neval=neval, alpha=0.3)

    # step 2 -- integ has adapted to f; keep results
    result = integ(f, nitn=10, neval=neval, alpha=False)
    print(result.summary())
    print(f"result = {result}    Q = {result.Q:.2f}")

    t2 = time.perf_counter()
    print(f"Time elapsed: {t2 - t1:.2f}")
    # --------------------- end of main part --------------------- #

    # -------------------------- DATA IO -------------------------- #
    if save:
        filepath = Path(directory) / (parameters.filename)
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
    beta_F_mu_vals = HY_beta_F_mu_vals

    n = 10
    # neval = 5 * 10**5
    dep = "T"
    m3 = 0
    # T = T0 * 46.41588833612777165
    # T = np.logspace(0, 3, 10)[4] * T0  # about 4 MeV
    # T = T0
    beta_F_mu = DEFAULT_VALUES["beta_F_mu"]

    n_vals = [x for x in range(5, 35, 5)]
    n_vals.append(1)
    n_vals = sorted(n_vals)

    m3_vals = list(np.linspace(0, 6, 7 + 6))
    m3_vals = sorted(m3_vals)

    T_vals = (
        np.array(
            [
                1e-2,
                # 1e-1,
                # 1.0,
                # 2.154434690031883814,
                # 4.641588833612778409,
                # 10.0,
                # 21.54434690031883193,
                # 46.41588833612777165,
                # 100.0,
                # 215.4434690031882269,
                # 464.1588833612777307,
                # 1000.0,
                10000.0,
            ]
        )
        * T0
    )

    # neval_vals = [10**8, 2*10**8, 5 * 10**8]
    for T in T_vals:
        for process in [
            "up->epa",
            "ue->eea",
            "uu->eua",
            "ep->upa",
            "ee->uea",
            "eu->uua",
        ]:
            for neval in [5 * 10**7]:
                print(f"\n--------\nStarting {process}\n--------")

                params = {
                    "process": process,
                    "beta_F_mu": beta_F_mu,
                    "T": T,
                    "m3": m3,
                    "n": n,
                    "neval": neval,
                }

                RESULTS_DIRECTORY = f"./results/"

                res = calc_emissivity(
                    dep,
                    params,
                    directory=RESULTS_DIRECTORY,
                    save=True,
                )


if __name__ == "__main__":
    main()
