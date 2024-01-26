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
    get_fermi_params_delta,
    get_masses,
    CONVERSION_FACTOR,
    DEFAULT_VALUES,
    Parameters,
    HY_beta_F_mu_vals,
)

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

T0 = DEFAULT_VALUES["T"]  # 10^9 K MeV


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
    """
    Calculate the axion emissivity once for a given set of parameters.
    The integral is performed using the vegas Python package.

    Parameters
    ----------
    dep : str
        A string which specifies the dependent variable. Possible choices
        are "T", "beta_F_mu", "n", or "m3". See the `Parameters` class in `constants.py`.
    params : dict
        A dictionary which must must contain the following key : value pairs,
        - "process" : str
          This determines which LFV process to do the integral for. Allowed
          values are,
                "ep->upa"
                "up->epa"
                "ee->uea"
                "uu->eua"
                "eu->uua"
                "ue->eea"
          The process is used to fetch the masses of the particles to use in
          the integrand.
        - "beta_F_mu" : float
          This is the muon velocity and must be a float between 0 and 1.
        - "m3" : float
          The mass of the axion in MeV.
        - "T" : float
          The temperature in MeV.
        - "n" : int
          This is used to determine the domain for energy integral. It needs
          to be sufficiently large that the integrator is not missing important
          peaks in the integrand, but shouldn't be too large that it spends a
          lot of time sampling regions where the integrand is small. 
        - "neval" : int
          This is a hyperparameter for the vegas integrator which determines
          the maximum number of steps to take in each iteration of the vegas
          algorithm.
          
          Optionally, params may contain "Delta" : float, which determines the
          energy gap between the chemical potentials of the incident particles
          in MeV. The user will probably not need to use this because I only
          added this functionality later to check that the emissivity becomes 
          small when the chemical potentials are mismatched. 
    directory : str 
        This controls where the results of the emissivity calculation are saved
        to disk if the `save` flag is set to True.
    save : bool
        Whether or not to save the results of the calculation to disk. If True, 
        this function will attempt to save a .csv file with a filename of the 
        following signature
            emissivity-vs-dep-param=val.csv
        e.g. 
            dep = "beta_F_mu"
            params = {
                "process" : "ee->uea",
                "T" : 43.09,
                "n" : 10,
                "m3" : 0
                "neval" : 100_000
            }
        will save a file to
            f"{directory}/ee-to-uea/emissivity-vs-beta_F_mu-T=4.309e+01-n=10-m3=0-neval=1000000.csv"
    """
    if kwargs["Delta"] is not None:
        params["Delta"] = kwargs["Delta"]
        Delta = kwargs["Delta"]
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
    
    # comment the next few lines out unless you want to mismatch the chemical 
    # potentials using "Delta"
    # (
    #     pFa,
    #     pFb,
    #     pF1,
    #     mu_a,
    #     mu_b,
    #     mu_1,
    #     mu_2,
    #     pFe,
    #     pFu,
    #     pFp,
    #     EFe,
    #     EFu,
    #     EFp,
    # ) = get_fermi_params_delta(process, beta_F_mu, Delta, T)

    # initialise Monte Carlo integrator object
    integ = vegas.Integrator(
        [
            [max(ma, mu_a - n * T), mu_a + n * T], # integration limits
            [max(mb, mu_b - n * T), mu_b + n * T], # order is important
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
    # print(f"Parameters\n----------\n{parameters.params_str}")

    # ------------------- COMPUTE THE INTEGRAL ------------------- #
    t1 = time.perf_counter()

    if Delta is None:
        f = Integrand(
            process,
            beta_F_mu,
            m3,
            T,
            CONVERSION_FACTOR,
            trivial=True, # trivial denotes a trivial (constant) matrix element
        )
    if Delta is not None:
        f = Integrand(
            process,
            beta_F_mu,
            m3,
            T,
            CONVERSION_FACTOR,
            Delta=Delta,
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
        filepath = (
            Path(directory)
            / "-to-".join(process.split("->"))
            / "emissivity-vs-delta-1.csv"
        )
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
    """
    The main routine called by this file. 
    This is where all the parameters are initialised.
    Whenever I use this script I manually change the parameters.
    """
    beta_F_mu_vals = HY_beta_F_mu_vals
    beta_F_mu = DEFAULT_VALUES["beta_F_mu"]

    n = 50
    # neval = 1 * 10**7
    dep = "T"
    m3 = 0
    # T = T0 * 46.41588833612777165
    # T = np.logspace(0, 3, 10)[4] * T0  # about 4 MeV
    # T = T0
    # beta_F_mu = DEFAULT_VALUES["beta_F_mu"]

    n_vals = [x for x in range(30, 100, 20)]
    n_vals.append(1)
    n_vals = sorted(n_vals)

    m3_vals = list(np.linspace(0, 6, 7 + 6))
    m3_vals = sorted(m3_vals)

    # Array of temperatures to iterate over is "T" is the dependent variable
    T_vals = (
        np.array(
            [
                1e-2,
                1e-1,
                1.0,
                2.154434690031883814,
                4.641588833612778409,
                10.0,
                21.54434690031883193,
                46.41588833612777165,
                100.0,
                215.4434690031882269,
                464.1588833612777307,
                1000.0,
                2000.0,
                3000.0,
                4000.0,
                10000.0,
            ]
        )
        * T0 # a fiducial temperature value given by 10^4 K
    )

    # T = T0

    # A list of neval values to iterate over when dep = "neval" .
    # this is useful if you want to check for convergence by 
    # calculating the integral for increasing sizes of neval.
    neval_vals = [5 * 10**5]
    
    # A list of processes to calculate the integral for, this is
    # never a dependent variable. You will get a separate output
    # file for each process.
    processes = [
        "up->epa",
        "ue->eea",
        "uu->eua",
        "ep->upa",
        "ee->uea",
        "eu->uua",
    ]

    for process in processes:
        for T in T_vals:
            for neval in neval_vals:
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
                    save=False,
                    Delta=None,
                )


if __name__ == "__main__":
    main()
