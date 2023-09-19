"""MAIN ROUTINE
Uses vegas Monte Carlo integrator to estimate axion emissivity integral from
neutron star (NS).
"""

import vegas
import numpy as np
import logging, sys


from integrand import Integrand
from constants import get_input_params

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)


def main():
    if len(sys.argv) > 1:
        beta_F_mu = float(sys.argv[1])
    else:
        beta_F_mu = 0.836788
    n = 14

    print(f"mu +/- {n} * T\n")

    T, ma, mb, m1, m2, pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2 = get_input_params(
        T=0.0861733,  # MeV
        ma=0.511,  # MeV (electron)
        mb=0.8 * 938.27208816,  # MeV (proton)
        m1=106.0,  # MeV (muon)
        beta_F_mu=beta_F_mu
        # pFa=100,  # MeV (electron)
        # pFb=226.0,  # MeV (proton)
        # pF1=162.0,  # MeV (muon)
    )

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
        f"beta_F_mu = {beta_F_mu}\n"
    )

    print(params_str)

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
    neval = 10**6

    f = Integrand(ma, mb, m1, m2, mu_a, mu_b, mu_1, mu_2, T)
    # step 1 -- adapt to f; discard results
    integ(f, nitn=10, neval=neval, alpha=0.1)

    # step 2 -- integ has adapted to f; keep results
    result = integ(f, nitn=20, neval=neval, alpha=False)
    print(result.summary())
    print("result = %s    Q = %.2f" % (result, result.Q))
    # print(f"result = {result.mean:.2e}")

    # Append result to file
    # with open("change-beta_F_mu.txt", "ba") as f:
    #     np.savetxt(f, np.array([[beta_F_mu, result.mean]]), delimiter=",")


if __name__ == "__main__":
    main()
