import numpy as np
from numpy import cos, sin
from numpy.linalg import norm as np_norm
from integrand import find_root, solution_exists, g, g_prime
import logging, sys
from constants import DEFAULT_VALUES, CONVERSION_FACTOR, get_fermi_params, get_masses
from integrand import Integrand
import vegas

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)


def generate_random_vars(ma, mb, m1, mu_a, mu_b, mu_1, T, SEED=30087):
    """Debugging function that generates integration variables randomly."""
    n = 3
    rng = np.random.default_rng(SEED)
    Ea = rng.uniform(max(ma, mu_a - n * T), mu_a + n * T)
    Eb = rng.uniform(max(mb, mu_b - n * T), mu_b + n * T)
    E1 = rng.uniform(max(m1, mu_1 - n * T), mu_1 + n * T)
    cosa = rng.uniform(-1, 1)
    cosb = rng.uniform(-1, 1)
    cos1 = rng.uniform(-1, 1)
    phia = rng.uniform(0, 2 * np.pi)
    phib = rng.uniform(0, 2 * np.pi)
    phi1 = rng.uniform(0, 2 * np.pi)

    return Ea, Eb, E1, cosa, cosb, cos1, phia, phib, phi1

def init_params(process, beta_F_mu):
    ma, mb, m1, m2 = get_masses(process)
    pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2 = get_fermi_params(
        process,
        beta_F_mu,
    )
    return ma, mb, m1, m2, pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2

def init_integrator(ma, mb, m1, m2, m3, pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2, T, n):
    """Initialise Monte Carlo integrator object"""
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
        # f"T = {T}\n"
        f"ma = {ma}\n"
        f"mb = {mb}\n"
        f"m1 = {m1}\n"
        f"m2 = {m2}\n"
        f"m2 = {m3}\n"
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

    print(f"mu +/- {n} * T\n")
    print(f"T = {T}")
    print(f"Parameters\n----------\n{params_str}")

    return integ

   

def check_energy_conservation(process, T, beta_F_mu, m3, tol=1e-8):

    ma, mb, m1, m2, pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2 = init_params(process, beta_F_mu)

    # n = 10
    # integ = init_integrator(ma, mb, m1, m2, m3, pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2, T, n)
    # f = Integrand(
    #     ma,
    #     mb,
    #     m1,
    #     m2,
    #     m3,
    #     mu_a,
    #     mu_b,
    #     mu_1,
    #     mu_2,
    #     T,
    #     conversion_factor=CONVERSION_FACTOR,
    # )

    count = 0
    count_2 = 0
    while count < 10_000:
        Ea, Eb, E1, cosa, cosb, cos1, phia, phib, phi1 = generate_random_vars(ma, mb, m1, mu_a, mu_b, mu_1, T, SEED=np.random.randint(low=0, high=10**4))

        sina = (1 - cosa**2) ** 0.5
        sinb = (1 - cosb**2) ** 0.5
        sin1 = (1 - cos1**2) ** 0.5

        pamag = (Ea**2 - ma**2) ** 0.5
        pbmag = (Eb**2 - mb**2) ** 0.5
        p1mag = (E1**2 - m1**2) ** 0.5

        pavec = pamag * np.array([sina * cos(phia), sina * sin(phia), cosa])
        pbvec = pbmag * np.array([sinb * cos(phib), sinb * sin(phib), cosb])
        p1vec = p1mag * np.array([sin1 * cos(phi1), sin1 * sin(phi1), cos1])

        Pvec = pavec + pbvec - p1vec
        Pmag = np_norm(Pvec)
        Epsilon = (Pmag**2 + m2**2) ** 0.5
        Pz = Pvec[2]

        # ---- Solve for p3mag using energy conservation ---- #
        C = Ea + Eb - E1
        guess = (C - Epsilon) / (1 - Pz / Epsilon)
        
        # continue if a positive value of p3mag exists which can
        # enforce energy conservation.
        if solution_exists(C, Pmag, Pz, m2, m3):
            p3mag = find_root(g, g_prime, guess, Ea + Eb - E1, Pmag, Pz, m2, m3, tol=tol)
            E3 = (p3mag**2 + m3**2) ** 0.5

            # Coord. system is chosen so that p3 points in z-direction.
            p3vec = p3mag * np.array([0, 0, 1])
            p2vec = pavec + pbvec - p1vec - p3vec
            E2 = (np_norm(p2vec) ** 2 + m2**2) ** 0.5

            if (abs(Ea + Eb - E1 - E2 - E3) == 0.0) and count_2 < 10:
                if count_2 == 0:
                    print("Exact energy conservation detected. This is suspicious...")
                print(f"p3mag = {p3mag}")
                print(f"C, a, b, c, d = {C}, {Pmag}, {Pz}, {m2}, {m3}")
                count_2 += 1

            assert abs(Ea + Eb - E1 - E2 - E3) < tol
            if count < 10:
                print(f"|Ea + Eb - E1 - E2 - E3| = {abs(Ea + Eb - E1 - E2 - E3)}")

            count += 1
    return True

def main():
    check_energy_conservation("ep->upa", DEFAULT_VALUES["T"], DEFAULT_VALUES["beta_F_mu"], 0.3, tol=1e-12)
    pass


from functools import partial

if __name__ == "__main__":
    main()
