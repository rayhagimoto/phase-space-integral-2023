import numpy as np
import logging, sys
from constants import get_input_params, DEFAULT_VALUES, CONVERSION_FACTOR
from integrand import integrand, reconstruct_momenta

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)

DEFAULT_VALUES["beta_F_mu"] = 0.836


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


def main():
    T, ma, mb, m1, m2, pFa, pFb, pF1, mu_a, mu_b, mu_1, mu_2 = get_input_params(
        **DEFAULT_VALUES
    )

    rng = np.random.default_rng()
    count = 0
    SEED = 0
    Ea, Eb, E1, cosa, cosb, cos1, phia, phib, phi1 = generate_random_vars(
        ma, mb, m1, mu_a, mu_b, mu_1, T, SEED
    )
    integ = integ = CONVERSION_FACTOR * integrand(
        Ea,
        Eb,
        E1,
        cosa,
        cosb,
        cos1,
        phia,
        phib,
        phi1,
        ma,
        mb,
        m1,
        m2,
        mu_a,
        mu_b,
        mu_1,
        mu_2,
        T,
    )
    while integ < 1e-13:
        SEED = rng.integers(100000)
        Ea, Eb, E1, cosa, cosb, cos1, phia, phib, phi1 = generate_random_vars(
            ma, mb, m1, mu_a, mu_b, mu_1, T, SEED
        )
        (E3_is_pos, reconstruction) = reconstruct_momenta(
            Ea, Eb, E1, cosa, cosb, cos1, phia, phib, phi1, ma, mb, m1, m2
        )
        if E3_is_pos:
            integ = CONVERSION_FACTOR * integrand(
                Ea,
                Eb,
                E1,
                cosa,
                cosb,
                cos1,
                phia,
                phib,
                phi1,
                ma,
                mb,
                m1,
                m2,
                mu_a,
                mu_b,
                mu_1,
                mu_2,
                T,
            )
        count += 1
    print(f"\nRANDOM_SEED = {SEED}")
    print(f"integ = {integ:.2e}")

    print(
        f"""Ea = {Ea}
Eb = {Eb}
E1 = {E1}
cosa = {cosa}
cosb = {cosb}
cos1 = {cos1}
phia = {phia}
phib = {phib}
"""
    )


from functools import partial

if __name__ == "__main__":
    main()
