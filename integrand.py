"""Exact integrand
The strategy employed here is to:
Sample EaNS, EbNS, cosbNS, phibNS in the NS rest frame. 
Sample E1CM, E2CM, cos1, phi1, phi12 in the CM frame.
Reconstruct momenta in the corresponding frame, lorentz transform
CM momenta back to the NS rest frame.


NIntegrate[
    
]
"""

import numpy as np
from numpy import cos, sin
from numpy.linalg import norm as np_norm
from functools import partial
from numba import jit


import logging


class Integrand:
    def __init__(self, ma, mb, m1, m2, mu_a, mu_b, mu_1, mu_2, T, conversion_factor):
        self.integrand = partial(
            integrand,
            ma=ma,
            mb=mb,
            m1=m1,
            m2=m2,
            mu_a=mu_a,
            mu_b=mu_b,
            mu_1=mu_1,
            mu_2=mu_2,
            T=T,
        )
        self.conversion_factor = conversion_factor

    def __call__(self, x):
        return self.conversion_factor * self.integrand(
            Ea=x[0],
            Eb=x[1],
            E1=x[2],
            cosa=x[3],
            cosb=x[4],
            cos1=x[5],
            phia=x[6],
            phib=x[7],
            phi1=x[8],
        )


def boost(four_vector, dir, gamma):
    """Boosts a four vector in a direction by a particular lorentz factor.

    Parameters
    ----------
    four_vector : np.ndarray(size=(4,), dtype=float)
        A four vector.
    dir : 2-tuple of floats
        A pair of angles (theta, phi) that determine the direction in which to
        boost. The angles are defined in the same coordinate system as the
        components of four_vector.
    gamma : float
        Lorentz factor that determines the amount of boost to apply.
    """
    # boost in z-direction
    boost_z = np.array(
        [
            [gamma, 0, 0, (gamma**2 - 1) ** 0.5],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [(gamma**2 - 1) ** 0.5, 0, 0, gamma],
        ]
    )

    return boost_z @ four_vector


@jit(nopython=True, error_mode="numpy")
def integrand(
    EaNS,
    EbNS,
    cosaNS,
    cosbNS,
    phiaNS,
    phibNS,
    E1CM,
    E2CM,
    cos1CM,
    phi1CM,
    phi12CM,
    ma,
    mb,
    m1,
    m2,
    mu_a,
    mu_b,
    mu_1,
    mu_2,
    T,
):
    # ----- Reconstruct NS frame momenta ----- #
    sinaNS = (1 - cosaNS**2) ** 0.5
    sinbNS = (1 - cosbNS**2) ** 0.5

    paNS = (EaNS**2 - ma**2) ** 0.5 * np.array(
        [sinaNS * cos(phiaNS), sinaNS * sin(phiaNS), cosaNS]
    )
    pbNS = (EbNS**2 - mb**2) ** 0.5 * np.array(
        [sinbNS * cos(phibNS), sinbNS * sin(phibNS), cosbNS]
    )

    pabNS = paNS + pbNS

    # Define a boost that takes you from NS frame to CM frame.
    pass


@jit(nopython=True, error_mode="numpy")
def fFD(E, mu, T):
    """Fermi-dirac distribution"""
    x = (E - mu) / T
    return 1 / (np.exp(x) + 1)


@jit(nopython=True, error_mode="numpy")
def thermal_factors(EaNS, EbNS, E1NS, E2NS, mu_a, mu_b, mu_1, mu_2, T):
    """Calculate fa * fb * (1 - f1) * (1 - f2) in NS frame.

    Parameters
    ----------
    EaNS : float
        Energy of ptle 'a' in NS frame.
    EbNS : float
        Energy of ptle 'b' in NS frame.
    E1NS : float
        Energy of ptle '1' in NS frame.
    E2NS : float
        Energy of ptle '2' in NS frame.
    mu_a : float
        Chemical potential of ptle 'a'.
    mu_b : float
        Chemical potential of ptle 'b'.
    mu_1 : float
        Chemical potential of ptle '1'.
    mu_2 : float
        Chemical potential of ptle '2'.
    T : float
        Temperature.
    """
    return (
        fFD(EaNS, mu_a, T)
        * fFD(EbNS, mu_b, T)
        * (1 - fFD(E1NS, mu_1, T))
        * (1 - fFD(E2NS, mu_2, T))
    )
