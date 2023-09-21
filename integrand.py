import numpy as np
from numpy import cos, sin
from numpy.linalg import norm as np_norm
from functools import partial
from numba import jit


import logging


# ------------------------ OVERVIEW OF THE INTEGRAND ------------------------ #

"""
emissivity_3 = Integrate[
       fa * fb * (1 - f1) * (1 - f2) * E3NS * |M.E|^2 *
       Sqrt(EaNS^2 - ma^2) * Sqrt(EbNS^2 - mb^2)  * Sqrt(E1NS^2 - m1^2) *
       E3NS * (4 * pi),
       {EaNS nearby fermi surface},
       {EbNS nearby fermi surface},
       {E1NS nearby fermi surface},
       {cosa, -1, 1}
       {cosb, -1, 1}
       {cos1, -1, 1}
       {phia, 0, 2 * pi}
       {phib, 0, 2 * pi}
       {phi1, 0, 2 * pi}
   ]

The integrand has several factors:
1. Thermal factors
2. Energy emitted by outgoing axion
3. Spin-summed matrix element squared
4. Measure factors arising from rewriting d^3 p in spherical coordinates.
"""


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


@jit(nopython=True, error_model="numpy")
def fFD(E, mu, T):
    """Fermi-Dirac Distribution

    Parameters
    ----------
        E : float
            Energy in arbitrary units (must match mu and T)
        mu : float
            Chemical potential
        T : float
            Temperature
    """
    x = (E - mu) / T
    if x < 100:
        return 1 / (np.exp(x) + 1)
    return 0


@jit(nopython=True, error_model="numpy")
def integrand(
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
):
    """Integrand, including measure factors.

    Below 'NS frame' means neutron star rest frame.

    Parameters
    ----------
        Ea : float
            Energy of ptle 'a' in NS frame.
        Eb : float
            Energy of ptle 'b' in NS frame.
        E1 : float
            Energy of ptle '1' in NS frame.
        cosa : float
            Polar angle of ptle 'a' measured relative to ptle '3' axis.
        cosb : float
            Polar angle of ptle 'b' measured relative to ptle '3' axis.
        cos1 : float
            Polar angle of ptle '1' measured relative to ptle '3' axis.
        phia : float
            Azimuthal angle of ptle 'a' measured relative to ptle '3' axis.
        phib : float
            Azimuthal angle of ptle 'b' measured relative to ptle '3' axis.
        phi1 : float
            Azimuthal angle of ptle '1' measured relative to ptle '3' axis.
        ma : float
            Mass of ptle 'a'.
        mb : float
            Mass of ptle 'b'.
        m1 : float
            Mass of ptle '1'.
        mu_a : float
            Chemical potential of ptle 'a'. Approximated by its Fermi Energy.
        mu_b : float
            Chemical potential of ptle 'b'. Approximated by its Fermi Energy.
        mu_1 : float
            Chemical potential of ptle '1'. Approximated by its Fermi Energy.
        mu_2 : float
            Chemical potential of ptle '2'. Approximated by its Fermi Energy.
        T : float
            Temperature assuming all speciies in thermal equilibrium.
    """
    norm = (4 * np.pi) / 2**5 / (2 * np.pi) ** 11

    # ------ MOMENTUM RECONSTRUCTION ------ #

    # First calculate E3.
    # If it's positive, then this point in phase space is
    # kinematically allowed, so continue.
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

    # E3 = p3mag since axion is massless
    E3 = ((Ea + Eb - E1) - Epsilon) / (1 - Pz / Epsilon)

    if E3 > 0:
        # jac_factor arises from using energy Dirac delta to fix
        # E3 to the value specified above.
        jac_factor = 1 / np.abs(
            1 + (E3 - Pz) / np.sqrt(Pmag**2 + m2**2 + E3**2 - 2 * E3 * Pz)
        )

        # Coord. system is chosen so that p3 points in z-direction.
        p3vec = E3 * np.array([0, 0, 1])
        p2vec = pavec + pbvec - p1vec - p3vec
        E2 = (np_norm(p2vec) ** 2 + m2**2) ** 0.5

        # ----- THE INTEGRAND ----- #
        _integrand = (
            norm
            * pamag
            * pbmag
            * p1mag
            * E3
            * jac_factor
            * E3
            * thermal_factors(Ea, Eb, E1, E2, mu_a, mu_b, mu_1, mu_2, T)
            * matrix_element_sq(
                Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2
            )
        )
        return _integrand
    return np.float64(0.0)


@jit(nopython=True, error_model="numpy")
def thermal_factors(Ea, Eb, E1, E2, mu_a, mu_b, mu_1, mu_2, T):
    """Thermal factors

    Parameters
    ----------
    Ea : float
        Energy of ptle 'a'
    Eb : float
        Energy of ptle 'b'
    E1 : float
        Energy of ptle '1'
    E2 : float
        Energy of ptle '2'
    mu_a : float
        Chemical potential of ptle 'a'. Approximated by its Fermi Energy.
    mu_b : float
        Chemical potential of ptle 'b'. Approximated by its Fermi Energy.
    mu_1 : float
        Chemical potential of ptle '1'. Approximated by its Fermi Energy.
    mu_2 : float
        Chemical potential of ptle '2'. Approximated by its Fermi Energy.
    T : float
        Temperature assuming all speciies in thermal equilibrium.
    """
    return (
        fFD(Ea, mu_a, T)
        * fFD(Eb, mu_b, T)
        * (1 - fFD(E1, mu_1, T))
        * (1 - fFD(E2, mu_2, T))
    )


@jit(nopython=True)
def lorentz_dot(E_i, p_ivec, E_j, p_jvec):
    """Lorentz dot product
    Using (-, +, +, +) convention to be consistent with HY Zhang.
    """
    return np.dot(p_ivec, p_jvec) - E_i * E_j


@jit(nopython=True)
def matrix_element_sq(
    Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2
):
    """Calculuated spin-summed matrix element squared.

    Parameters
    ----------

    """

    gaemu_sq = (10**-11) ** 2
    e = np.sqrt(4 * np.pi / 137)
    norm = 128 * gaemu_sq * e**4 / (ma**2 - m1**2) ** 2

    pa_dot_p1 = lorentz_dot(Ea, pavec, E1, p1vec)
    pb_dot_p3 = lorentz_dot(Eb, pbvec, E3, p3vec)
    p2_dot_p3 = lorentz_dot(E2, p2vec, E3, p3vec)

    Eb2 = Eb - E2
    pb2vec = pbvec - p2vec
    pb2_sq = lorentz_dot(Eb2, pb2vec, Eb2, pb2vec)

    result = ((-pa_dot_p1 - ma * m1) * pb_dot_p3 * p2_dot_p3) / ((pb2_sq) ** 2)
    return norm * result
