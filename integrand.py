import numpy as np
from numpy import cos, sin
from numpy.linalg import norm as np_norm
from functools import partial
from numba import njit


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


def integrand(
    Ea: float,
    Eb: float,
    E1: float,
    cosa: float,
    cosb: float,
    cos1: float,
    phia: float,
    phib: float,
    phi1: float,
    ma: float,
    mb: float,
    m1: float,
    m2: float,
    mu_a: float,
    mu_b: float,
    mu_1: float,
    mu_2: float,
    T: float,
) -> float:
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
    (E3_is_pos, reconstruction) = reconstruct_momenta(
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
    )

    if E3_is_pos:
        (
            pavec,
            pbvec,
            p1vec,
            p2vec,
            p3vec,
            E2,
            E3,
            pamag,
            pbmag,
            p1mag,
            jac_factor,
        ) = reconstruction

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
    return 0


def reconstruct_momenta(
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
):
    """Reconstruct momenta from integration variables."""
    sina = np.sqrt(1 - cosa**2)
    sinb = np.sqrt(1 - cosb**2)
    sin1 = np.sqrt(1 - cos1**2)

    pamag = np.sqrt(Ea**2 - ma**2)
    pbmag = np.sqrt(Eb**2 - mb**2)
    p1mag = np.sqrt(E1**2 - m1**2)

    pavec = pamag * np.array([sina * cos(phia), sina * sin(phia), cosa])
    pbvec = pbmag * np.array([sinb * cos(phib), sinb * sin(phib), cosb])
    p1vec = p1mag * np.array([sin1 * cos(phi1), sin1 * sin(phi1), cos1])

    Pvec = pavec + pbvec - p1vec
    Pmag = np_norm(Pvec)
    Epsilon = np.sqrt(Pmag**2 + m2**2)
    Pz = Pvec[2]

    # E3 = p3mag since axion is massless
    E3 = ((Ea + Eb - E1) - Epsilon) / (1 - Pz / Epsilon)
    logging.debug(f"Check that small p3mag approximation is valid: {E3 / Epsilon}")

    # Check if E3 > 0
    # if not, then invalid kinematic parameters have been chosen
    # and we should return 0.
    E3_is_pos = E3 > 0
    if E3_is_pos:
        # jac_factor arises from using energy Dirac delta to fix
        # E3 to the value specified above.
        jac_factor = 1 / np.abs(
            1 + (E3 - Pz) / np.sqrt(Pmag**2 + m2**2 + E3**2 - 2 * E3 * Pz)
        )

        # Coord. system is chosen so that p3 points in z-direction.
        p3vec = E3 * np.array([0, 0, 1])
        p2vec = pavec + pbvec - p1vec - p3vec
        E2 = np.sqrt(np_norm(p2vec) ** 2 + m2**2)

        logging.debug(f"E3 / sqrt() = {E3}")
        logging.debug(f"Check energy conservation: {Ea + Eb - E1 - E2 - E3}")

        logging.debug(
            f"Check momentum conservation: {pavec + pbvec - p1vec - p2vec - p3vec}"
        )
        logging.debug(f"Check 'a' on-shell: {(Ea**2 - np_norm(pavec)**2)}")
        logging.debug(f"Check 'b' on-shell: {(Eb**2 - np_norm(pbvec)**2) / mb**2}")
        logging.debug(f"Check '1' on-shell: {(E1**2 - np_norm(p1vec)**2) / m1**2}")
        logging.debug(f"Check '2' on-shell: {(E2**2 - np_norm(p2vec)**2) / m2**2}")
        logging.debug(f"Check '3' on-shell: {(E3**2 - np_norm(p3vec)**2)}")

        return (
            E3_is_pos,
            [
                pavec,
                pbvec,
                p1vec,
                p2vec,
                p3vec,
                E2,
                E3,
                pamag,
                pbmag,
                p1mag,
                jac_factor,
            ],
        )
    return E3_is_pos, None


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


def matrix_element_sq(
    Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2
):
    """Calculuated spin-summed matrix element squared.

    Parameters
    ----------

    """

    def lorentz_dot(E_i, p_ivec, E_j, p_jvec):
        """Lorentz dot product
        Using (-, +, +, +) convention to be consistent with HY Zhang.
        """
        return np.dot(p_ivec, p_jvec) - E_i * E_j

    gaemu_sq = (10**-11) ** 2
    e = np.sqrt(4 * np.pi / 137)
    norm = 128 * gaemu_sq * e**4 / (ma**2 - m1**2) ** 2

    pa_dot_p1 = lorentz_dot(Ea, pavec, E1, p1vec)
    pb_dot_p3 = lorentz_dot(Eb, pbvec, E3, p3vec)
    # pb_dot_p2 = lorentz_dot(Eb, pbvec, E2, p2vec)
    p2_dot_p3 = lorentz_dot(E2, p2vec, E3, p3vec)

    Eb2 = Eb - E2
    pb2vec = pbvec - p2vec
    pb2_sq = lorentz_dot(Eb2, pb2vec, Eb2, pb2vec)

    result = ((-pa_dot_p1 - ma * m1) * pb_dot_p3 * p2_dot_p3) / ((pb2_sq) ** 2)
    return norm * result
