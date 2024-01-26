"""Methods for computing the emissivity integrand.

Throughout this file we use a (- + + +) metric signature. 


First, we describe the integrand.

1. OVERVIEW OF THE INTEGRAND

There are 6 processes that are relevant channels for LFV interactions.
These are listed below for convenience.
                                Processes
                                ---------
                                ep -> upa
                                up -> epa
                                ee -> uea
                                uu -> eua
                                eu -> uua
                                ue -> eea

For all of these processes, the axion emissivity from neutron stars is 
given by the following integral, for which this module provides methods
for computing the integrand. Note that all observables are measured in
the rest frame of the neutron star.

            Ɛ = (4 π) / S *
                ∫  fa * fb * (1 - f1) * (1 - f2)    // (1) Thermal factors
                    * E3                            // (2) Axion energy
                    * |M.E|^2                       // (3) Matrix element
                    * Sqrt(Ea^2 - ma^2)             // (4) Measure factors
                    * Sqrt(Eb^2 - mb^2)             // (4)
                    * Sqrt(E1^2 - m1^2)             // (4)
                    * Sqrt(E3^2 - m3^2) * (4 * pi) / E2  
                    
                    dEa dEb dE1 dcosa dcosb dcos1 dphia dphib dphi1

The integrand has several factors:
1. Thermal factors
2. Energy emitted by outgoing axion
3. Spin-summed matrix element squared
4. Measure factors arising from rewriting d^3 p in spherical coordinates.

The factor of (4 pi) comes from choosing a coordinate system in which
the axion's momentum p3vec points in the z-direction so that the integral
over dΩ₃ becomes trivial.


2. DESCRIPTION OF INTEGRAND MODULE

The main class is the Integrand class. Initialising an instance of the
Integrand class requires the user to input the process (e.g. "ep->upa"),
the Fermi momentum of the muon, the axion mass, and the temperature. 
An Integrand has a __call__ method which expects a 9-dimensional array
as input. E.g.

    process = "ep->upa"


    f = Integrand(process, beta_F_mu, m3, T). 
    x = [x0, x1, x2, x3, x4, x5, x6, x7, x8]
    # f(x) --> some number

When f is initialised, it uses the process, beta_F_mu, m3, and T to
uniquely determine all relevant parameters such as the masses 
(ma, mb, m1, m2) and chemical potentials (mu_a, mu_b, mu_1, mu_2).

While the emissivity integrand has the same form shown in eq. (1) for
each process, the matrix element differs. Moreover, if the axion is 
assumed to be massless, then many simplifications arise which motivate
definitions of independent methods for that case. 
"""

import numpy as np
from numpy import cos, sin
from numpy.linalg import norm as np_norm
from functools import partial
from numba import jit

from constants import get_fermi_params, get_fermi_params_delta, get_masses, alpha_EM


# ----- INTEGRAND CLASS ----- #
class Integrand:
    """Callable object which evaluates the integrand at a given value of the (9) integration variables."""

    def __init__(
        self, process, beta_F_mu, m3, T, conversion_factor, Delta=None, trivial=False
    ):
        ma, mb, m1, m2 = get_masses(process)

        if Delta is None:
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
            ) = get_fermi_params(process, beta_F_mu)

        if Delta is not None:
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
            ) = get_fermi_params_delta(process, beta_F_mu, Delta, T)
        # Thomas-Fermi wavenumber
        # kTF_sq = 4.0 * alpha_EM * (pFe * EFe + pFu * EFu + pFp * EFp) / np.pi
        # print(f"kTF_sq = {kTF_sq}")
        kTF_sq = 0.0

        if process.split("->")[0] in ["ep", "up"]:
            self.symmetry_factor = 1
        else:
            # symmetry factor for all other channels
            self.symmetry_factor = 2

        if trivial:
            self.integrand = partial(
                integrand_massless_axion,
                ma=ma,
                mb=mb,
                m1=m1,
                m2=m2,
                mu_a=mu_a,
                mu_b=mu_b,
                mu_1=mu_1,
                mu_2=mu_2,
                T=T,
                kTF_sq=kTF_sq,
                matrix_element_sq=trivial_matrix_element_sq,
            )
        elif m3 == 0:
            self.integrand = partial(
                integrand_massless_axion,
                ma=ma,
                mb=mb,
                m1=m1,
                m2=m2,
                mu_a=mu_a,
                mu_b=mu_b,
                mu_1=mu_1,
                mu_2=mu_2,
                T=T,
                kTF_sq=kTF_sq,
                matrix_element_sq=_fetch_matrix_element_sq(process, m3),
            )
        else:
            self.integrand = partial(
                integrand_massive_axion,
                ma=ma,
                mb=mb,
                m1=m1,
                m2=m2,
                m3=m3,
                mu_a=mu_a,
                mu_b=mu_b,
                mu_1=mu_1,
                mu_2=mu_2,
                T=T,
                kTF_sq=kTF_sq,
                matrix_element_sq=_fetch_matrix_element_sq(process, m3),
            )

        # A constant that converts from MeV^5 to the desired units,
        # e.g. ergs/cm^3/s
        self.conversion_factor = conversion_factor

    def __call__(self, x):
        return (
            self.conversion_factor
            * self.integrand(
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
            / self.symmetry_factor
        )


@jit(nopython=True, error_model="numpy")
def integrand_massive_axion(
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
    m3,
    mu_a,
    mu_b,
    mu_1,
    mu_2,
    T,
    matrix_element_sq,
):
    """Integrand, including measure factors. Works also when m3 > 0.

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
            Mass of ptle 'a' in MeV.
        mb : float
            Mass of ptle 'b' in MeV.
        m1 : float
            Mass of ptle '1' in MeV.
        m2 : float
            Mass of ptle '2' in MeV.
        m3 : float
            Mass of ptle '3' in MeV.
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

    # ---- Solve for p3mag using energy conservation ---- #
    C = Ea + Eb - E1
    guess = (C - Epsilon) / (1 - Pz / Epsilon)

    # continue if a positive value of p3mag exists which can
    # enforce energy conservation.
    if solution_exists(C, Pmag, Pz, m2, m3):
        p3mag = find_root(g, g_prime, guess, Ea + Eb - E1, Pmag, Pz, m2, m3)
        E3 = (p3mag**2 + m3**2) ** 0.5

        # Coord. system is chosen so that p3 points in z-direction.
        p3vec = p3mag * np.array([0, 0, 1])
        p2vec = pavec + pbvec - p1vec - p3vec
        E2 = (np_norm(p2vec) ** 2 + m2**2) ** 0.5

        # jac_factor arises from using energy Dirac delta to fix
        # p3mag to the value specified above.
        jac_factor = 1 / abs(g_prime(p3mag, C, Pmag, Pz, m2, m3))

        # ----- THE INTEGRAND ----- #
        _integrand = (
            norm
            * pamag
            * pbmag
            * p1mag
            * p3mag
            / E2
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
def integrand_massless_axion(
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
    kTF_sq,
    matrix_element_sq,
):
    """Integrand, including measure factors. Only valid for m3 = 0.

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
        m2 : float
            Mass of ptle '2'.
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

    guess = p3mag = ((Ea + Eb - E1) - Epsilon) / (1 - Pz / Epsilon)
    C = Ea + Eb - E1
    if solution_exists(C, Pmag, Pz, m2, 0):
        p3mag = find_root(g, g_prime, guess, Ea + Eb - E1, Pmag, Pz, m2, 0)
        E3 = (p3mag**2) ** 0.5

        # Coord. system is chosen so that p3 points in z-direction.
        p3vec = p3mag * np.array([0, 0, 1])
        p2vec = pavec + pbvec - p1vec - p3vec
        E2 = (np_norm(p2vec) ** 2 + m2**2) ** 0.5

        # jac_factor arises from using energy Dirac delta to fix
        # p3mag to the value specified above.
        jac_factor = 1 / abs(g_prime(p3mag, C, Pmag, Pz, m2, 0))

        # ----- THE INTEGRAND ----- #
        _integrand = (
            norm
            * pamag
            * pbmag
            * p1mag
            * E3
            * jac_factor
            * E3
            / E2
            * thermal_factors(Ea, Eb, E1, E2, mu_a, mu_b, mu_1, mu_2, T)
            * matrix_element_sq(
                Ea,
                Eb,
                E1,
                E2,
                E3,
                pavec,
                pbvec,
                p1vec,
                p2vec,
                p3vec,
                ma,
                mb,
                m1,
                m2,
                kTF_sq,
            )
        )
        return _integrand
    return np.float64(0.0)


@jit(nopython=True, error_model="numpy")
def integrand_massless_axion_approximate_p3mag(
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
    matrix_element_sq,
):
    """Integrand, including measure factors. Only valid for m3 = 0.

    This method differs from integrand_massless_axion by approximating
    p3mag by doing a series expansion of E2(p3mag) assuming p3mag / Epsilon
    is small where Epsilon = sqrt(| pavec + pbvec - p1vec |^2 + m2^2).

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
        m2 : float
            Mass of ptle '2'.
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
    p3mag = ((Ea + Eb - E1) - Epsilon) / (1 - Pz / Epsilon)
    E3 = p3mag

    if E3 > 0:
        # jac_factor arises from using energy Dirac delta to fix
        # E3 to the value specified above.
        jac_factor = 1 / abs(
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
            / E2
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


# -------- General methods & physics ------- #


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
    if x < 200:
        return 1 / (np.exp(x) + 1)
    return 0


@jit(nopython=True, error_model="numpy")
def lorentz_dot(E_i, p_ivec, E_j, p_jvec):
    """Lorentz dot product
    Using (-, +, +, +) convention to be consistent with HY Zhang.
    """
    return np.dot(p_ivec, p_jvec) - E_i * E_j


# --------------------------------------------------------------------- #
#                      DEFINITION OF MATRIX ELEMENTS                    #
# --------------------------------------------------------------------- #


def trivial_matrix_element_sq(
    Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2, kTF_sq
):
    return 1.0


#                             Massless axion                            #
#                             --------------                            #


def _fetch_matrix_element_sq(process, m3):
    in_state = process.split("->")[0]
    # if m3 == 0:
    # currently the matrix element for the massive and massless
    # axion cases are treated as the same, but this is just an
    # approximation. I will need to edit the code to incorporate
    # the full matrix element properly.
    if in_state in ["ep", "up"]:
        return lp_matrix_element_sq_massless
    if in_state in ["ee", "uu"]:
        return ll_matrix_element_sq_massless
    if in_state in ["eu", "ue"]:
        return llp_matrix_element_sq_massless


@jit(nopython=True, error_model="numpy")
def lp_matrix_element_sq_massless(
    Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2, kTF_sq
):
    """Calculate spin-summed matrix element squared for lp -> l'pa processes."""

    gaemu_sq = (10**-11) ** 2
    e = (4 * np.pi / 137) ** 0.5
    norm = 128 * gaemu_sq * e**4 / (ma**2 - m1**2) ** 2

    pa_dot_p1 = lorentz_dot(Ea, pavec, E1, p1vec)
    pb_dot_p3 = lorentz_dot(Eb, pbvec, E3, p3vec)
    p2_dot_p3 = lorentz_dot(E2, p2vec, E3, p3vec)

    Eb2 = Eb - E2
    pb2vec = pbvec - p2vec
    pb2_sq = lorentz_dot(Eb2, pb2vec, Eb2, pb2vec)

    q_sq = pb2_sq + kTF_sq

    result = -((pa_dot_p1 + ma * m1) * pb_dot_p3 * p2_dot_p3) / q_sq**2
    return norm * result


@jit(nopython=True, error_model="numpy")
def ll_matrix_element_sq_massless(
    Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2, kTF_sq
):
    """Calculate spin-summed matrix element squared for ll -> l'la processes."""
    # same as (lp)
    lp_me_sq = lp_matrix_element_sq_massless(
        Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2, kTF_sq
    )
    # (lp) but with a <-->  b
    permuted_me_sq = lp_matrix_element_sq_massless(
        Eb, Ea, E1, E2, E3, pbvec, pavec, p1vec, p2vec, p3vec, mb, ma, m1, m2, kTF_sq
    )
    # interaction term
    T_ll = ll_interaction_massless(
        Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2, kTF_sq
    )

    return lp_me_sq + permuted_me_sq + T_ll


@jit(nopython=True, error_model="numpy")
def llp_matrix_element_sq_massless(
    Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2, kTF_sq
):
    """Calculate spin-summed matrix element squared for ll -> l'la processes."""
    # same as (lp)
    lp_me_sq = lp_matrix_element_sq_massless(
        Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2, kTF_sq
    )
    # (lp) but with 1 <-->  2
    permuted_me_sq = lp_matrix_element_sq_massless(
        Ea, Eb, E2, E1, E3, pavec, pbvec, p2vec, p1vec, p3vec, ma, mb, m2, m1, kTF_sq
    )
    # interaction term
    T_llp = llp_interaction_massless(
        Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2, kTF_sq
    )

    return lp_me_sq + permuted_me_sq + T_llp


@jit(nopython=True, error_model="numpy")
def ll_interaction_massless(
    Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2, kTF_sq
):
    """T^(ll) interaction term for M.E. (See eq. 2.33 of HY notes.)"""

    gaemu_sq = (10**-11) ** 2
    e = (4 * np.pi / 137) ** 0.5
    norm = 64 * gaemu_sq * e**4 / (ma**2 - m1**2) ** 2

    pa_dot_pb = lorentz_dot(Ea, pavec, Eb, pbvec)
    pa_dot_p1 = lorentz_dot(Ea, pavec, E1, p1vec)
    pa_dot_p2 = lorentz_dot(Ea, pavec, E2, p2vec)
    pa_dot_p3 = lorentz_dot(Ea, pavec, E3, p3vec)
    pb_dot_p1 = lorentz_dot(Eb, pbvec, E1, p1vec)
    pb_dot_p2 = lorentz_dot(Eb, pbvec, E2, p2vec)
    pb_dot_p3 = lorentz_dot(Eb, pbvec, E3, p3vec)
    p1_dot_p3 = lorentz_dot(E1, p1vec, E3, p3vec)
    p2_dot_p3 = lorentz_dot(E2, p2vec, E3, p3vec)

    Eb2 = Eb - E2
    pb2vec = pbvec - p2vec
    pb2_sq = lorentz_dot(Eb2, pb2vec, Eb2, pb2vec)

    Ea2 = Ea - E2
    pa2vec = pavec - p2vec
    pa2_sq = lorentz_dot(Ea2, pa2vec, Ea2, pa2vec)

    qa_sq = pa2_sq + kTF_sq
    qb_sq = pb2_sq + kTF_sq

    return (
        norm
        * p2_dot_p3
        * (
            (pb_dot_p1 + mb * m1) * pa_dot_p3
            + (pa_dot_p1 + ma * m1) * pb_dot_p3
            - (pa_dot_pb + ma * mb) * p1_dot_p3
        )
        / qa_sq
        / qb_sq
    )


@jit(nopython=True, error_model="numpy")
def llp_interaction_massless(
    Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2, kTF_sq
):
    """T^(ll') interaction term for M.E.
    See eq. 2.36 of HY notes. and surrounding discussion
    """

    return ll_interaction_massless(
        E1, E2, Ea, Eb, E3, p1vec, p2vec, pavec, pbvec, p3vec, m1, m2, ma, mb, kTF_sq
    )


#                              Massive axion                            #
#                             --------------                            #


# -------------- Energy conservation methods -------------- #
"""These methods are only used when m3 > 0. Otherwise I use an
approximate analytic solution for p3mag.

The methods here are all written for the purpose of finding the value
of p3mag that enforces energy conservation by minimizing
|Ea + Eb - E1 - E2 - E3|. 
Notice that energy conservation can be written as 
                    (Ea + Eb - E1) - E2 - E3 == 0 .
Grouping Ea + Eb - E1 into one constant called C, and writing E2 and E3
as functions of p3mag yields
                    C - E2(p3mag) -  E3(p3mag) ,
which can in turn be written in the form 
        C - sqrt(x^2 - 2 b x + a^2 + c^2) -  sqrt(x^2 + d^2) ,
where 
                            x = p3mag
                            a = Pmag
                            b = Pz
                            c = m2
                            d = m3.

An energy-conserving value of p3mag exists when the minimum value of
    f(x) = sqrt(x^2 - 2 b x + a^2 + c^2) +  sqrt(x^2 + d^2)
subject to the constraint x > 0 is less than C. I found analytically
that the minimizing value of x > 0 (denoted x_*) is given by
    x_* = b d / [ sqrt(a^2 - b^2 + c^2) + d ]      for b > 0
    x_* = 0                                        otherwise .
"""


@jit(nopython=True)
def x_star(a, b, c, d):
    """Value of x that minimizes f on [0, infinity]."""
    if b > 0:
        return b * d / ((a**2 - b**2 + c**2) ** 0.5 + d)
    return 0.0


@jit(nopython=True)
def f(x, a, b, c, d):
    x1 = (x**2 - 2 * b * x + a**2 + c**2) ** 0.5
    x2 = (x**2 + d**2) ** 0.5
    return x1 + x2


@jit(nopython=True)
def f_prime(x, a, b, c, d):
    x1_prime = (x - b) / (x**2 - 2 * b * x + a**2 + c**2) ** 0.5
    x2_prime = x / (x**2 + d**2) ** 0.5
    return x1_prime + x2_prime


@jit(nopython=True)
def g(x, C, a, b, c, d):
    """Abstract form of energy conservation.

    For clarity:
        x = p3mag (the thing we're trying to solve for)
        C = Ea + Eb - E1
        a = Pmag
        b = Pz
        c = m2
        d = m3
    """
    return C - f(x, a, b, c, d)


@jit(nopython=True)
def g_prime(x, C, a, b, c, d):
    """Derivative of g"""
    return -f_prime(x, a, b, c, d)


@jit(nopython=True)
def find_root(g, g_prime, x0, C, a, b, c, d, tol=1e-13, maxiter=10):
    """Minimize g using Newton-Raphson iteration."""

    x_old = x0
    g_old = g(x_old, C, a, b, c, d)

    i = 0
    error = tol + 0.1
    while (error > tol) and (i < maxiter):
        x_new = x_old - g_old / g_prime(x_old, C, a, b, c, d)
        g_new = g(x_new, C, a, b, c, d)

        error = abs(g_new)
        i += 1

        x_old = x_new
        g_old = g_new

    return x_new


@jit(nopython=True)
def solution_exists(C, a, b, c, d):
    """Determine if a root of g(x) exists such that x > 0.
    Since g(x) = C - f(x), the root only exists if the global
    minimum of f is less than or equal to C. This works because
    as x --> +/- infinity f(x) -> +infinity.
    """
    return f(x_star(a, b, c, d), a, b, c, d) < C


# ----------- END OF ENERGY CONSERVATION METHOD SECTION ------------ #
