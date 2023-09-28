import numpy as np
from numpy import cos, sin
from numpy.linalg import norm as np_norm
from functools import partial
from numba import jit


# ------------------------ OVERVIEW OF THE INTEGRAND ------------------------ #

"""
emissivity_3 = Integrate[

       fa * fb * (1 - f1) * (1 - f2) 
       * E3NS 
       * |M.E|^2 
       * Sqrt(EaNS^2 - ma^2) 
       * Sqrt(EbNS^2 - mb^2)  
       * Sqrt(E1NS^2 - m1^2) 
       * (E3NS^2 - m3^2) * (4 * pi) / E2,

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
    def __init__(
        self, ma, mb, m1, m2, m3, mu_a, mu_b, mu_1, mu_2, T, conversion_factor
    ):
        if m3 == 0:
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
that the minimizing value of x > 0 (denoted x*) is given by
    x* = b d / [ sqrt(a^2 - b^2 + c^2) + d ]      for b > 0
    x* = 0                                        otherwise .
"""

@jit(nopython=True)
def x_star(a, b, c, d):
    """Value of x that minimizes f on [0, infinity]."""
    if b > 0:        
        return b * d / ((a**2 - b**2 + c**2)**0.5 + d)
    return 0.

@jit(nopython=True)
def f(x, a, b, c, d):
    x1 = (x**2 - 2*b*x + a**2 + c**2)**0.5
    x2 = (x**2 + d**2)**0.5
    return x1 + x2

@jit(nopython=True)
def f_prime(x, a, b, c, d):
    x1_prime = (x - b) / (x**2 - 2*b*x + a**2 + c**2)**0.5
    x2_prime = x /  (x**2 + d**2)**0.5
    return  x1_prime + x2_prime

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
    return - f_prime(x, a, b, c, d)

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
            Mass of ptle 'a'.
        mb : float
            Mass of ptle 'b'.
        m1 : float
            Mass of ptle '1'.
        m2 : float
            Mass of ptle '2'.
        m3 : float
            Mass of ptle '3'.
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

        # # ----- FSA INTEGRAND ----- #
        # p3vec = E3 * np.array([0, 0, 1])
        # p2vec = pavec + pbvec - p1vec  # note: p3vec has been ignored here!
        # E2 = (np_norm(p2vec) ** 2 + m2**2) ** 0.5
        # # Fermi surface approximation fixes Ea -> mu_a, Eb -> mu_b, etc...
        # pFa = (mu_a**2 - ma**2) ** 0.5
        # pFb = (mu_b**2 - mb**2) ** 0.5
        # pF1 = (mu_1**2 - m1**2) ** 0.5
        # _integrand = (
        #     norm
        #     * pFa
        #     * pFb
        #     * pF1
        #     * E3
        #     / E2
        #     * 1  # jac_factor is trivial if p3vec is ignored in momentum conservation. This is because it makes E2 independent of E3.
        #     * E3
        #     * thermal_factors(Ea, Eb, E1, E2, mu_a, mu_b, mu_1, mu_2, T)
        #     * fsa_matrix_element_sq(
        #         mu_a, mu_b, mu_1, ma, mb, m1, pavec, pbvec, p1vec, p2vec, p3vec
        #     )
        # )
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


@jit(nopython=True, error_model="numpy")
def lorentz_dot(E_i, p_ivec, E_j, p_jvec):
    """Lorentz dot product
    Using (-, +, +, +) convention to be consistent with HY Zhang.
    """
    return np.dot(p_ivec, p_jvec) - E_i * E_j


@jit(nopython=True, error_model="numpy")
def matrix_element_sq(
    Ea, Eb, E1, E2, E3, pavec, pbvec, p1vec, p2vec, p3vec, ma, mb, m1, m2
):
    """Calculuated spin-summed matrix element squared.

    Parameters
    ----------

    """

    gaemu_sq = (10**-11) ** 2
    e = (4 * np.pi / 137) ** 0.5
    norm = 128 * gaemu_sq * e**4 / (ma**2 - m1**2) ** 2

    pa_dot_p1 = lorentz_dot(Ea, pavec, E1, p1vec)
    pb_dot_p3 = lorentz_dot(Eb, pbvec, E3, p3vec)
    p2_dot_p3 = lorentz_dot(E2, p2vec, E3, p3vec)

    Eb2 = Eb - E2
    pb2vec = pbvec - p2vec
    pb2_sq = lorentz_dot(Eb2, pb2vec, Eb2, pb2vec)

    result = ((-pa_dot_p1 - ma * m1) * pb_dot_p3 * p2_dot_p3) / ((pb2_sq) ** 2)
    return norm * result


# @jit(nopython=True, error_model="numpy")
# def fsa_matrix_element_sq(EFa, EFb, EF1, ma, mb, m1, pavec, pbvec, p1vec, p2vec, p3vec):
#     """Spin-summed matrix element squared using Fermi-surface approximation.

#     From eq. (2.29) of Hong Yi's notes.
#     """
#     e_em = (4 * np.pi / 137) ** 0.5
#     gaemu_sq = (10**-11) ** 2
#     norm = 32 * e_em**4 * gaemu_sq

#     E3 = np_norm(p3vec)
#     pmaga = np_norm(pavec)
#     pmagb = np_norm(pbvec)
#     pmag1 = np_norm(p1vec)
#     pmag2 = np_norm(p2vec)

#     beta_F_a = (1 - (ma / EFa) ** 2) ** 0.5
#     beta_F_b = (1 - (mb / EFb) ** 2) ** 0.5
#     beta_F_1 = (1 - (m1 / EF1) ** 2) ** 0.5

#     ca1 = np.dot(pavec, p1vec) / pmaga / pmag1
#     cb2 = np.dot(pbvec, p2vec) / pmagb / pmag2
#     cb3 = pbvec[-1] / pmagb
#     c23 = p2vec[-1] / pmag2

#     G = (
#         (1 - beta_F_b * cb3)
#         * (1 - beta_F_b * c23)
#         * (1 - beta_F_a * beta_F_1 * ca1)
#         / (1 - cb2) ** 2
#     )

#     _matrix_element_sq = (
#         norm
#         * E3**2
#         / EFa**2
#         / EFb**2
#         / beta_F_b**4
#         / (beta_F_a**2 - beta_F_1**2) ** 2
#         * G
#     )

#     return _matrix_element_sq
