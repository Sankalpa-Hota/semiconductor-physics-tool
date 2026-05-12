# drude_model.py
import numpy as np

q  = 1.602176634e-19   # C
m0 = 9.10938356e-31    # kg


def mobility_drude(tau, m_eff):
    # returns m^2/V.s
    return q * tau / m_eff


def conductivity(n, mu):
    # n in cm^-3, mu in cm^2/V.s, returns S/m
    return n * 1e6 * mu * 1e-4 * q


def mean_free_path(v, tau):
    # v in m/s, tau in s, returns m
    return v * tau


if __name__ == "__main__":
    n     = 1e18
    tau   = 0.24e-15
    vF    = 1e6
    m_eff = 0.26 * m0
    mu    = mobility_drude(tau, m_eff) * 1e4
    sigma = conductivity(n, mu)
    l     = mean_free_path(vF, tau)
    print(f"mu    = {mu:.2f} cm^2/V.s")
    print(f"sigma = {sigma:.3e} S/m")
    print(f"l     = {l:.2e} m")
