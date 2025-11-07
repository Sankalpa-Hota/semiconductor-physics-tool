# drude_model.py
import numpy as np

q = 1.602176634e-19  # C
m0 = 9.10938356e-31  # kg

def mobility_drude(tau, m_eff):
    """
    Mobility μ = q * τ / m*
    tau in seconds, m_eff in kg
    returns mobility in m^2/V.s
    """
    return q * tau / m_eff

def conductivity(n, mu):
    """
    Electrical conductivity σ = n * q * μ
    n in cm^-3, μ in cm^2/V.s
    returns σ in S/cm
    """
    # Convert units: cm^-3 * cm^2/V.s * C
    sigma = n * mu * q * 1e-2
    return sigma

def mean_free_path(v, tau):
    """
    Mean free path l = v * tau
    v in m/s, tau in s
    """
    return v * tau

# Default example
if __name__ == "__main__":
    n = 1e18  # cm^-3
    tau = 0.24e-15  # s
    vF = 1e6  # m/s
    m_eff = 0.26 * m0  # Si electrons

    mu = mobility_drude(tau, m_eff) * 1e4  # convert to cm^2/V.s
    sigma = conductivity(n, mu)
    l = mean_free_path(vF, tau)

    print(f"Mobility μ = {mu:.2f} cm^2/V.s")
    print(f"Conductivity σ = {sigma:.3e} S/cm")
    print(f"Mean free path l = {l:.2e} m")

