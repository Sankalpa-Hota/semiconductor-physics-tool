import numpy as np

def acoustic_phonon_mobility(T, B=1e-3):
    return B * T**(-3/2)

def ionized_impurity_mobility(T, Nd, A=1e21):
    return A * T**(3/2) / Nd

def mobility_phonon(T, m_eff=1.0, Nd=1e18):
    """
    Returns phonon-limited mobility for Flask app in cm^2/V.s
    """
    mu_ac = acoustic_phonon_mobility(T)
    mu_imp = ionized_impurity_mobility(T, Nd)
    # Combine via Matthiessen's rule
    mu_total = 1 / (1/mu_ac + 1/mu_imp)
    return mu_total * 1e4
