import numpy as np


def acoustic_phonon_mobility(T, mu_300=1350.0):
    # Phonon-limited mobility calibrated to Si electrons at 300K.
    # mu_300 = 1350 cm^2/V.s for electrons, 450 for holes.
    return mu_300 * (300.0 / max(float(T), 1.0)) ** 2.3


def ionized_impurity_mobility(T, Nd):
    # Brooks-Herring approximation, returns cm^2/V.s.
    if Nd <= 0:
        return 1e8
    T   = max(float(T), 1.0)
    Nd  = max(float(Nd), 1.0)
    den = Nd * max(np.log(1.0 + 4.6e13 * T**2 / Nd), 1e-12)
    return 4e21 * T**1.5 / den


def mobility_phonon(T, m_eff=None, Nd=1e18):
    # Combined mobility via Matthiessen's rule, returns cm^2/V.s.
    # m_eff kept for API compatibility but empirical model is used.
    mu_L = acoustic_phonon_mobility(T)
    mu_I = ionized_impurity_mobility(T, Nd)
    return 1.0 / (1.0 / mu_L + 1.0 / mu_I)


if __name__ == "__main__":
    print(f"mu at 300K Nd=1e17: {mobility_phonon(300, Nd=1e17):.1f} cm^2/V.s")
    print(f"mu at 300K Nd=1e19: {mobility_phonon(300, Nd=1e19):.1f} cm^2/V.s")
    print(f"mu at 600K Nd=1e17: {mobility_phonon(600, Nd=1e17):.1f} cm^2/V.s")
