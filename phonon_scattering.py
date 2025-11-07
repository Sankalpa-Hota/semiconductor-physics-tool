# phonon_scattering.py
import numpy as np
import matplotlib.pyplot as plt

def acoustic_phonon_mobility(T, B=1e-3):
    """
    Acoustic phonon scattering: μ ∝ T^-3/2
    B is material constant
    """
    return B * T**(-3/2)

def ionized_impurity_mobility(T, Nd, A=1e21):
    """
    Ionized impurity scattering: μ ∝ T^3/2 / Nd
    """
    return A * T**(3/2) / Nd

def mobility_phonon(T, m_eff=1.0, Nd=1e18):
    """
    Combined or default phonon-limited mobility function for Flask app
    Returns mobility in cm^2/V.s
    """
    # Example: just using acoustic phonon mobility for now
    # You can later combine with impurity scattering if desired
    return acoustic_phonon_mobility(T) * 1e4  # scale to cm^2/V.s

if __name__ == "__main__":
    T = np.linspace(50, 600, 100)
    mu_acoustic = acoustic_phonon_mobility(T)
    mu_impurity = ionized_impurity_mobility(T, Nd=1e18)
    mu_combined = mobility_phonon(T)

    plt.plot(T, mu_acoustic, label='Acoustic Phonon Scattering')
    plt.plot(T, mu_impurity, label='Ionized Impurity Scattering')
    plt.plot(T, mu_combined, label='mobility_phonon (for Flask)')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Mobility (cm^2/V.s)')
    plt.title('Carrier Mobility vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()
