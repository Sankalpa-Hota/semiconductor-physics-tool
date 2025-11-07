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

if __name__ == "__main__":
    T = np.linspace(50, 600, 100)
    mu_acoustic = acoustic_phonon_mobility(T)
    mu_impurity = ionized_impurity_mobility(T, Nd=1e18)

    plt.plot(T, mu_acoustic, label='Acoustic Phonon Scattering')
    plt.plot(T, mu_impurity, label='Ionized Impurity Scattering')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Mobility (arbitrary units)')
    plt.title('Carrier Mobility vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.show()

