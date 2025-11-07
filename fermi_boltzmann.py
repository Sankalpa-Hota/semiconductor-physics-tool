# fermi_boltzmann.py
import numpy as np

# Constants
kB = 8.617333262145e-5  # eV/K
q = 1.602176634e-19     # C

def intrinsic_carrier_concentration(Nc, Nv, Eg, T=300):
    """
    Compute intrinsic carrier concentration ni
    ni = sqrt(Nc*Nv)*exp(-Eg/(2*kB*T))
    """
    ni = np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * kB * T))
    return ni

def fermi_level_n_type(Ec, Nd, ni, T=300):
    """
    Compute Fermi level for n-type semiconductor
    E_f = E_c + kT*ln(Nd/ni)
    """
    Ef = Ec + kB*T*np.log(Nd/ni)
    return Ef

def fermi_level_p_type(Ev, Na, ni, T=300):
    """
    Compute Fermi level for p-type semiconductor
    E_f = E_v - kT*ln(Na/ni)
    """
    Ef = Ev - kB*T*np.log(Na/ni)
    return Ef

def carrier_concentration(Ef, Ec, Ev, Nc, Nv, T=300):
    """
    Calculate electron and hole concentrations
    n = Nc * exp(-(Ec-Ef)/kT)
    p = Nv * exp(-(Ef-Ev)/kT)
    """
    n = Nc * np.exp(-(Ec - Ef)/(kB*T))
    p = Nv * np.exp(-(Ef - Ev)/(kB*T))
    return n, p

# Example default values
if __name__ == "__main__":
    Nc = 1e19  # cm^-3
    Nv = 6e18  # cm^-3
    Eg = 0.66  # eV for Ge
    Nd = 1e18  # n-type doping
    Ec = 0.0
    Ev = -Eg
    T = 300

    ni = intrinsic_carrier_concentration(Nc, Nv, Eg, T)
    Ef = fermi_level_n_type(Ec, Nd, ni, T)
    n, p = carrier_concentration(Ef, Ec, Ev, Nc, Nv, T)

    print(f"Intrinsic ni = {ni:.3e} cm^-3")
    print(f"Fermi level Ef = {Ef:.3f} eV")
    print(f"Electron concentration n = {n:.3e} cm^-3")
    print(f"Hole concentration p = {p:.3e} cm^-3")

