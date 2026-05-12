# fermi_boltzmann.py
"""
Semiconductor statistics in the non-degenerate (Boltzmann) limit.

Intrinsic concentration uses the standard mass-action form
  ni = sqrt(Nc * Nv) * exp(-Eg / (2 kB T))
with Eg and kB*T in eV (kB in eV/K). Defaults Nc = 2.8e19 cm^-3,
Nv = 1.04e19 cm^-3, Eg = 1.12 eV at 300 K match widely used Si
tables (e.g. Green, JAP 67, 2944 (1990); see also Wikipedia:
Intrinsic carrier concentration / Effective mass (semiconductor)).
"""
import numpy as np

kB = 8.617333262145e-5   # eV/K
q  = 1.602176634e-19     # C

# Literature defaults for Si at 300 K (order ni ~ 10^10 cm^-3 with the formula above)
NC_SI_CM3_300K = 2.8e19
NV_SI_CM3_300K = 1.04e19
EG_SI_EV_300K = 1.12


def intrinsic_carrier_concentration(Nc, Nv, Eg, T=300):
    T = max(float(T), 1.0)
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2.0 * kB * T))


def intrinsic_fermi_level(Ec, Ev, Nc, Nv, T=300):
    """
    Intrinsic Fermi level (non-degenerate, parabolic bands):
    Ei = (Ec + Ev)/2 + (kT/2) ln(Nv/Nc).
    """
    T = max(float(T), 1.0)
    return 0.5 * (Ec + Ev) + 0.5 * kB * T * np.log(max(Nv, 1e-30) / max(Nc, 1e-30))


def fermi_level_n_type(Ec, Ev, Nc, Nv, Nd, ni, T=300):
    """n-type: Ef = Ei + kT ln(Nd/ni) (Boltzmann, non-degenerate)."""
    T = max(float(T), 1.0)
    Ei = intrinsic_fermi_level(Ec, Ev, Nc, Nv, T)
    if ni <= 0:
        return Ei
    if Nd <= 0:
        return Ei
    return Ei + kB * T * np.log(max(Nd, 1e-30) / max(ni, 1e-30))


def fermi_level_p_type(Ec, Ev, Nc, Nv, Na, ni, T=300):
    """p-type: Ef = Ei − kT ln(Na/ni) (Boltzmann, non-degenerate)."""
    T = max(float(T), 1.0)
    Ei = intrinsic_fermi_level(Ec, Ev, Nc, Nv, T)
    if ni <= 0:
        return Ei
    if Na <= 0:
        return Ei
    return Ei - kB * T * np.log(max(Na, 1e-30) / max(ni, 1e-30))


def carrier_concentration(Ef, Ec, Ev, Nc, Nv, T=300):
    T = max(float(T), 1.0)
    n = Nc * np.exp(-(Ec - Ef) / (kB * T))
    p = Nv * np.exp(-(Ef - Ev) / (kB * T))
    return n, p


if __name__ == "__main__":
    Nc = 2.8e19
    Nv = 1.04e19
    Eg = 1.12
    Nd = 1e17
    Ec = 0.0
    Ev = -Eg
    T  = 300
    ni = intrinsic_carrier_concentration(Nc, Nv, Eg, T)
    Ef = fermi_level_n_type(Ec, Ev, Nc, Nv, Nd, ni, T)
    n, p = carrier_concentration(Ef, Ec, Ev, Nc, Nv, T)
    print(f"ni = {ni:.3e} cm^-3")
    print(f"Ef = {Ef:.4f} eV")
    print(f"n  = {n:.3e} cm^-3")
    print(f"p  = {p:.3e} cm^-3")

