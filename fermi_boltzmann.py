# fermi_boltzmann.py
import numpy as np

kB = 8.617333262145e-5   # eV/K
q  = 1.602176634e-19     # C


def intrinsic_carrier_concentration(Nc, Nv, Eg, T=300):
    T = max(float(T), 1.0)
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2.0 * kB * T))


def fermi_level_n_type(Ec, Nd, ni, T=300):
    T = max(float(T), 1.0)
    if ni <= 0 or Nd <= 0:
        return Ec
    return Ec + kB * T * np.log(max(Nd, 1e-30) / max(ni, 1e-30))


def fermi_level_p_type(Ev, Na, ni, T=300):
    T = max(float(T), 1.0)
    if ni <= 0 or Na <= 0:
        return Ev
    return Ev - kB * T * np.log(max(Na, 1e-30) / max(ni, 1e-30))


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
    Ef = fermi_level_n_type(Ec, Nd, ni, T)
    n, p = carrier_concentration(Ef, Ec, Ev, Nc, Nv, T)
    print(f"ni = {ni:.3e} cm^-3")
    print(f"Ef = {Ef:.4f} eV")
    print(f"n  = {n:.3e} cm^-3")
    print(f"p  = {p:.3e} cm^-3")

