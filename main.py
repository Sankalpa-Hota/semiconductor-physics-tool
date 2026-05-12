from flask import Flask, render_template, request, jsonify
from chatbot import chatbot_bp
from brillouin_zones import plot_brillouin_zones
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from scipy.interpolate import interp1d
import os

import fermi_boltzmann as fb
import drude_model as dm
import phonon_scattering as ps
import kronig_penney as kp
import reciprocal_lattice as rl

app = Flask(__name__)
app.register_blueprint(chatbot_bp)

# Physical constants
k_B  = 1.380649e-23
hbar = 1.0545718e-34
q    = 1.60218e-19
m0   = 9.10938e-31
eV   = 1.60218e-19

COLORS = [
    '#00f0ff', '#00ff9d', '#ffb800', '#ff3e8a',
    '#9b5de5', '#4cc9f0', '#7bed9f', '#ffd32a', '#f72585'
]

DARK_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,15,26,1)',
    font=dict(family='Space Mono, monospace', color='#6a8aaa', size=10),
    xaxis=dict(gridcolor='#1a2840', zerolinecolor='#1a2840',
               linecolor='#1a2840', tickfont=dict(size=9)),
    yaxis=dict(gridcolor='#1a2840', zerolinecolor='#1a2840',
               linecolor='#1a2840', tickfont=dict(size=9)),
    margin=dict(l=55, r=18, t=38, b=50),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1a2840',
                borderwidth=1, font=dict(size=9)),
)


def make_layout(title='', **kw):
    d = dict(**DARK_LAYOUT)
    d['title'] = dict(text=title, font=dict(size=10, color='#2e4460'), x=0.01)
    d.update(kw)
    return go.Layout(**d)


def gf(form, name, default):
    try:
        v = form.get(name, '').strip()
        return float(v) if v else default
    except Exception:
        return default


def pplot(fig):
    return plot(fig, output_type='div', include_plotlyjs=False)


# Physics helpers

def fermi_dirac(E, Ef, T):
    x = (E - Ef) / (k_B * max(T, 1) / eV)
    return 1.0 / (np.exp(np.clip(x, -500, 500)) + 1)


def ni_calc(Nc, Nv, Eg, T):
    return np.sqrt(Nc * Nv) * np.exp(-Eg * eV / (2 * k_B * max(T, 1)))


def ef_n(Ec, Nd, ni, T):
    if ni <= 0 or Nd <= 0:
        return Ec
    return Ec + (k_B * max(T, 1) / eV) * np.log(max(Nd, 1e-30) / max(ni, 1e-30))


def mu_calc(T, Nd):
    T    = max(float(T), 1.0)
    mu_L = 1350.0 * (T / 300.0) ** -2.3
    if Nd > 0:
        Nd    = max(float(Nd), 1.0)
        denom = Nd * max(np.log(1 + 4.6e13 * T**2 / Nd), 1e-12)
        mu_I  = 4e21 * T**1.5 / denom
    else:
        mu_I = 1e8
    return 1.0 / (1.0/mu_L + 1.0/mu_I)


def sigma_calc(n, mu):
    # n in cm^-3, mu in cm^2/V.s, returns S/m
    return n * 1e6 * mu * 1e-4 * q


def kp_dispersion(a, V0_J, b, N=600):
    E_arr  = np.linspace(1e-5 * eV, V0_J * 3.5, N)
    k_list = []
    E_list = []
    for E in E_arr:
        try:
            alpha = np.sqrt(2 * m0 * E) / hbar
            if E < V0_J:
                beta = np.sqrt(2 * m0 * (V0_J - E)) / hbar
                term = (np.cos(alpha*(a-b)) * np.cosh(np.clip(beta*b, -300, 300))
                        + (beta**2 - alpha**2) / (2*alpha*beta)
                        * np.sin(alpha*(a-b)) * np.sinh(np.clip(beta*b, -300, 300)))
            else:
                gamma = np.sqrt(2 * m0 * (E - V0_J)) / hbar
                term  = (np.cos(alpha*(a-b)) * np.cos(gamma*b)
                         - (gamma**2 + alpha**2) / (2*alpha*gamma)
                         * np.sin(alpha*(a-b)) * np.sin(gamma*b))
            if -1.0 <= term <= 1.0:
                k_list.append(np.arccos(np.clip(term, -1, 1)) / a)
                E_list.append(E / eV)
        except Exception:
            pass
    return np.array(k_list), np.array(E_list)


# Plot functions

def plt_fermi_dirac(Ef, T):
    E   = np.linspace(Ef - 0.6, Ef + 0.6, 600)
    fig = go.Figure(layout=make_layout('Fermi-Dirac Distribution'))
    for i, Ti in enumerate([100, 200, T, 500, 800]):
        fig.add_trace(go.Scatter(
            x=fermi_dirac(E, Ef, Ti), y=E,
            mode='lines', name=f'{int(Ti)} K',
            line=dict(color=COLORS[i % len(COLORS)], width=2)))
    fig.add_hline(y=Ef,
                  line=dict(color='rgba(255,184,0,0.4)', dash='dot', width=1))
    fig.update_layout(xaxis_title='f(E)', yaxis_title='Energy (eV)')
    return pplot(fig)


def plt_dos(m_eff_ratio, Ec, Ev, T, Ef):
    m_eff = m_eff_ratio * m0
    E     = np.linspace(Ev - 0.1, Ec + 1.0, 800)
    dos_c = np.zeros_like(E)
    cb    = E > Ec
    if cb.any():
        dos_c[cb] = ((1/(2*np.pi**2)) * (2*m_eff/hbar**2)**1.5
                     * np.sqrt((E[cb]-Ec)*eV) * eV / 1e6)
    dos_v = np.zeros_like(E)
    vb    = E < Ev
    if vb.any():
        dos_v[vb] = ((1/(2*np.pi**2)) * (2*m_eff/hbar**2)**1.5
                     * np.sqrt((Ev-E[vb])*eV) * eV / 1e6)
    norm = max(dos_c.max(), dos_v.max(), 1e-10)
    f    = fermi_dirac(E, Ef, T)
    fig  = go.Figure(layout=make_layout('Density of States & Fermi Occupation'))
    fig.add_trace(go.Scatter(x=E, y=dos_c/norm, mode='lines', name='DoS CB',
        line=dict(color=COLORS[0], width=2),
        fill='tozeroy', fillcolor='rgba(0,240,255,0.05)'))
    fig.add_trace(go.Scatter(x=E, y=dos_v/norm, mode='lines', name='DoS VB',
        line=dict(color=COLORS[3], width=2),
        fill='tozeroy', fillcolor='rgba(255,62,138,0.05)'))
    fig.add_trace(go.Scatter(x=E, y=f, mode='lines', name='f(E)',
        line=dict(color=COLORS[2], width=1.8, dash='dash')))
    fig.add_trace(go.Scatter(x=E, y=(dos_c/norm)*f, mode='lines',
        name='Occupied CB',
        line=dict(color=COLORS[1], width=2),
        fill='tozeroy', fillcolor='rgba(0,255,157,0.05)'))
    fig.update_layout(xaxis_title='Energy (eV)', yaxis_title='Normalized')
    return pplot(fig)


def plt_ni_vs_T(Nc, Nv, Eg):
    T  = np.linspace(150, 900, 400)
    ni = np.array([ni_calc(Nc, Nv, Eg, Ti) for Ti in T])
    fig = go.Figure(layout=make_layout('ni vs Temperature'))
    fig.add_trace(go.Scatter(x=T, y=ni, mode='lines',
        line=dict(color=COLORS[0], width=2.5),
        fill='tozeroy', fillcolor='rgba(0,240,255,0.04)'))
    fig.update_layout(xaxis_title='T (K)',
                      yaxis_title='ni (cm^-3)', yaxis_type='log')
    return pplot(fig)


def plt_mobility_vs_T(Nd):
    T    = np.linspace(80, 700, 400)
    mu   = np.array([mu_calc(Ti, Nd) for Ti in T])
    mu_L = 1350.0 * (T / 300.0) ** -2.3
    fig  = go.Figure(layout=make_layout('Mobility vs Temperature'))
    fig.add_trace(go.Scatter(x=T, y=mu, mode='lines', name='Total mu',
        line=dict(color=COLORS[1], width=2.5)))
    fig.add_trace(go.Scatter(x=T, y=mu_L, mode='lines', name='Lattice mu',
        line=dict(color=COLORS[0], width=1.5, dash='dash')))
    fig.update_layout(xaxis_title='T (K)',
                      yaxis_title='mu (cm^2/V.s)', yaxis_type='log')
    return pplot(fig)


def plt_carrier_vs_T(Nc, Nv, Eg, Nd):
    T    = np.linspace(150, 700, 400)
    Ec   = 0.0
    Ev   = -Eg
    n_a, p_a, ni_a = [], [], []
    for Ti in T:
        ni_v = ni_calc(Nc, Nv, Eg, Ti)
        Ef_v = ef_n(Ec, Nd, ni_v, Ti)
        kT   = k_B * max(Ti, 1) / eV
        n_v  = Nc * np.exp(-(Ec - Ef_v) / kT)
        p_v  = ni_v**2 / max(n_v, 1.0)
        n_a.append(n_v)
        p_a.append(p_v)
        ni_a.append(ni_v)
    fig = go.Figure(layout=make_layout('Carrier Concentrations vs T'))
    fig.add_trace(go.Scatter(x=T, y=n_a, mode='lines', name='n (electrons)',
        line=dict(color=COLORS[0], width=2)))
    fig.add_trace(go.Scatter(x=T, y=p_a, mode='lines', name='p (holes)',
        line=dict(color=COLORS[3], width=2)))
    fig.add_trace(go.Scatter(x=T, y=ni_a, mode='lines', name='ni',
        line=dict(color=COLORS[1], width=1.5, dash='dash')))
    fig.update_layout(xaxis_title='T (K)',
                      yaxis_title='(cm^-3)', yaxis_type='log')
    return pplot(fig)


def plt_conductivity_vs_T(Nc, Nv, Eg, Nd):
    T    = np.linspace(150, 700, 400)
    vals = []
    for Ti in T:
        ni_v = ni_calc(Nc, Nv, Eg, Ti)
        n_v  = max(Nd, ni_v)
        mu_v = mu_calc(Ti, Nd)
        vals.append(sigma_calc(n_v, mu_v))
    fig = go.Figure(layout=make_layout('Conductivity vs Temperature'))
    fig.add_trace(go.Scatter(x=T, y=vals, mode='lines',
        line=dict(color=COLORS[3], width=2.5),
        fill='tozeroy', fillcolor='rgba(255,62,138,0.04)'))
    fig.update_layout(xaxis_title='T (K)',
                      yaxis_title='sigma (S/m)', yaxis_type='log')
    return pplot(fig)


def plt_ef_vs_doping(Nc, Nv, Eg, T):
    Nd_arr = np.logspace(13, 20, 300)
    Ec     = 0.0
    Ev     = -Eg
    Ef_arr = [ef_n(Ec, Nd, ni_calc(Nc, Nv, Eg, T), T) for Nd in Nd_arr]
    fig    = go.Figure(layout=make_layout('Fermi Level vs Doping (n-type)'))
    fig.add_trace(go.Scatter(x=Nd_arr, y=Ef_arr, mode='lines',
        line=dict(color=COLORS[4], width=2.5)))
    fig.add_hline(y=Ec,
                  line=dict(color='rgba(0,240,255,0.3)', dash='dot'),
                  annotation_text='Ec',
                  annotation_font=dict(color='#00f0ff', size=9))
    fig.add_hline(y=Ev,
                  line=dict(color='rgba(255,62,138,0.3)', dash='dot'),
                  annotation_text='Ev',
                  annotation_font=dict(color='#ff3e8a', size=9))
    fig.update_layout(xaxis_title='Nd (cm^-3)', xaxis_type='log',
                      yaxis_title='Ef (eV)')
    return pplot(fig)


def plt_band_diagram(Ef, Ec, Ev, Eg):
    x   = [0, 2]
    fig = go.Figure(layout=make_layout('Energy Band Diagram'))
    fig.add_trace(go.Scatter(x=x, y=[Ec, Ec], mode='lines', name='Ec (CB)',
        line=dict(color=COLORS[0], width=3)))
    fig.add_trace(go.Scatter(x=x, y=[Ev, Ev], mode='lines', name='Ev (VB)',
        line=dict(color=COLORS[3], width=3)))
    fig.add_trace(go.Scatter(x=x, y=[Ef, Ef], mode='lines', name='Ef',
        line=dict(color=COLORS[2], width=2, dash='dash')))
    mid = (Ec + Ev) / 2.0
    fig.add_trace(go.Scatter(x=x, y=[mid, mid], mode='lines', name='Ei',
        line=dict(color='rgba(255,255,255,0.13)', width=1, dash='dot')))
    fig.add_hrect(y0=Ev, y1=Ec,
                  fillcolor='rgba(0,240,255,0.03)', line_width=0)
    fig.update_layout(xaxis=dict(visible=False), yaxis_title='Energy (eV)')
    return pplot(fig)


def plt_resistivity_vs_doping():
    Nd_arr = np.logspace(13, 20, 300)
    rho_n, rho_p = [], []
    for Nd in Nd_arr:
        mu_n  = mu_calc(300, Nd)
        mu_p  = mu_n * 0.47
        sig_n = sigma_calc(Nd, mu_n)
        sig_p = sigma_calc(Nd, mu_p)
        rho_n.append(1.0/sig_n if sig_n > 0 else 1e10)
        rho_p.append(1.0/sig_p if sig_p > 0 else 1e10)
    fig = go.Figure(layout=make_layout('Resistivity vs Doping'))
    fig.add_trace(go.Scatter(x=Nd_arr, y=rho_n, mode='lines', name='n-type',
        line=dict(color=COLORS[0], width=2)))
    fig.add_trace(go.Scatter(x=Nd_arr, y=rho_p, mode='lines', name='p-type',
        line=dict(color=COLORS[3], width=2)))
    fig.update_layout(xaxis_title='Doping (cm^-3)', xaxis_type='log',
                      yaxis_title='rho (ohm.m)', yaxis_type='log')
    return pplot(fig)


def plt_bandgap_vs_T(Eg_300=1.12):
    T     = np.linspace(1, 600, 400)
    alpha = 4.73e-4
    beta  = 636.0
    ref   = Eg_300 + alpha * 300**2 / (300 + beta)
    Eg    = ref - alpha * T**2 / (T + beta)
    fig   = go.Figure(layout=make_layout('Bandgap vs Temperature (Varshni)'))
    fig.add_trace(go.Scatter(x=T, y=Eg, mode='lines',
        line=dict(color=COLORS[2], width=2.5),
        fill='tozeroy', fillcolor='rgba(255,184,0,0.04)'))
    fig.update_layout(xaxis_title='T (K)', yaxis_title='Eg (eV)')
    return pplot(fig)


def plt_kp_1d(a, V0_J, b):
    k_vals, E_vals = kp_dispersion(a, V0_J, b)
    if len(k_vals) == 0:
        return ("<p style='color:#6a8aaa;padding:1rem;"
                "font-family:monospace;font-size:0.75rem'>"
                "No allowed bands — adjust V0 or barrier width.</p>")
    fig = go.Figure(layout=make_layout('Kronig-Penney Band Structure E(k)'))
    fig.add_trace(go.Scatter(
        x=k_vals/1e10, y=E_vals, mode='markers',
        marker=dict(color=COLORS[0], size=2.5, opacity=0.8),
        name='Allowed E'))
    fig.update_layout(xaxis_title='k (1/A)', yaxis_title='E (eV)')
    return pplot(fig)


def plt_kp_2d(a, V0_J, b):
    V0_range      = np.linspace(0.5*eV, V0_J * 2.5, 30)
    all_k, all_E, all_V = [], [], []
    for V in V0_range:
        k, E = kp_dispersion(a, V, b, N=200)
        if len(k):
            all_k.extend(k)
            all_E.extend(E)
            all_V.extend([V/eV] * len(k))
    if not all_k:
        return ("<p style='color:#6a8aaa;padding:1rem;"
                "font-family:monospace;font-size:0.75rem'>"
                "Insufficient data for this parameter range.</p>")
    fig = go.Figure(layout=make_layout('KP Band Map vs Barrier Height'))
    fig.add_trace(go.Scatter(
        x=np.array(all_k)/1e10, y=all_E, mode='markers',
        marker=dict(
            color=all_V, colorscale='Plasma', size=2, opacity=0.65,
            colorbar=dict(title='V0 (eV)', tickfont=dict(size=8))),
        name='Allowed E'))
    fig.update_layout(xaxis_title='k (1/A)', yaxis_title='E (eV)')
    return pplot(fig)


def plt_kp_3d(a, V0_J, b):
    k_vals, E_vals = kp_dispersion(a, V0_J, b, N=200)
    if len(k_vals) < 4:
        return ("<p style='color:#6a8aaa;padding:1rem;"
                "font-family:monospace;font-size:0.75rem'>"
                "Insufficient KP data for 3D surface plot.</p>")
    k_all = np.concatenate([-k_vals[::-1], k_vals])
    E_all = np.concatenate([E_vals[::-1],  E_vals])
    kx, ky = np.meshgrid(k_all/1e10, k_all/1e10)
    try:
        fi = interp1d(k_all/1e10, E_all, kind='linear',
                      bounds_error=False,
                      fill_value=(E_all[0], E_all[-1]))
        Ez = np.sqrt(fi(np.abs(kx))**2 * 0.5 + fi(np.abs(ky))**2 * 0.5)
    except Exception:
        r  = np.sqrt(kx**2 + ky**2)
        fi = interp1d(k_all/1e10, E_all, kind='linear',
                      bounds_error=False,
                      fill_value=(E_all[0], E_all[-1]))
        Ez = fi(r)

    fig = go.Figure(layout=make_layout('3D Band Structure E(kx, ky)'))
    fig.add_trace(go.Surface(
        x=kx, y=ky, z=Ez,
        colorscale='Plasma', opacity=0.88,
        colorbar=dict(title='E (eV)',
                      tickfont=dict(size=8, color='#6a8aaa')),
        contours=dict(z=dict(show=True,
                             color='rgba(0,240,255,0.3)', width=1))
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='kx (1/A)', gridcolor='#1a2840',
                       backgroundcolor='rgba(10,15,26,1)'),
            yaxis=dict(title='ky (1/A)', gridcolor='#1a2840',
                       backgroundcolor='rgba(10,15,26,1)'),
            zaxis=dict(title='E (eV)',   gridcolor='#1a2840',
                       backgroundcolor='rgba(10,15,26,1)'),
            bgcolor='rgba(10,15,26,1)',
        ),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
    )
    return pplot(fig)


def plt_reciprocal_lattice():
    return rl.plot_reciprocal_lattice_3d()


def plt_hall(Nc, Nv, Eg, Nd):
    T   = 300
    ni  = ni_calc(Nc, Nv, Eg, T)
    n   = max(Nd, ni)
    B   = np.linspace(0, 3, 300)
    R_H = 1.0 / (n * q * 1e6)
    V_H = R_H * 1e3 * B
    fig = go.Figure(layout=make_layout('Hall Voltage vs Magnetic Field'))
    fig.add_trace(go.Scatter(x=B, y=V_H * 1e3, mode='lines',
        line=dict(color=COLORS[4], width=2.5),
        fill='tozeroy', fillcolor='rgba(155,93,229,0.04)'))
    fig.update_layout(xaxis_title='B (T)',
                      yaxis_title='V_H (mV)  [I=1mA, d=1mm]')
    return pplot(fig)


def plt_iv_diode(T, Eg):
    V   = np.linspace(-0.5, 0.8, 500)
    ni  = ni_calc(2e19, 1e19, Eg, T)
    I0  = q * ni * 1e-4
    I   = I0 * (np.exp(np.clip(q*V/(k_B*max(T,1)), -500, 500)) - 1)
    fig = go.Figure(layout=make_layout('p-n Junction I-V Characteristic'))
    fig.add_trace(go.Scatter(x=V, y=I*1e3, mode='lines',
        line=dict(color=COLORS[1], width=2.5)))
    fig.add_vline(x=0, line=dict(color='rgba(255,255,255,0.07)', width=1))
    fig.add_hline(y=0, line=dict(color='rgba(255,255,255,0.07)', width=1))
    fig.update_layout(xaxis_title='V (V)', yaxis_title='I (mA)')
    return pplot(fig)


def plt_schottky(Eg=1.12):
    phi_m = 4.5
    chi   = 4.05
    phi_B = phi_m - chi
    x     = np.linspace(0, 5, 400)
    Ec    = -phi_B * np.exp(-x / 1.5)
    Ev    = Ec - Eg
    fig   = go.Figure(layout=make_layout('Schottky Barrier Band Diagram'))
    fig.add_trace(go.Scatter(x=x, y=Ec, mode='lines', name='Ec',
        line=dict(color=COLORS[0], width=2)))
    fig.add_trace(go.Scatter(x=x, y=Ev, mode='lines', name='Ev',
        line=dict(color=COLORS[3], width=2)))
    fig.add_vline(x=0,
                  line=dict(color='rgba(255,255,255,0.13)',
                            dash='dash', width=1),
                  annotation_text='Metal|Semi',
                  annotation_font=dict(color='#6a8aaa', size=8))
    fig.update_layout(xaxis_title='x (nm)', yaxis_title='E (eV)')
    return pplot(fig)


def plt_phonon():
    q_arr   = np.linspace(-np.pi, np.pi, 400)
    m1, m2, C = 1.0, 2.0, 1.0
    sq_a    = (C*(m1+m2)/(m1*m2)
               - C/(m1*m2)*np.sqrt((m1+m2)**2
                 - 4*m1*m2*np.sin(q_arr/2)**2))
    sq_o    = (C*(m1+m2)/(m1*m2)
               + C/(m1*m2)*np.sqrt((m1+m2)**2
                 - 4*m1*m2*np.sin(q_arr/2)**2))
    sq_a    = np.clip(sq_a, 0, None)
    sq_o    = np.clip(sq_o, 0, None)
    fig     = go.Figure(layout=make_layout('Phonon Dispersion (1D Diatomic Chain)'))
    fig.add_trace(go.Scatter(x=q_arr, y=np.sqrt(sq_a), mode='lines',
        name='Acoustic', line=dict(color=COLORS[1], width=2)))
    fig.add_trace(go.Scatter(x=q_arr, y=np.sqrt(sq_o), mode='lines',
        name='Optical',  line=dict(color=COLORS[2], width=2)))
    fig.update_layout(xaxis_title='q (units of 1/a)', yaxis_title='omega (arb.)')
    return pplot(fig)


def plt_depletion(Nc, Nd, Eg):
    eps_r = 11.7
    eps   = eps_r * 8.854e-12
    Vbi   = np.linspace(0.1, 1.5, 300)
    Na_m  = max(Nc * 1e3, 1e16) * 1e6
    Nd_m  = max(Nd,        1e13) * 1e6
    W     = np.sqrt(2 * eps * Vbi * (Na_m + Nd_m) / (q * Na_m * Nd_m))
    fig   = go.Figure(layout=make_layout('Depletion Width vs Built-in Voltage'))
    fig.add_trace(go.Scatter(x=Vbi, y=W*1e9, mode='lines',
        line=dict(color=COLORS[5], width=2.5),
        fill='tozeroy', fillcolor='rgba(76,201,240,0.04)'))
    fig.update_layout(xaxis_title='Vbi (V)', yaxis_title='W (nm)')
    return pplot(fig)


# Routes

@app.route('/', methods=['GET', 'POST'])
def home():
    # Defaults — Silicon at 300K
    Nc          = 2.8e19
    Nv          = 1.04e19
    Eg          = 1.12
    Nd          = 1e17
    T           = 300.0
    tau         = 0.24e-15
    vF          = 1e6
    m_eff_ratio = 0.26
    a           = 5e-10
    V0          = 10.0
    b           = 2e-10
    bz_lattice  = 'square'
    bz_a        = 1.0
    bz_b        = 1.5
    bz_angle    = 120.0
    bz_zones    = 4

    if request.method == 'POST':
        Nc          = gf(request.form, 'Nc',    Nc)
        Nv          = gf(request.form, 'Nv',    Nv)
        Eg          = gf(request.form, 'Eg',    Eg)
        Nd          = gf(request.form, 'Nd',    Nd)
        T           = gf(request.form, 'T',     T)
        tau         = gf(request.form, 'tau',   tau)
        vF          = gf(request.form, 'vF',    vF)
        m_eff_ratio = gf(request.form, 'm_eff', m_eff_ratio)
        a           = gf(request.form, 'a',     a)
        V0          = gf(request.form, 'V0',    V0)
        b           = gf(request.form, 'b',     b)
        bz_lattice  = request.form.get('bz_lattice', bz_lattice)
        bz_a        = gf(request.form, 'bz_a',     bz_a)
        bz_b        = gf(request.form, 'bz_b',     bz_b)
        bz_angle    = gf(request.form, 'bz_angle', bz_angle)
        bz_zones    = int(gf(request.form, 'bz_zones', bz_zones))

    Ec    = 0.0
    Ev    = -Eg
    V0_J  = V0 * eV
    m_eff = m_eff_ratio * m0

    ni    = fb.intrinsic_carrier_concentration(Nc, Nv, Eg, T)
    Ef    = fb.fermi_level_n_type(Ec, Nd, ni, T)
    n, pp = fb.carrier_concentration(Ef, Ec, Ev, Nc, Nv, T)
    mu    = dm.mobility_drude(tau, m_eff) * 1e4
    sigma = dm.conductivity(n, mu)
    l_mfp = dm.mean_free_path(vF, tau)

    bz_plot = plot_brillouin_zones(
        lattice=bz_lattice, a=bz_a, b=bz_b,
        angle=bz_angle, n_zones=bz_zones)

    plots = dict(
        fd      = plt_fermi_dirac(Ef, T),
        dos     = plt_dos(m_eff_ratio, Ec, Ev, T, Ef),
        ni_T    = plt_ni_vs_T(Nc, Nv, Eg),
        mu_T    = plt_mobility_vs_T(Nd),
        carr_T  = plt_carrier_vs_T(Nc, Nv, Eg, Nd),
        cond_T  = plt_conductivity_vs_T(Nc, Nv, Eg, Nd),
        ef_dop  = plt_ef_vs_doping(Nc, Nv, Eg, T),
        band    = plt_band_diagram(Ef, Ec, Ev, Eg),
        rho_dop = plt_resistivity_vs_doping(),
        eg_T    = plt_bandgap_vs_T(Eg),
        kp1d    = plt_kp_1d(a, V0_J, b),
        kp2d    = plt_kp_2d(a, V0_J, b),
        kp3d    = plt_kp_3d(a, V0_J, b),
        recip   = plt_reciprocal_lattice(),
        hall    = plt_hall(Nc, Nv, Eg, Nd),
        iv      = plt_iv_diode(T, Eg),
        schot   = plt_schottky(Eg),
        phonon  = plt_phonon(),
        depl    = plt_depletion(Nc, Nd, Eg),
        bz_plot = bz_plot,
    )

    return render_template(
        'index.html',
        Nc=Nc, Nv=Nv, Eg=Eg, Nd=Nd, T=T,
        tau=tau, vF=vF, m_eff=m_eff_ratio,
        a=a, V0=V0, b=b,
        ni=ni, Ef=Ef, n=n, p=pp,
        mu=mu, sigma=sigma, l=l_mfp,
        bz_lattice=bz_lattice, bz_a=bz_a, bz_b=bz_b,
        bz_angle=bz_angle, bz_zones=bz_zones,
        **plots
    )


@app.route('/api/bz', methods=['POST'])
def api_bz():
    data    = request.get_json() or {}
    lattice = data.get('lattice', 'square')
    a       = float(data.get('a',     1.0))
    b       = float(data.get('b',     1.5))
    angle   = float(data.get('angle', 120.0))
    n_zones = int(data.get('zones',   4))
    try:
        div = plot_brillouin_zones(
            lattice=lattice, a=a, b=b,
            angle=angle, n_zones=n_zones)
        return jsonify({'plot': div})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('summary.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
