# main.py
from flask import Flask, render_template, request
import fermi_boltzmann as fb
import drude_model as dm
import phonon_scattering as ps
import kronig_penney as kp
import reciprocal_lattice as rl
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    # --- Default parameters ---
    Nc, Nv, Eg, Nd, Ec, Ev, T = 1e19, 6e18, 0.66, 1e18, 0.0, -0.66, 300
    tau, vF, m_eff = 0.24e-15, 1e6, 0.26*9.11e-31
    a_default, V0_default, b_default = 5e-10, 10, 2e-10  # Kronig-Penney defaults

    # --- Overwrite with user inputs ---
    if request.method == "POST":
        Nc = float(request.form.get("Nc", Nc))
        Nv = float(request.form.get("Nv", Nv))
        Eg = float(request.form.get("Eg", Eg))
        Nd = float(request.form.get("Nd", Nd))
        T = float(request.form.get("T", T))
        tau = float(request.form.get("tau", tau))
        vF = float(request.form.get("vF", vF))
        m_eff = float(request.form.get("m_eff", m_eff))
        a_default = float(request.form.get("a", a_default))
        V0_default = float(request.form.get("V0", V0_default))
        b_default = float(request.form.get("b", b_default))

    # --- Physics Computations ---
    ni = fb.intrinsic_carrier_concentration(Nc, Nv, Eg, T)
    Ef = fb.fermi_level_n_type(Ec, Nd, ni, T)
    n, p = fb.carrier_concentration(Ef, Ec, Ev, Nc, Nv, T)
    mu = dm.mobility_drude(tau, m_eff) * 1e4  # cm^2/Vs
    sigma = dm.conductivity(n, mu)
    l = dm.mean_free_path(vF, tau)

    # --- Example Plot 1: ni vs T ---
    T_values = np.linspace(200, 800, 200)
    ni_values = np.array([fb.intrinsic_carrier_concentration(Nc, Nv, Eg, Ti) for Ti in T_values])
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=T_values, y=ni_values, mode='lines+markers', name='ni(T)'))
    fig1.update_layout(title='Intrinsic Carrier Concentration vs Temperature',
                       xaxis_title='Temperature (K)', yaxis_title='ni (cm^-3)')
    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)

    # --- Example Plot 2: Mobility vs Temperature (Phonon scattering) ---
    mu_values = np.array([ps.mobility_phonon(Ti, m_eff) for Ti in T_values])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=T_values, y=mu_values, mode='lines+markers', name='μ(T)'))
    fig2.update_layout(title='Mobility vs Temperature (Phonon Scattering)',
                       xaxis_title='Temperature (K)', yaxis_title='Mobility (cm^2/V.s)')
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)

    # --- Kronig-Penney Plots ---
    kp_plot_1d = kp.plot_kronig_penney_1d(a=a_default, V0=V0_default, b=b_default)
    kp_plot_2d = kp.plot_kronig_penney_2d(a=a_default, V0=V0_default, b=b_default)

    return render_template("index.html",
                           Nc=Nc, Nv=Nv, Eg=Eg, Nd=Nd, T=T,
                           ni=ni, Ef=Ef, n=n, p=p,
                           mu=mu, sigma=sigma, l=l,
                           plot_div1=plot_div1, plot_div2=plot_div2,
                           kp_plot_1d=kp_plot_1d, kp_plot_2d=kp_plot_2d,
                           a=a_default, V0=V0_default, b=b_default)

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
