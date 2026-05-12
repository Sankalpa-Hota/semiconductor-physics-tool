# OPEN-Quantum

OPEN-Quantum is a web application built with Flask, Plotly, and Python. You can explore semiconductor models through the browser: change parameters and see plots update.

It covers topics such as:

- Fermi–Dirac and Boltzmann carrier statistics
- Drude model (conductivity and mobility)
- Phonon scattering and mean free path
- Kronig–Penney band structure
- Reciprocal lattice geometry in 3D

The audience is students, teachers, and anyone learning solid-state or semiconductor physics.

## Live demo

https://semiconductor-physics-tool.onrender.com

## Source code

https://github.com/Sankalpa-Hota/semiconductor-physics-tool

## Features

- Web interface for editing simulation inputs
- Each chart has **Render this** / **Close this** controls; only open figures are generated on **Compute parameters**, which keeps CPU and RAM low on small hosting plans.
- Physics-oriented defaults (silicon-oriented starting parameters)
- Suited for teaching and self-study

## Tech stack

- Backend: Python, Flask
- Plots: Plotly, NumPy, Shapely (Brillouin zone geometry)
- Frontend: HTML, CSS, Jinja templates, JavaScript
- Production server: Gunicorn (recommended for deployment)

## Running locally

1. Create a virtual environment (optional but recommended).
2. Install dependencies:

   `pip install -r requirements.txt`

3. Start the development server:

   `python main.py`

   Then open the URL shown in the terminal (by default port 5000 unless `PORT` is set).

## Deployment on Render

Render and similar hosts need a production process, not the Flask development server. Use a start command that runs Gunicorn, for example:

`gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 180 main:app`

or run the provided `start.sh` script if your host uses it as the start command.

Python version: Render reads `.python-version` in the repository root (see also `INFO.txt` for brief deployment notes).

Build step example:

`pip install -r requirements.txt`

## License

MIT License. See `LICENSE`.
