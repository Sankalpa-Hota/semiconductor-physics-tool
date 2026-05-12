#!/usr/bin/env sh
# Render / PaaS: set Start Command to:  bash start.sh
# (Dashboard Start Command overrides Procfile—do not use "python main.py" in production.)
set -eu
exec gunicorn --bind "0.0.0.0:${PORT}" --workers 1 --timeout 180 --graceful-timeout 60 main:app
