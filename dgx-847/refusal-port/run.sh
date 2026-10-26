#!/usr/bin/env bash
# Entrypoint del pod: aplica el mod y deja que la receta de ursuciprian haga el
# resto. Los args del Deployment pasan tal cual, asi que la receta no se toca:
#   args: [bash, /opt/ursu-mod/run.sh, /etc/ursu/serve-2x.sh]
# (y serve-1x.sh para el 1x). Si el mod no aplica, el pod muere aqui y no sirve
# con el dial ausente: fail-closed, igual que en produccion.
set -euo pipefail
cd "$(dirname "$0")"
python3 install.py
exec bash "$@"
