#!/usr/bin/env bash
# DGX-852 14-10 ~08:15Z: la 2x cayó a las ~07:44Z por DiskPressure en dgx1 (las vistas TP=3 del fork llevaron el disco al 88 %;
# kubelet rechazó al worker en bucle). Disco liberado (port tp12 + PLE tp2): la 2x abajo y arriba juntas y la cadena del GDN de nuevo.
set -u
. /home/user/bench-results/dgx-847/ops/residente-lib.sh
R=http://spark2:8000; . $O/espera-2x.sh
reinicia_2x && espera_2x || { log "gdn: la 2x no vuelve"; exit 1; }
log "gdn: 2x E2 de vuelta"
exec bash $O/cadena-gdn.sh
