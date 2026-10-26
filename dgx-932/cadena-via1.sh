#!/usr/bin/env bash
# DGX-932 vía 1: cuando el volcado del TP=2 acaba bien (y su vuelta deja el TP=2 en A), lanza la calibración en dgx3.
V=/home/user/bench-results/dgx-932
until grep -qE '^[0-9:]+Z (HECHO|FALLO|VUELTA: el TP=2 no)' $V/volcado-tp2-b.log; do sleep 30; done
if grep -q 'volcado completo' $V/volcado-tp2-b.log && grep -q '^[0-9:]*Z HECHO' $V/volcado-tp2-b.log && [ -s $V/volcado/consultas-rango0.jsonl ]; then
  exec $V/calib/calib-dgx3.sh > $V/calib/calib-dgx3.log 2>&1
fi
echo "$(date -u +%H:%M:%SZ) cadena vía 1: el volcado no acabó bien; no lanzo la calibración" >> $V/calib/calib-dgx3.log
