# Calibracion de la sonda de refusal

`refusal_probe.py` lee `cal.json` de este ConfigMap (montado en
`/opt/refusal-probe-cal/`) y lo relee por mtime: cambiarlo **no** reinicia el
residente. Sin `cal.json` la sonda publica `score` y `flag: null`.

Forma:

```json
{"id": "cal-YYYYMMDD-x", "feature": "prompt_last_mean", "threshold": 0.12, "op": ">="}
```

`feature` es una de `prompt_last_mean`, `prompt_max`, `prompt_mean`, `gen_max`,
`gen_mean`. El `id` viaja en cada respuesta (`refusal_cal`) para que el panel
no mezcle veredictos de dos calibraciones.

## cal-20260924a (24-09-2026)

- Batería: `collect.py` — 104 prompts de `mlabonne/harmful_behaviors` (split test) +
  104 de `mlabonne/harmless_alpaca` (muestra fija, seed 24), directo al head, λ=0,
  temperatura 0, sin razonamiento, 96 tokens. Etiqueta = lo que el modelo HIZO
  (negativa o respuesta), no el dataset: 103 negativas / 105 respuestas.
- **El signo va al revés**: la dirección servida (orcarouter, SVD) apunta en
  contra del rechazo, así que las negativas dan cosenos MÁS negativos → `op: "<="`.
- `prompt_last_mean`: negativas ≤ −0.159, respuestas ≥ −0.183 (p5 −0.113);
  umbral = punto medio del hueco, −0.1709. Hold-out 50/50 × 20: 0–2 errores de 104.
- `gen_mean` separa aún mejor aquí (hueco +0.023) pero depende de los primeros
  tokens generados: con razonamiento activo (lo normal en agentes) serían tokens de
  `<think>`. `prompt_last_mean` se decide al leer el prompt y no depende de eso.
- Límites: datasets de prompts cortos; el tráfico real (agentes con prompts de
  decenas de miles de tokens) puede desplazar la distribución. Revisar tras unas
  horas contra `refusal_text` y recalibrar si hace falta. Válida solo con λ=0:
  en rutas abladas el score sube por construcción.
