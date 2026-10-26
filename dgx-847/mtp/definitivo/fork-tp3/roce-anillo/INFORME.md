# DGX-852 · RoCE en el triángulo sin switch (parche de arranque TP=3)

**Bug raíz.** El proxy del fork empareja HCAs por índice: el QP local `h` se
conecta al slot `h` del blob remoto y el stripe reparte el payload entre las dos
puertas de cada vecino (`_roce_proxy.c`, `roce_connect`/`post_op` originales). En
el anillo cada vecino vive en un puerto distinto (dgx1.f1 ↔ dgx3.f0), así que en
TP=3 la mitad del payload sale por la puerta que no llega a ese rango: spin
eterno o corrupción silenciosa. **Arreglo:** enrutar por par `(peer → HCA local,
HCA remota)`, una puerta por vecino, con el mapa en `B12X_ROCE_PEER_HCAS` por
nombres de dispositivo. Sin la variable, el comportamiento es idéntico al
original (TP=2 byte a byte, demostrado en CPU).

## Ficheros (este directorio)

| fichero | qué es |
|---|---|
| `patch_roce_ring.py` | parche de arranque: 15 ediciones ancladas, idempotente, fail-closed (exit 3 si un ancla no casa, no escribe nada). Puerta: `B12X_ROCE_RING=1` (vacío → inerte exit 0). `_proxy.py` recompila el `.c` por sha256, así que el parche se recoge solo en el primer arranque. |
| `roce_ring_map.py` | genera `B12X_ROCE_HCA` + `B12X_ROCE_PEER_HCAS` del rango desde `ring.conf` (mismo mecanismo que `QWEN_NCCL_RING_HOST`/`nccl_ring_graph.py`). `--host dgx2\|dgx1\|dgx3`, `--all`, `--check`, `--half A\|B`. Rangos: dgx2=0, dgx1=1, dgx3=2 (como `ursu-3x.yaml`). |
| `ring.conf` | copia congelada de `dgx-infra/scripts/roce-link/ring.conf` (commit 3106496b, sha256 del cuerpo 14fd2930…; 4 líneas de procedencia encima). |
| `test_patch_roce_ring.py` | suite de CPU (70 comprobaciones, `python3 -I` → exit 0). |
| `test_roce_ring.py` | prueba de red dentro de la imagen en los 3 Sparks (la lanza Dani). |
| `stub_verbs/` | doble de libibverbs + banco de CPU (`harness_roce_post.c`): monta los 3 rangos en un proceso y registra cada `ibv_post_send`/`ibv_modify_qp`. |

## Ediciones (líneas sobre el árbol YA parcheado)

`b12x/comm/roce/_roce_proxy.c`
- **:44** `ROCE_ABI_VERSION` 3→4 (el loader de Python rechaza el .so viejo).
- **:87** campo `connected` (congelea las rutas tras el connect).
- **:90-94** ctx: `peer_n_hca[16]`, `peer_h_local[16][2]`, `peer_h_remote[16][2]`.
- **:273-277** `roce_create`: init identidad (`peer_n_hca=2`, `h_local=h_remote=k`) → sin mapa, todo el camino viejo.
- **:289-303** `roce_set_peer_hca(ctx, peer, lh, rh)`: un cable por par; rechaza índices malos y llamadas tras `start`/`connect` (fail-closed).
- **:320-326** `connect_qp(..., int rh)`: usa `mtu[rh]/qp_num[rh]/lid[rh]/gid[rh]` del blob (la HCA remota del par, no el índice local).
- **:385-395** `roce_connect`: conecta solo los pares enrutados (identidad ⇒ los mismos QP que antes) y marca `connected`.
- **:436-517** `post_op`: el stripe recorre las HCAs enrutadas del par; con ruta (route_n=1) el payload **completo** sale por esa puerta y se escriben **todos** los slots de flag del receptor (`flag[src][slot][h]`, h=0..n_hca-1) sobre el mismo QP, solo el último señalado — entrega RC in-order ⇒ el kernel (`_oneshot_cute.py:261-271`, sin tocar) ve cubiertos todos los flags. `rkey = peer_rkey[rh][p]`: el rkey por HCA ya existía, no hace falta nada nuevo en la memoria registrada.

`b12x/comm/roce/_proxy.py`
- **:115** binding ctypes de `roce_set_peer_hca`.
- **:134** check ABI `!= 4`.
- **:218-228** `Proxy.set_peer_hcas({peer: (lh, rh)})`.

`b12x/comm/roce/roce_oneshot.py`
- **:120-147** `parse_peer_hcas("1:f0:f0,2:f1:f1")` → dict; malformado = error (no hang).
- **:149-201** `validate_peer_routes`: simetría cruzada (mi remota = su local y viceversa) + nombres existentes contra lo que publica cada rango.
- **:372-377** lee `B12X_ROCE_PEER_HCAS` (solo si `B12X_ROCE_RING=1` ya cargó el parche).
- **:401** config exchange: `peer_hcas_present` → todo o nada, error colectivo si unos rangos traen mapa y otros no.
- **:432-450** segundo `_exchange` con `hca_names` + `peer_hcas` ANTES de `connect`, valida, y aplica `set_peer_hcas`.

vLLM (`b12x_roce_all_reduce.py`): sin cambios — `SUPPORTED_WORLD_SIZES` ya llega a 16.

## Pruebas de CPU — `python3 -I test_patch_roce_ring.py` → **70 ok, 0 fallos**

- Puerta inerte sin `B12X_ROCE_RING`; valor inválido → exit 2; ancla corrupta en árbol virgen → exit 3 **sin escribir ningún fichero**; idempotencia (segundo run no reescribe); `--dry-run` no toca nada; `py_compile` de los 2 `.py`.
- **No hay cabeceras de libibverbs en esta x86** (`/usr/include/infiniband/verbs.h` no existe), así que el `gcc -fsyntax-only` del `.c` original y del parcheado se hace contra el doble `stub_verbs/` (mismos nombres de struct/miembro que la ABI real). Además el banco **enlaza y ejecuta** el proxy completo.
- **Identidad demostrada:** el stream de RTR+WR+STATS del `.c` parcheado en modo identidad es **byte a byte idéntico** al del `.c` original (8 WR, 4 terminaciones) — sin `B12X_ROCE_PEER_HCAS` el TP=2 no cambia ni un post.
- **Enrutado:** 6 WR (payload 4096 B + 2 flags por par), cada par solo por su QP (1000/1101), `raddr` en la ventana `recv` del vecino, flags cubriendo los 2 slots, último señalizado (flags 4→6), ningún WR por las puertas equivocadas, 2 terminaciones.
- Generador: los 3 mapas esperados, simetría A y B, mitad B sobre `roceP2p1s0f*`, y las rutas del banco == las que imprime `roce_ring_map.py`.

## Cómo lanzar la prueba (Dani; yo no toco el clúster)

Con los 3 pods de `fork-tp3` arriba (`aplicar-3x.sh`) y este directorio copiado a `/tmp/roce-anillo` de cada pod:

```bash
# EN CADA POD, los tres a la vez (NODE_RANK y QWEN_NCCL_RING_HOST vienen del manifiesto):
B12X_ROCE_RING=1 python3 -I /tmp/roce-anillo/patch_roce_ring.py
eval "$(python3 -I /tmp/roce-anillo/roce_ring_map.py --host "$QWEN_NCCL_RING_HOST")"
NODE_RANK=$NODE_RANK MASTER_ADDR=spark2 MASTER_PORT=25001 \
  GLOO_SOCKET_IFNAME=enP7s7 B12X_ROCE_SPIN_LIMIT=30000000 \
  python3 -I /tmp/roce-anillo/test_roce_ring.py
```

El host network del pod ya expone las dos puertas del anillo; `test_roce_ring.py`
aborta sin `B12X_ROCE_PEER_HCAS` o sin el parche aplicado (preflight
`B12X_ROCE_RING_POST` en el `.c` instalado). Comprueba fp32 y bf16 de 4 KB a
2 MB contra la suma esperada (6.0), latencia µs/op (100 iters, cuda events) y
bytes por HCA (ambas puertas del rango deben cargar tráfico, una por vecino).

## Riesgos

1. **El mapa debe ser simétrico o el arranque aborta** (validación cruzada en el exchange): un `ring.conf` desactualizado o un `B12X_ROCE_HCA` reordenado se convierte en error colectivo en vez de cuelgue. Es el diseño elegido; no hay modo "solo local".
2. **ABI 4 rompe con `.so` cacheado viejo**: `_proxy.py` recompila por sha256 del fuente, pero si alguien fija el directorio de compilación entre imágenes con ABI 3, el check `!= 4` aborta con mensaje claro. Sin riesgo silencioso.
3. **Ruta de un flag por par = 2 WR extra por operación** (payload sin stripe a un vecino). En TP=3 no hay stripe posible (un cable por vecino); el coste es 8 bytes, no ancho de banda.
4. **Mitad A fija** (rocep1s0f*); si un cable de la mitad A se muere, hay que relanzar con `--half B` en los tres rangos a la vez.
5. **`discover_hcas` sigue recortando a 2 HCAs** (`[:2]`): por eso la receta fija `B12X_ROCE_HCA` explícitamente; con NCCL_IB_HCA suelto el orden puede no ser el del mapa → la validación lo detecta y aborta.
6. No probado en hardware (límites del encargo): la verificación real es el comando de arriba en los 3 Sparks.
