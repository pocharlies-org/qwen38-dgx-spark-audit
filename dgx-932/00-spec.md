Rol: cto · Fecha: 2026-10-15 · Sesión: 5d35c318-9261-4879-b52f-24fa3c4db837 · Estado: LISTO

# DGX-932 · B2 con autoajuste BF16 en dgx3

## Objetivo
Saber si el GDN b12x entero (prefill y decode, con registros compactos y checkpoints diferidos) con estado BF16 alcanza o supera al mixto (A) cuando sus kernels están ajustados para BF16.

## Criterios de aceptación
- [ ] El TP=1 de dgx3 arranca en B2 con autoajuste y la caché BF16 queda en el disco del nodo; el TP=2 sigue sirviendo todo el tiempo.
- [ ] Medido igual que A: decode c1/c4/c8, prefill en frío 2K/16K/64K, greedy y tools.
- [ ] Tabla B2 ajustado frente a A (TP=1: 47/102/137 tok/s; 1,40/8,32/32,4 s) con propuesta.
- [ ] Producción del TP=1 vuelve a A salvo decisión de Dani.
