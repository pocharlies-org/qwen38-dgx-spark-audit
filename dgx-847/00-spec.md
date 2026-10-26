Rol: cto · Fecha: 2026-10-12 · Estado: LISTO

# DGX-847: receta de ursuciprian para qwen38, TP=2 en dgx1+dgx2 y 1× en dgx3, con nuestra batería

Repo: ninguno (prueba en vivo con kubectl; ops en ~/bench-results/dgx-847/ops). Pedido por Dani el 12-10 («montalo»).

## Qué y por qué
Dani quiere valorar dejar dgx1+dgx2 con la receta TP=2 de ursuciprian y dgx3 con su receta 1×, para poder parar dgx3 cuando
quiera usarlo para vídeo. Antes de cambiar nada en producción, se miden las dos recetas TAL CUAL (v2.0.0 2× y v2.2.0 1×, mismas
imágenes por digest, mismos flags y env, lanzadas como sparkrun 0.3.6) con la misma batería que qwen38 TP=3 en DGX-817/820.

## Criterios
- [ ] Ventana: ArgoCD sin auto-sync (root y ai), árbitro en sync-paused, residente a 0; y al cerrar todo devuelto (auto-sync, sync de ai, árbitro ready, qwen38 TP=3 /health 200) antes de las 03:00 Madrid.
- [ ] Parser de tools: chat completions con tools, con y sin stream, dos llamadas en paralelo, args JSON válidos y vuelta final en texto; y Claude Code por /v1/messages si el servidor lo sirve. En las dos recetas.
- [ ] decode v4 (c1/4/8/10 + 64K c8/c10), llama-benchy medium con prefix caching y la batería de calidad de 7 suites en las dos recetas, con la calidad cortada al cierre de la ventana si no da tiempo.
- [ ] Comparación con qwen38 TP=3 4K (DGX-817) en una nota adjunta, y la decisión para Dani (adoptar o no, qué falta: dial de refusal, empaquetado k8s, regla del árbitro con dgx3 opcional).

## Cómo se verifica
Resultados en ~/bench-results/dgx-847/{2x,1x}/ (tools-check.json, decode/, benchy/, quality/) y logs de ventana/cierre en ops/.
