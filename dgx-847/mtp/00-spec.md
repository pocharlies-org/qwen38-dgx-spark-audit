# DGX-852 · qwen38 TP=3: MTP k=2-4 y checkpoint NVFP4 de ursuciprian sobre nuestra pila

**Pedido:** Dani, 13-10-2026, tras DGX-847: «Subamos el mtp y probemos su checkpoint nvfp4 sobre nuestra pila».

## Qué
1. **MTP k=2, 3 y 4** en el residente qwen38 TP=3 (`SPECULATIVE_CONFIG.num_speculative_tokens`, hoy 1). Ventana en los Sparks:
   root y ai sin auto-sync; réplicas por el árbitro (jobs de pausa/resume). Por cada k: decode v4 (c1/4/8/10 ×3, misma sonda
   que DGX-817) y llama-benchy medium (pp2048/tg128, profundidad 0 y 65535, c1/4/8, 3 runs). Script: `dgx-847/ops/vuelta-residente.sh`.
2. **Checkpoint `ursuciprian/Qwen3.8-Flash-Next-NVFP4-GDN-MSE`** (NVFP4) en nuestra pila de vLLM a TP=3: viabilidad primero
   (formato de cuantización, reparto entre 3 rangos, cabeza MTP, vocabulario de borrador), medición después si se puede.

3. **Combinación propuesta (Dani: «haz lo que propones», 13-10, en vivo):** en nuestro TP=3, checkpoint NVFP4 de ursu
   (vista TP=3) + paquete `b12x` 1.3.0 (extra `vllm[b12x]`, instalado en el arranque por `launch.sh`, sin imagen nueva) para
   MoE/GEMM + estado SSM en BF16 con GDN en triton (lo nuestro; E2 de DGX-847 probó que la mezcla funciona) + MTP k=2. Batería
   de calidad completa (7 suites) sobre esa combinación.
4. **Vocabulario recortado del borrador MTP con castellano:** portar `VLLM_MTP_DRAFT_VOCAB` de su fork a nuestra vLLM (parche de
   arranque) y una lista de 131 072 ids construida con nuestro tráfico (castellano, código, tools). Su lista cubre el 100 % del
   inglés y el 96,5 % del castellano técnico; objetivo ≥ 99,5 %. Con el borrador más barato, repetir MTP k=3 y k=4.

## Criterios de aceptación
- [ ] Tabla por k: tok/s c1/4/8/10, aceptación del MTP por posición, benchy medium (incluida la 1.ª respuesta a 64K con caché).
- [ ] La k que más sube el decode sin empeorar el contexto largo con caché queda fijada por PR en k8s-ai-pocharlies.
- [ ] Combinación NVFP4 + b12x + BF16/GDN triton + MTP k=2 sirviendo en TP=3, medida (decode v4 + benchy) y con la batería de calidad sin regresión frente a DGX-817
- [ ] Vocabulario del borrador con castellano (≥ 99,5 % de cobertura) portado y medido con MTP k=3/k=4
- [ ] Veredicto del checkpoint NVFP4: viable o no, con la causa; si es viable, la misma tabla.

## Referencia
DGX-847 (recetas de ursuciprian frente a nuestro TP=3): resultados en `~/bench-results/dgx-847/`.
