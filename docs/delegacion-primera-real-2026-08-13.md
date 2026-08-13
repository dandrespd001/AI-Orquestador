# Primera delegación real por el embudo: qué falló y qué se arregló

Fecha: 2026-08-13. Encargo: un cambio de ~20 líneas en
`tests/perf/bench_multisemilla.cpp` de CHUNSA001, con los anclajes ya resueltos
por el Arquitecto (número de línea y patrón literal de cada punto de edición).

Es la primera vez que se usa `tools/chunsa_ai.sh run` para trabajo real. El
encargo era pequeño **a propósito**: probar el embudo, no exprimirlo.

## Lo que pasó

| Intento | Ruta | Resultado |
|---|---|---|
| 1 | `deepseek-v4-flash` (`patch_api_writer`) | `MODEL_PROCESS_FAILED`, 93 s, 0 bytes |
| 2 | idem, tras dar códigos a los errores | `PROVIDER_EMPTY_PATCH`, 4,3 s |
| 3 | idem, tras separar las dos causas | **`REASONING_BUDGET_EXHAUSTED`** |
| 4 | `gpt-5.6-sol` (`native_writer`), reencaminado por política | `PATCH_READY`, gates verdes |

## La causa raíz, medida

Reproduciendo la petición exacta que fallaba:

```
finish_reason      = "length"
content            = 0 caracteres
reasoning_content  = 48 838 caracteres
completion_tokens  = 16 384   de los cuales reasoning_tokens = 16 384
prompt_tokens      = 25 682
```

**El razonamiento consumió el presupuesto entero y el modelo no emitió ni un
token de respuesta.** No falló el modelo, ni la credencial, ni la red, ni el
tamaño del contexto: la misma ruta contestaba un sondeo en 1 068 ms y respondía
correctamente a una pregunta sobre 60 KB de contexto.

Dos arreglos plausibles, **descartados midiéndolos**:

- `max_tokens` de 16 384 a **32 768**: el razonamiento se sigue cortando en
  16 384 y `content` sigue vacío.
- Enviar `reasoning_effort` (`low`, `medium`): la pasarela **lo ignora**;
  `reasoning_tokens` sale 16 384 en los dos casos.

El tope es de aguas arriba, así que el arreglo no está en los parámetros de la
petición: está en **el tamaño del encargo**.

## Por qué esto importa más allá de este encargo

Tres de los cuatro modelos «rechazados» del perfil `chunsa` —`qwen-web-max`,
`kat-coder-free`, `local-qwen36-agent`— tienen **exactamente esta firma**:
`FAILED`, `provenance_verified: false`, latencias de 15-21 s y ningún motivo
registrado. Es muy probable que nunca estuvieran rotos: estaban **sin
diagnosticar**, porque `patch_worker.py` colapsaba todo error a una sola cadena.

Y hay un límite estructural que conviene tener escrito: **`_bounded_context`
incluye SIEMPRE el fichero de destino**, así que un encargo de escritura no se
puede aligerar quitándolo de `input_paths`. Pedir un diff unificado contra un
fichero de 71 KB es caro en salida por construcción. Para ficheros grandes, el
escritor **nativo** —que edita en el worktree en vez de emitir el diff entero—
no tiene ese techo.

## Lo que se arregló

1. **`PatchApiError` lleva un `code`** de un conjunto cerrado, elegido por
   nosotros y sin texto del proveedor. Los 13 sitios de `raise` lo fijan.
   `_http_code()` separa 401/403 (credencial), 429 (cuota) y 5xx (proveedor).
2. **`patch_worker` propaga el código** por el borde del subproceso, en vez de
   `type(exc).__name__`. El mensaje, que puede citar al proveedor, se queda
   dentro.
3. **El clasificador acepta ese código** cuando llega. Las heurísticas por
   subcadena que había nunca podían acertar: buscaban prosa del proveedor que a
   propósito no cruza el borde.
4. **`REASONING_BUDGET_EXHAUSTED`** distingue la firma de arriba de una
   respuesta genuinamente vacía, porque el arreglo es distinto: encoger la
   petición, no reintentar.
5. **Cuatro tests nuevos** en `tests/test_patch_api.py`, que no tenía ninguno
   para esta ruta. Suite: **69/69**.

## Un hueco del envoltorio, encontrado al arbitrar

El intento 4 terminó en `PATCH_READY` con los gates en verde… y **el parche se
perdió**. `ai-orchestrator run` sólo escribe el diff si se le pasa `--output`, y
`tools/chunsa_ai.sh` no exponía ese argumento: el resultado traía el `sha256`
del parche y nada más.

Un encargo de 11 minutos que pasa los gates y cuyo diff no se puede leer es
inútil, porque **la revisión no se delega**. El envoltorio acepta ahora una ruta
de salida opcional: `tools/chunsa_ai.sh run <spec.json> [salida.diff]`.

Es el segundo hueco del mismo tipo en este envoltorio, después del de `doctor`:
la política estaba escrita y la herramienta no permitía cumplirla.
