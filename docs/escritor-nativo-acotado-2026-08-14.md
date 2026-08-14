# El escritor nativo acotado: un modelo externo que ejecuta, sin llave del repo

**Fecha:** 2026-08-14 · **Suite:** 83/83 (antes 72/72) · **Canario:**
`deepseek-v4-pro` 5/5, `promotion_ready=true`
(`docs/receipts/deepseek-v4-pro-native-2026-08-14.json`).

## El diagnostico que casi fue el equivocado

La revision decia que `_qwen_sandbox_command(..., writable=False)` tenia el
parametro cableado a False y que habilitarlo abriria una via de delegacion
capaz de ejecutar. Es cierto que estaba cableado. **No es cierto que
habilitarlo hiciera nada**, y comprobarlo costo una consulta:

```
interseccion (transporte qwen) x (execution_profile native_writer): []
```

Los cinco `native_writer` de la flota estaban todos en `codex_cli` o
`claude_cli`, transportes que ni siquiera llegan a esa linea. Los nueve de
transporte Qwen eran `patch_api_writer` sin excepcion, y en `adapters.py:259`
un `patch_api_writer` que escribe se desvia al `patch_worker` antes. El arreglo
tal como estaba propuesto habria sido el cuarto inerte de la serie.

## Los dos privilegios que venian empaquetados

`native_writer` no era un permiso, eran dos:

- **(a)** editar el arbol de trabajo directamente — lo que permite iterar,
  hacer fase roja y tocar ficheros que un escritor por parche no aguanta;
- **(b)** auto-aplicar el resultado al repo de origen sin revision
  (`adapters.py:565`).

Un modelo externo necesita (a). Nadie ha medido nunca que merezca (b). El
perfil nuevo **`native_patch_writer`** concede (a) y niega (b): recibe montaje
escribible y `--approval-mode auto-edit`, pero la comprobacion de auto-aplicar
sigue siendo igualdad exacta con `"native_writer"`, asi que termina siempre en
`PATCH_READY` con `EXTERNAL_WRITER_REVIEW_REQUIRED`.

Ese test **paso en verde desde el primer momento**, y esa es su funcion: fija
que anadir el perfil no concede auto-aplicar por accidente.

## Por que el primer canario fallo, y por que no era el sandbox

Primera pasada: `NO_CHANGES`. Sondeado el montaje a mano, escribia y el fichero
aparecia en el host — el sandbox estaba bien. La causa estaba en la respuesta
del propio modelo:

```json
"tools":["read_file","edit","notebook_edit","get_goal","update_goal"]
"thinking":"The task is simple... Let me use write_file."
```

**Qwen 0.21.12 en modo `--bare` no expone `write_file`.** Comprobado tambien
sin ninguna exclusion: la lista solo gana `run_shell_command`, que nosotros
vetamos a proposito. Las unicas herramientas de escritura son `edit` y
`notebook_edit`.

El canario pedia **crear** un fichero, que es justo el caso que `edit` hace
mal. El trabajo real es **editar** `step.hpp` y `ai_stub.hpp`. Cambiado el
canario a modificar un fichero rastreado, la misma ruta sale a la primera y
5/5.

## La brecha que un montaje escribible vuelve viva

`collect_patch` listaba los no rastreados con `--exclude-standard`, asi que un
fichero cubierto por `.gitignore` no pasaba por la allowlist. Nunca llega al
diff — pero si vive en el arbol donde corren los gates, y un gate ejecuta el
codigo del modelo.

**Quitar `--exclude-standard` a secas habria roto el bucle de reparacion.** El
orden real es escritor -> `collect_patch` -> gates -> reparacion -> escritor, y
en la segunda vuelta los `build/` y `__pycache__` que dejaron los gates siguen
ahi: `collect_patch` los habria leido como rutas cambiadas y habria lanzado
`WRITE_SCOPE_VIOLATION` por obra de los gates, no del modelo.

La regla correcta no es «no hay ficheros ignorados» sino **«el escritor no creo
ficheros ignorados»**, y se expresa con `snapshot_ignored()` antes de cada
pasada. Dos tests fijan las dos mitades: el fichero ignorado que crea el
escritor se rechaza; el que dejo un gate no rompe la reparacion.

## Permitido no es promovido

`run_writer_canaries` leia el perfil de `models.toml`, asi que medir la ruta
nativa exigia promoverla antes — el orden contrario al que pide la politica.
El parametro `--execution-profile` mide un perfil que el modelo aun no tiene.
El recibo ahora registra `execution_profile` **y**
`configured_execution_profile`: un recibo que no dice que midio no es
evidencia de nada.

## Alcance de la promocion

Solo `deepseek-v4-pro`. Los otros ocho modelos de transporte Qwen siguen
entregando parche hasta que su propio canario lo justifique. Un cambio de
perfil no se hereda por familia ni por proveedor.

**`config/models.toml` no es la configuracion viva.** `config_root()` apunta a
`~/.config/ai-orchestrator`; sin `config config sync --apply` la promocion
queda escrita y sin efecto. Comprobado en la maquina despues de sincronizar:

```
qwen con montaje escribible: ['deepseek-v4-pro']
qwen que auto-aplican      : []
```

## Lo que esto no arregla

La restriccion medida que hundio las delegaciones anteriores era el **tamano de
fichero**: `step.hpp` son 130 KB y `ai_stub.hpp` 2600 lineas, y un escritor por
parche agota su presupuesto de razonamiento antes de terminar. Un escritor
nativo lee por trozos y edita por trozos, que es la razon por la que deberia
aguantarlos — **pero eso no esta medido todavia**. El canario prueba que la
ruta escribe, no que aguante un fichero de 130 KB. Esa es la siguiente
medicion, y hasta hacerla el veto sobre los tres ficheros grandes sigue en pie.
