# AI Orchestrator

Control plane local para seleccionar y auditar modelos, proveedores y
capacidades oficiales, comunitarias, privadas y locales. DeepSeek Harness es la
interfaz principal; la política, las credenciales, el enrutado y la auditoría
permanecen en Python. GPT se ejecuta con Codex, Claude con Claude Code, DeepSeek
con Qwen Code sobre OmniRoute, MiniMax mediante Responses API, Kimi con su CLI
nativo y GLM 5.2 promocional mediante FX. Cada repositorio puede endurecer
estas reglas con un perfil propio.

El sistema nunca guarda prompts ni respuestas en telemetria. Registra hashes,
modelo solicitado y observado, latencia, estado de cuota y gates. No realiza
fallback silencioso ni convierte una suscripcion en pago por uso.

## Inicio rapido

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
AI_ORCHESTRATOR_CONFIG="$PWD/config" .venv/bin/ai-orchestrator doctor --profile default
AI_ORCHESTRATOR_CONFIG="$PWD/config" .venv/bin/ai-orchestrator route --spec examples/task.json
.venv/bin/ai-orchestrator control providers
.venv/bin/ai-orchestrator harness-spec
```

Instale la configuración versionada antes de usar el comando sin el override:

```bash
.venv/bin/ai-orchestrator config sync
.venv/bin/ai-orchestrator config sync --apply
```

`sync` sólo informa deriva. `--apply` valida primero, crea una copia privada de
los archivos sustituidos, instala de forma atómica y nunca toca
`credentials/`. `doctor` falla si la copia activa deja de coincidir con la
configuración versionada.

## Plano modular y Harness

Harness está fijado a `@deepseek-ai/dsh@0.1.0-rc.7`, escucha explícitamente en
`127.0.0.1:3080` y usa un perfil administrativo sin código comunitario. Se
comunica con el control plane por un socket Unix privado `0600`; ese broker sólo
expone inventario, diagnóstico y simulación. No expone lectura de secretos,
instalación ni activación.

Las unidades de usuario son `systemd/ai-orchestrator-broker.service` y
`systemd/ai-orchestrator-harness.service`. La segunda usa el paquete ya
instalado en la caché de npm (`npx --offline`); las actualizaciones se preparan
en un slot paralelo y requieren compatibilidad, seguridad, evaluación y
aprobación humana antes de promoverse:

```bash
.venv/bin/ai-orchestrator component inventory
.venv/bin/ai-orchestrator component stage --name deepseek-harness \
  --artifact /ruta/al/paquete --version 0.1.0-rc.8 --sha256 HASH
.venv/bin/ai-orchestrator component validate --name deepseek-harness \
  --compatibility-passed --security-passed --evaluation-passed
.venv/bin/ai-orchestrator component promote --name deepseek-harness --approve
```

El panel local (`ai-orchestrator panel`) muestra proveedores, estado de
credenciales, modelos, rutas, consumo, perfiles, extensiones, actualizaciones y
sesiones. Los campos de credencial son de sólo escritura; cada arranque entrega
por terminal una capacidad humana válida para una sola mutación.

La superficie MCP (`ai-orchestrator mcp-serve`) ofrece inventario, diagnóstico y
simulación de rutas sin secretos. También expone dos acciones explícitas:
`orchestrator_run_task` reutiliza la política, adaptadores, worktrees, gates y
telemetría del proyecto para todos los proveedores configurados;
`orchestrator_delegate_abacus` usa el login Pro oficial en modo acotado. La MCP
nunca activa extensiones, escribe credenciales ni aplica un parche al repositorio
fuente: cualquier escritura delegada termina en `PATCH_READY`. Harness recibe la
superficie administrativa de consulta a través del broker Unix; Codex, Claude,
Kimi, Qwen, OpenCode, Abacus y FX pueden recibir configuraciones mínimas por perfil
desde el mismo catálogo:

```bash
.venv/bin/ai-orchestrator client-config --client codex --profile default
.venv/bin/ai-orchestrator capability --package research --client kimi
```

La arquitectura completa y el flujo de confianza están en
[Control modular y extensiones comunitarias](docs/modular-control-plane.md).

## Extensiones comunitarias

Plugins, skills y MCP pueden proceder de GitHub, Git privado, rutas locales,
npm, PyPI, `stdio` o HTTPS. Ser comunitario no los invalida: determina el
aislamiento y la aprobación. El ciclo obligatorio es descubrir, copiar a
cuarentena, inspeccionar, probar de forma aislada, aprobar el hash, activar por
perfil y conservar rollback:

```bash
.venv/bin/ai-orchestrator extension discover --manifest extension.json
.venv/bin/ai-orchestrator extension quarantine --manifest extension.json --payload /ruta/descargada
.venv/bin/ai-orchestrator extension inspect --id example
.venv/bin/ai-orchestrator extension mark-tested --id example --sandbox container --passed
.venv/bin/ai-orchestrator extension approve --id example --sha256 HASH --approver operador --approve
.venv/bin/ai-orchestrator extension activate --id example --profile default --approve
.venv/bin/ai-orchestrator extension rollback --id example --profile default --approve
```

No se ejecutan scripts de instalación en el host. Un worker no monta HOME, el
socket administrativo ni secretos de proveedor; la red queda cerrada hasta que
un proxy de salida aplique la lista de dominios. El shell arbitrario exige una
excepción humana específica. Extensiones de riesgo alto sólo avanzan con un
contenedor desechable e imagen inmutable.

Las claves API se cifran con `systemd-creds` desde stdin, se rotan de forma
atómica y nunca se devuelven. OAuth de Codex, Claude, Kimi, Abacus y FX
permanece en los almacenes oficiales:

```bash
printf '%s' "$VALOR" | .venv/bin/ai-orchestrator credential store \
  --provider minimax --name minimax-api-key --approve
```

`run` solo ejecuta tareas clasificadas como `public` o `internal` y autorizadas
por el perfil. Tareas financieras, secretos, despliegue y el bridge MiniMax de
Algoritmic producen `HANDOFF_REQUIRED`.

Una tarea antigua es de sólo lectura. Para permitir cambios hay que declarar
el alcance y la política explícitamente:

```json
{
  "repo": "/ruta/al/repo",
  "task_type": "tests",
  "sensitivity": "internal",
  "prompt_file": "task.md",
  "input_paths": ["src"],
  "output_contract": "unified_diff",
  "write_mode": "validated_apply",
  "allowed_write_paths": ["tests"],
  "gate_profile": "python-unittest",
  "max_repairs": 2
}
```

Los modos son `read_only`, `validated_patch` y `validated_apply`. En los dos
últimos el modelo sólo modifica un worktree efímero. `validated_apply` aplica
el diff sin stage ni commit únicamente para escritores nativos, cuando pasan
los gates y el repositorio de origen estaba limpio y continúa en el mismo
commit. DeepSeek, MiniMax, Kimi y cualquier escritor externo devuelven siempre
`PATCH_READY` para revisión, incluso con gates verdes.

## Reparto de modelos

- Claude Opus: arquitectura, seguridad y revisión crítica.
- GPT Sol: implementación compleja, determinismo y aceptación; Terra: auditoría,
  estadística y revisión independiente; Luna: Godot/UI y lotes acotados.
- DeepSeek V4 Flash: tests, boilerplate y cambios acotados.
- MiniMax M3: contexto largo, documentación y refactorizaciones masivas.
- Kimi K3: exploración e investigación; cuando haya cuota escribe mediante ACP.
- FX/GLM 5.2: revisión y documentación de sólo lectura durante la promoción;
  segunda opción de investigación después de Kimi.
- Gemini 3.6 Flash gratuito: tests, boilerplate, documentación e investigación
  exclusivamente con sensibilidad `public`; promovido después de 5/5 canarios.

Codex y Claude usan sus clientes oficiales. DeepSeek y MiniMax usan Qwen para
lectura dentro de bubblewrap, sin shell, agentes ni herramientas web. Para
escritura usan sus APIs autenticadas en un subproceso con límite duro y
devuelven un diff que el orquestador valida y aplica en el worktree. Kimi
usa ACP: el prompt viaja por JSON-RPC, el cliente controla cada lectura/escritura
y no anuncia capacidad de terminal. Las rutas web/gratuitas sólo son elegibles
para tareas `public` y después de promocionarlas individualmente de `candidate`
a `available` tras los canarios de contención.

## Almacenamiento

- configuracion: `~/.config/ai-orchestrator` (`0700`, sin secretos);
- estado/telemetria: `~/.local/share/ai-orchestrator` (`0700`/`0600`);
- cache: `~/.cache/ai-orchestrator`;
- temporales: `/tmp/ai-orchestrator-$UID` (`0700`).

Use `storage audit` para inspeccion y `storage maintain` primero sin `--apply`.
La compactacion nunca opera sobre una base si detecta el cliente activo.

Los registros de extensiones y actualizaciones usan SQLite/WAL con control de
generacion. En el primer acceso importan automaticamente los antiguos
`registry.json` y `state.json`; no es necesario borrarlos ni migrarlos a mano.

## Garantias operativas

- La ruta simulada y la ejecutada comparten gates y puntuacion; la telemetria
  de metadatos alimenta calidad, latencia y cache sin conservar prompts.
- La admision usa leases SQLite entre procesos, respeta `max_concurrency` por
  modelo y serializa el mismo repositorio salvo autorizacion explicita.
- El contexto se valida antes de iniciar el proveedor y el supervisor limita
  stdout a 32 MiB y stderr a 1 MiB, terminando todo el grupo de procesos.
- Broker y panel limitan conexiones, trabajadores y tiempos de espera. Las
  unidades systemd tambien fijan memoria, tareas y tiempos de parada.
- MCP negocia las versiones soportadas y ofrece recursos de solo lectura con
  paginacion para inventarios y diagnostico.
- El runtime soportado es Python 3.12, 3.13 y 3.14; CI prueba las tres versiones
  y exige tests, lint y formato limpios.

## Calidad, optimizacion y gateways

Promptfoo esta fijado por lockfile y se ejecuta sin credenciales, sin telemetria,
sin resultados crudos persistentes y dentro de un namespace sin red:

```bash
.venv/bin/ai-orchestrator quality-preflight --project-root .
.venv/bin/ai-orchestrator quality-eval --project-root . \
  --config evals/promptfoo/promptfooconfig.yaml \
  --output /tmp/ai-orchestrator-$UID/quality.json
```

DSPy es optativo y solo admite experimentos de `tests`, `docs` y `review`, con
un baseline verde, dataset separado y presupuesto maximo de 40 llamadas. El
comando actual es un gate de admision: no llama modelos ni optimiza en vivo.

```bash
.venv-dspy/bin/ai-orchestrator prompt-optimize \
  --experiment evals/dspy/tests-v1.toml \
  --baseline /tmp/ai-orchestrator-$UID/quality.json \
  --output /tmp/ai-orchestrator-$UID/dspy-preflight.json
```

LiteLLM y Bifrost son candidatos de laboratorio, no dependencias de produccion.
Los manifiestos exigen imagen/version inmutable, loopback, logging de contenido
desactivado y atestacion real de DeepSeek/Kimi/MiniMax. Un benchmark simulado
nunca puede promover un gateway.

Los proveedores oficiales provisionados se comprueban sin conservar respuestas:

```bash
.venv/bin/ai-orchestrator provider-canary \
  --output /tmp/ai-orchestrator-$UID/official-providers.json
```

La promoción de escritura requiere cinco intentos limpios:

```bash
.venv/bin/ai-orchestrator writer-canary --model MiniMax-M3 --attempts 5
```

DeepSeek usa la ruta explícita `ds/deepseek-v4-flash` de OmniRoute. MiniMax usa
`POST https://api.minimax.io/v1/responses` con `MiniMax-M3` y `store=false`, y
resuelve `minimax-api-key.cred` mediante `systemd-creds`, sin copiar el secreto
a la configuración de Qwen. Kimi descubre alias desde su servidor local y sólo
los valida después de un canario de procedencia. Cuando termina la cuota
reporta `QUOTA_UNAVAILABLE` sin fallback.

Abacus CLI queda fijado a `2.6.9` y usa inicio de sesión humano. Se prefiere la
cuota Pro; SuperComputer exige aprobación y créditos incluidos suficientes, y
nunca compra créditos automáticamente. La política de modelos y sus benchmarks
está en [Rutas Abacus por calidad/precio](docs/abacus-model-routing-2026-08-20.md).

FX queda fijado a `0.0.4` y usa el OAuth ya guardado por su cliente oficial. La
ruta `zai/glm-5.2` es sólo lectura, no hereda claves API, no guarda sesión y se
inhabilita automáticamente después del 1 de septiembre de 2026 para no pasar a
pago al terminar la promoción. La configuración y el canario están en
[FX/GLM 5.2 promocional](docs/fx-glm-5.2-promotion-2026-08.md).

OmniRoute recibe `OMNIROUTE_API_KEY` y `STORAGE_ENCRYPTION_KEY` mediante
`LoadCredentialEncrypted`; ambas fueron retiradas de `~/.omniroute/.env`. La
unidad escucha sólo en loopback y el wrapper desactiva el registro de cuerpos.

Los procesos headless se lanzan en una sesion POSIX propia. El supervisor cierra
stdin, aplica un limite total y termina el grupo completo con TERM/KILL; Qwen
usa `--output-format json`, nunca `stream-json`, y un resultado con
`is_error=true` no puede registrarse como exito.

La configuracion reproducible y el rollback estan documentados en
[la migracion OpenCode/Qwen](docs/opencode-qwen-migration-2026-08-12.md).

Las rutas gratuitas/locales se registran primero como `candidate`. Solo Gemini
gratuito está actualmente `available`; Qwen local, Qwen Web, Kimi gratuito y
KAT Coder permanecen inactivos porque no superaron el canario de escritura.

Consulte [la evaluacion del stack](docs/tool-evaluation-2026-08-05.md) para los
resultados medidos, bloqueos y siguiente paso.

El contenido de sesiones Harness no fijadas se purga a los siete días; una
sesión fijada permanece hasta que el operador la libere. La telemetría conserva
sólo metadatos durante un máximo de 90 días y nunca acepta cuerpos de prompts o
respuestas.
