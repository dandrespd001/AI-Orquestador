# AI Orchestrator

Control plane local para seleccionar y auditar modelos de suscripciones
oficiales. GPT se ejecuta con Codex, Claude con Claude Code, DeepSeek con Qwen
Code sobre OmniRoute, MiniMax con Qwen contra su endpoint oficial y Kimi con su
CLI nativo. Cada repositorio puede endurecer estas reglas con un perfil propio.

El sistema nunca guarda prompts ni respuestas en telemetria. Registra hashes,
modelo solicitado y observado, latencia, estado de cuota y gates. No realiza
fallback silencioso ni convierte una suscripcion en pago por uso.

## Inicio rapido

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
AI_ORCHESTRATOR_CONFIG="$PWD/config" .venv/bin/ai-orchestrator doctor --profile default
AI_ORCHESTRATOR_CONFIG="$PWD/config" .venv/bin/ai-orchestrator route --spec examples/task.json
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
`https://api.minimax.io/anthropic` y resuelve `minimax-api-key.cred` mediante
`systemd-creds`, sin copiar el secreto a la configuración de Qwen. Kimi reporta
`QUOTA_UNAVAILABLE` sin fallback cuando termina la cuota.

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
