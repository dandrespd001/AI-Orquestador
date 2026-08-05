# AI Orchestrator

Control plane local para seleccionar y auditar modelos de suscripciones
oficiales. GPT se ejecuta con Codex, Claude con Claude Code y DeepSeek/Kimi/
MiniMax mediante OmniRoute. Cada repositorio puede endurecer estas reglas con
un perfil propio.

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

`run` solo ejecuta tareas clasificadas como `public` o `internal` y autorizadas
por el perfil. Tareas financieras, secretos, despliegue y el bridge MiniMax de
Algoritmic producen `HANDOFF_REQUIRED`.

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

DeepSeek usa el CLI autenticado de OmniRoute con `env-key` en modo no-log;
MiniMax usa el proveedor oficial de OpenCode en agente `auditor`; Kimi usa su
CLI oficial y reporta `QUOTA_UNAVAILABLE` sin fallback cuando termina la cuota.

Consulte [la evaluacion del stack](docs/tool-evaluation-2026-08-05.md) para los
resultados medidos, bloqueos y siguiente paso.
