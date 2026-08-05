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

