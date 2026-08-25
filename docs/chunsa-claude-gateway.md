# Gateway Claude de CHUNSA

Claude Code coordina CHUNSA con Sonnet y esfuerzo `high`, pero toda ejecución,
escritura o delegación cruza AI-Orchestrator. La configuración versionada está en
`integrations/chunsa/`: `.mcp.json` registra el servidor ligado al perfil y
`.claude/settings.json` instala los hooks compartidos.

`SessionStart` exige Sonnet/high, el repositorio exacto, MCP y readiness. Sólo
entonces crea una atestación privada de sesión. `PreToolUse` falla cerrado sin
ella: admite lecturas dentro del repositorio, grafo, gates exactos, Git de sólo
lectura y entradas de AI-Orchestrator; deniega edición directa, agentes, CLIs de
modelos, Git mutante y otros MCP.

## Workflows

`orchestrator_submit_task` recibe `prompt` y `task` inline. `repo`, `profile`,
`prompt` y `prompt_file` están prohibidos dentro de `task`; el servidor deriva el
repositorio de `chunsa`. Cada ruta encadena proveedores independientes y nunca
hace fallback:

| Trabajo | Flujo |
| --- | --- |
| Arquitectura | Opus → Grok 4.6 |
| Implementación | Qwen3.8 Max (plan) → Sonnet → Grok 4.6 |
| Refactor masivo | Qwen3.8 Max (plan) → Sol → Grok 4.6 |
| Tests | Qwen3.8 Max (plan) → DeepSeek Flash → Terra |
| Documentación | Qwen3.8 Max (plan) → Sonnet → Grok 4.6 |
| C++ acotado | DeepSeek Flash, o Pro si es corto → Sol |
| Determinismo | Sol → Opus |
| Godot/UI | Qwen opcional → Kimi → Sonnet |
| Seguridad / investigación | Grok 4.6 → Terra / Opus |

Grok es siempre de sólo lectura. Qwen sólo planifica hasta acumular cinco
canarios consecutivos; su modo de parche sigue siendo opt-in, se valida en un
worktree desechable y requiere revisión independiente. Qwen y Grok tienen
concurrencia uno y límites móviles de 12 y 6 intentos por 24 horas,
respectivamente. Un fallo consume intento y no activa reintentos, compras ni otro
modelo.

Ejemplo humano por CLI; el prompt no pasa por argumentos ni por un archivo:

```bash
printf '%s' 'Implementa la corrección descrita.' | tools/chunsa_ai.sh submit task.json
tools/chunsa_ai.sh artifact ID
tools/chunsa_ai.sh accept ID SHA256
# o bien
tools/chunsa_ai.sh reject ID 'motivo breve'
```

Un `task.json` mínimo de escritura es:

```json
{
  "task_type": "implementation",
  "write_mode": "validated_patch",
  "allowed_write_paths": ["addons/chunsa_sim/core"],
  "gate_profile": "profile"
}
```

Los parches permanecen privados durante siete días. `accept` vuelve a comprobar
SHA-256, HEAD, árbol limpio, alcance y gates antes de aplicar el diff; `reject`
es terminal. Ninguno está expuesto por MCP.

## Operación

```bash
tools/chunsa_ai.sh doctor       # disponibilidad actual
tools/chunsa_ai.sh doctor-full  # disponibilidad + cumplimiento histórico
ai-orchestrator workflow-canary --model abacus-qwen3.8-max --attempts 5
ai-orchestrator workflow-canary --model abacus-grok-4.6 --attempts 5
```

`doctor` comprueba configuración, binarios, autenticación, MCP, broker,
OmniRoute y sincronización. Los hallazgos históricos de almacenamiento aparecen
en `compliance`, pero sólo hacen fallar `doctor-full`. Si el socket del broker no
responde, MCP conserva una recuperación directa sin relajar el perfil.
