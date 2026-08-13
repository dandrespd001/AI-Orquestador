# Migracion headless OpenCode → Qwen Code

Fecha de decision: 2026-08-12.

## Estado objetivo

- Codex y Claude conservan sus clientes nativos.
- DeepSeek se ejecuta con Qwen Code y el modelo explicito
  `ds/deepseek-v4-flash` de OmniRoute.
- MiniMax se ejecuta con Qwen Code contra
  `https://api.minimax.io/anthropic`; el cliente añade `/v1/messages`.
- Kimi conserva su cliente oficial mediante ACP y no participa en fallback
  automatico.
- OpenCode queda disponible solo detras de `tools/oc_guard.sh` en CHUNSA001.

Qwen recibe el prompt por stdin y EOF para tareas de lectura, siempre en `plan`
dentro de bubblewrap. En escritura, DeepSeek y MiniMax usan una llamada API
tool-free dentro de un subproceso con límite duro; el modelo devuelve un diff
unificado y el orquestador lo valida y aplica en el worktree temporal. Qwen sólo
ve el worktree en lectura y un HOME efímero; se excluyen `run_shell_command`,
`agent`, web y skills por su nombre canónico, y se limitan reloj, turnos y herramientas.
`stream-json` permanece prohibido hasta que su semántica de error sea fiable.

Qwen 0.21.10 tambien puede envolver un fallo de conexion como un resultado
`success` con exit 0. El adaptador rechaza cualquier resultado vacio o cuyo
texto final empiece por `[API Error: ...]`, ademas de comprobar `subtype`,
`is_error` y la procedencia exacta del modelo.

## Credenciales

La ejecución no depende de `~/.qwen/settings.json`. No debe contener
`OMNIROUTE_API_KEY` ni `MINIMAX_API_KEY`, ni existir `~/.qwen/.env` creado por
automatización. Las credenciales se inyectan al proceso desde un entorno
acotado.

MiniMax se resuelve en este orden: `CREDENTIALS_DIRECTORY`, variable explícita
y `~/.config/ai-orchestrator/credentials/minimax-api-key.cred` descifrada en
memoria con `systemd-creds`. Si falta, el resultado es
`CREDENTIAL_UNAVAILABLE`. No se extraen secretos de OpenCode.

Kimi recibe el prompt por JSON-RPC, no por argv. El cliente ACP anuncia lectura
y escritura de texto, pero no terminal; cada ruta se comprueba contra la
allowlist. Una política local efímera deshabilita Bash, subagentes, cron y web y
se retira antes de construir el diff.

## OmniRoute

El servicio debe ejecutarse sin tray y escuchar solamente en
`127.0.0.1:20128`. La variable correcta en 3.8.49 es
`OMNIROUTE_SERVER_HOST`; `OMNIROUTE_HOST` no controla el bind. Los trabajos
auditados usan IDs concretos, no combos, y no hacen fallback entre familias.

El detector local de procesos de `omniroute health` no reconoce el daemon
envuelto por systemd. La comprobacion operativa debe indicar el endpoint:

```bash
omniroute --base-url http://127.0.0.1:20128 health
```

El corte quedó completado el 2026-08-12: la unidad usa
`LoadCredentialEncrypted` para la clave del gateway y la clave de cifrado de
storage. Ambas variables se retiraron del `.env` sólo después de comprobar un
arranque y un canario DeepSeek; el respaldo privado se conserva bajo
`~/.local/share/ai-orchestrator/backups/`.

## Verificacion y rollback

1. Ejecutar `.venv/bin/python -m unittest discover -s tests -v` con
   `PYTHONPATH=src`.
2. Comprobar `qwen --version` y exigir cinco canarios de escritura consecutivos
   antes de promover un modelo.
3. Verificar que el puerto 20128 solo aparece en loopback y que no hay cuerpos
   de prompt/respuesta en logs.
4. Instalar y comprobar la configuración con `ai-orchestrator config sync
   --apply`; la ruta de backup aparece en el recibo.
5. Para rollback, restaurar la copia privada de configuración y cambiar los
   modelos externos a `read_only`/`candidate`. OpenCode no se elimina.
