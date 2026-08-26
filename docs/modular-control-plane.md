# Control modular con ecosistema comunitario

## Límites de arquitectura

DeepSeek Harness es la interfaz, no la raíz de confianza. AI Orchestrator sigue
siendo propietario de política, credenciales, presupuestos, procedencia,
telemetría y activaciones. El perfil `harness-admin` no carga código comunitario
y sólo alcanza el broker por un socket Unix del mismo UID y modo `0600`.

El broker administrativo no ofrece métodos de escritura. Sus métodos son inventario de
proveedores, modelos, perfiles, autenticación y capacidades; política y
simulación de rutas; diagnósticos; consumo; extensiones; actualizaciones y
sesiones. Las respuestas pasan por una segunda eliminación defensiva de campos
con forma de secreto o cuerpo.

Harness está fijado a `@deepseek-ai/dsh@0.1.0-rc.7`. El comando y la unidad de
systemd incluyen `--host 127.0.0.1 --port 3080 --no-open`. Como Harness continúa
en *developer preview*, ninguna actualización es automática: se valida en un
slot paralelo y se conserva el slot anterior para rollback.

## Contratos públicos

- `ProviderSpec`: endpoint, protocolo, modelos, origen de autenticación, cuota,
  confianza, capacidades y canario. Rechaza secretos incluso anidados.
- `AuthProfileStatus`: disponibilidad y origen; no puede representar el valor
  de una credencial.
- `RoutePlan`: selección, alternativas informativas, puntuaciones, presupuesto
  y permisos. `fallback_allowed` siempre es falso.
- `UsageRecord`: tokens, caché, créditos, coste estimado, latencia y resultado;
  no contiene campos de prompt, respuesta, cuerpo o contenido.
- `ExtensionManifest`: clase, fuente, versión/commit, SHA-256, licencia,
  permisos, compatibilidad, perfiles, herramientas y dominios.

Las restricciones de sensibilidad, confianza, capacidad de escritura, cuota y
presupuesto se aplican antes de puntuar calidad observada, cuota incluida,
coste marginal, ventana de contexto, caché y latencia. Una alternativa requiere
una nueva decisión; jamás se convierte en fallback silencioso.

El perfil CHUNSA fija `claude-opus` como arquitecto. Los transportes delegados
usan `native_patch_writer`, `patch_api_writer` o `kimi_acp_writer`: escriben sólo
en un worktree desechable y entregan `PATCH_READY` para revisión.

## Credenciales y proveedores

Las credenciales API entran por stdin a `systemd-creds`, nunca por argv. El
directorio se mantiene `0700`, el blob `0600`, y `os.replace` hace atómica la
rotación. La API y el panel devuelven únicamente un recibo de generación.

Codex, Claude, Kimi, Abacus y FX conservan OAuth en sus clientes oficiales. Kimi
descubre modelos mediante `/api/v1/models` en loopback, rechaza respuestas con
campos de credencial y exige un canario de alias. MiniMax usa Responses API en
`/v1/responses`, `MiniMax-M3`, Bearer y `store=false`; el puente Anthropic queda
retirado. Abacus se fija inicialmente a `2.6.9`, requiere login humano y prohíbe
compras automáticas. SuperComputer sólo se planifica contra créditos incluidos.

Los canarios acotados están registrados para Codex/OpenAI, Claude/Anthropic,
DeepSeek, MiniMax, Kimi, OmniRoute, Abacus y FX/GLM 5.2. FX se ejecuta sólo en
modo lectura, sin sesión, sin claves API heredadas y con caducidad de promoción.
El CLI Abacus 2.6.9 observado en la
cuenta autenticada admite `--print`, `--model` y prompt por stdin. Su adaptador
MCP no lee el token, no hereda claves API, deshabilita herramientas, ejecuta en
un directorio temporal y sólo admite datos `public` o `internal`. El primer
canario de inferencia real sigue requiriendo aprobación humana porque consume
cuota incluida.

La MCP registrada en Claude y Codex comparte una sola entrada de ejecución para
los adaptadores principales: `orchestrator_run_task` recibe la ruta de un
`TaskSpec`, aplica el perfil y un pin opcional, registra sólo metadatos y prohíbe
`validated_apply`. Abacus conserva `orchestrator_delegate_abacus` por su contrato
OAuth/CLI distinto. Así, añadir un proveedor al catálogo y su adaptador lo hace
accesible desde ambos clientes sin duplicar política ni credenciales.

## Extensiones

Fuentes admitidas: GitHub, Git privado, local, npm, PyPI, MCP `stdio` y MCP
HTTPS. Niveles de confianza: oficial, mantenida por proveedor, comunitaria
verificada, comunitaria experimental y privada/local.

El registro aplica esta máquina de estados:

1. `discover`: valida manifiesto y fuente sin ejecutar código.
2. `quarantine`: copia sólo archivos regulares, sin enlaces ni especiales, y
   fija el hash del árbol.
3. `inspect`: revisa dependencias, scripts npm, procesos, red, credenciales,
   triggers de skills, esquemas MCP, herramientas y dominios.
4. `mark-tested`: acepta sólo un recibo de Bubblewrap o contenedor; riesgo alto
   exige contenedor.
5. `approve`: requiere identidad humana y coincidencia exacta del SHA-256.
6. `activate`: vuelve a calcular el hash, verifica perfil y permisos, y registra
   el estado anterior.
7. `rollback`: verifica también el payload anterior antes de reactivarlo.

Una referencia aprobada es inmutable: no puede reutilizarse con otro hash o
manifiesto. El catálogo genera configuraciones por perfil para Harness, Codex,
Claude, Kimi, Qwen, OpenCode y Abacus. No crea MCP globales ni incrusta
credenciales.

Los workers usan Bubblewrap por defecto y contenedor desechable para riesgo
alto o acceso directo al host. Nunca montan HOME, el socket administrativo o
secretos de proveedor. La red cruda está desactivada; una petición de red queda
pendiente de un proxy que limite los dominios declarados. El shell requiere una
excepción humana separada.

## Retención, eficiencia y operación

- Sesiones Harness: siete días; las fijadas se excluyen hasta liberación.
- Telemetría: sólo metadatos, máximo 90 días.
- OmniRoute: vista previa antes de purga y registro de cuerpos desactivado.
- Paquetes: `games`, `systems`, `research` y `academic`; sólo se carga el
  relacionado con la tarea.
- Configuración: prefijos estables, límites de salida, caché y contexto del
  grafo reducen esquemas y tokens.

Referencias primarias: [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness),
[publicación de plugins](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/publish.md),
[sesiones](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session.md),
[MiniMax Responses API](https://platform.minimax.io/docs/api-reference/responses-create),
[Kimi providers](https://www.kimi.com/code/docs/en/kimi-code-cli/configuration/providers.html)
y [Abacus CLI](https://api.abacus.ai/help/abacusai-desktop/cli-installation).
