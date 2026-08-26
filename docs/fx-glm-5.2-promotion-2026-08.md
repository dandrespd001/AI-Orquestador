# FX/GLM 5.2 promocional (2026-08)

FX se integra como proveedor temporal de primera clase, no como sustituto del
control plane. El cliente está fijado a `0.0.4`, el modelo exacto es
`zai/glm-5.2` y la ruta local se llama `fx-glm-5.2-promo`.

La promoción fue anunciada como acceso gratuito durante dos semanas a partir
del 19 de agosto de 2026. Como esa duración procede del anuncio de lanzamiento
y podría cambiar, AI-Orchestrator usa un corte conservador el **1 de septiembre
de 2026**. Al día siguiente la ruta devuelve `promotion_expired`; no cambia a
otro modelo ni habilita pago. Referencias: [repositorio oficial de
FX](https://github.com/vercel-labs/fx), [modelos y `FX_MODEL`](https://fx.sh/docs/configure-fx/models),
[ficha oficial de GLM 5.2 en AI Gateway](https://vercel.com/ai-gateway/models/glm-5.2/faq)
y [anuncio de la promoción](https://www.blackbox.ai/blog/glm-5-2-free-vercel-fx).

## Política efectiva

- Primera ruta para `review` y `docs` mientras la promoción esté vigente.
- Segunda ruta para `research`, después de Kimi, para no romper la ruta OAuth ya
  promovida.
- No participa en `implementation`, `tests` ni `long_context` como escritor.
- Capacidad `read` únicamente; cualquier `validated_patch` o `validated_apply`
  descarta FX antes de ejecutarlo.
- `FX_PERMISSION_MODE=ask`: una herramienta que necesite aprobación detiene la
  ejecución no interactiva. No se usa `auto`, porque ese modo puede invocar un
  revisor adicional.
- `--no-save`, prompt por `stdin`, `--no-additional-dirs`, máximo 12 pasos,
  actualizaciones y Herdr desactivados.
- No existe fallback silencioso ni gasto incremental permitido.

Las garantías de `ask`, `auto` y `yolo` están documentadas en [permisos de
FX](https://fx.sh/docs/configure-fx/permissions). El contrato de `ask`, JSON,
`stdin` y `--no-save` está en [fx ask](https://fx.sh/docs/using-fx/fx-ask).

## Autenticación y claves

No se configura una API key en AI-Orchestrator. Se reutiliza el OAuth creado por
`fx login` en el almacén oficial `~/.fx/auth.json`; el orquestador nunca abre ni
serializa ese archivo. `AI_GATEWAY_API_KEY`, `OPENAI_API_KEY` y otras claves se
eliminan del entorno del worker. La ruta sólo conserva `HOME` para que el propio
cliente FX pueda resolver su sesión oficial.

No se registra el MCP de AI-Orchestrator dentro de FX: FX es un worker delegado,
no otro plano administrativo. Esto evita recursión y exposición innecesaria de
herramientas. La MCP común continúa registrada en Claude y Codex, desde donde
`orchestrator_run_task` puede seleccionar FX con la misma política.

## Operación

```bash
# Inspección local, sin inferencia
fx status

# Canario único, exacto y sin conservar respuesta
.venv/bin/ai-orchestrator provider-canary --provider fx --timeout 45

# Simular la ruta desde el catálogo común
.venv/bin/ai-orchestrator control simulate --profile default --spec task.json
```

El canario sólo se considera válido si FX devuelve JSON, el texto exacto, el
modelo `zai/glm-5.2`, ningún identificador de sesión y código de salida cero. El
recibo conserva hashes y tamaños, nunca el texto de respuesta.
