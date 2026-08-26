# Rutas Abacus por calidad/precio (2026-08-20)

Esta política separa dos decisiones: qué modelo parece mejor en evaluaciones
externas y qué modelo se puede ejecutar con seguridad desde el orquestador. Los
costes publicados por benchmarks son estimaciones del benchmark, no una tarifa
de créditos de Abacus. La política local sigue siendo `pro_included_credits_only`
y nunca compra créditos ni hace fallback automáticamente.

## Evidencia externa

| Fuente | Resultado relevante | Consecuencia local |
| --- | --- | --- |
| [SWE-rebench / Inference Report](https://inference.report/benchmarks/2026-08-11) | Fable 5, Grok 4.5, Opus 5, GLM 5.2 y GPT-5.6 Sol quedan muy próximos y con intervalos solapados en el mismo scaffold. Artificial Analysis sitúa a Qwen3.8 Max por debajo de los frontier, pero a menor coste estimado. | No pagar el modelo frontier por defecto; reservarlo para escalado explícito. |
| [Akita LLM Coding Benchmark](https://github.com/akitaonrails/llm-coding-benchmark/blob/master/docs/success_report.v2.md) | Qwen3.8 Max y Grok 4.6 alcanzan 92/A; MiniMax-M3 91/A; Qwen3.8 27B cae a 51/C. El informe observa variaciones de hasta 69 puntos según el harness. | Max es el delegado de implementación; 27B no es escritor predeterminado; los canarios se ejecutan con el mismo harness real. |
| [VulcanBench Qwen3.8 Max](https://vulcanbench.com/assets/reports/vulcanbench-qwen38-max-report.pdf) | En 23 tareas de PR posteriores al corte, Qwen3.8 Max obtiene 81,2 % con esfuerzo bajo, 71,0 % con medio y 55,1 % con xhigh. | Usar esfuerzo bajo/medio; nunca asumir que más razonamiento mejora este modelo. |
| [InfraBench](https://infraben.ch/leaderboard) | Grok 4.5 aparece fuerte en tareas de infraestructura; Qwen3.8 Max y GLM 5.2 son sensibles al harness y muestran intervalos amplios. | Grok 4.6 se usa como crítico/revisor, no como escritor autónomo con acceso al host. |

GLM 5.3 es reciente y, a esta fecha, no tiene todavía una evaluación externa
independiente y repetida comparable a las anteriores. Por eso está disponible
sólo con `allow_experimental=true`, nunca por selección automática.

## Matriz efectiva

| Tarea | Ruta principal | Motivo |
| --- | --- | --- |
| Arquitectura de CHUNSA | Claude (se conserva fuera de Abacus) | Restricción de gobierno del proyecto. |
| Código acotado, pruebas y documentación de volumen | DeepSeek V4 Flash o MiniMax-M3 por las rutas existentes | Mejor coste marginal y canarios locales ya promovidos. |
| Implementación multiarchivo o refactor complejo delegado a Abacus | Qwen3.8 Max | Mejor equilibrio observado entre código real y capacidad agente; esfuerzo bajo/medio. |
| Revisión, razonamiento, investigación y segunda opinión de seguridad | Grok 4.6 | Señal externa fuerte como razonador/revisor; se ejecuta sin herramientas. |
| Qwen3.8 27B | Sólo pin explícito | Buen dato en algunos benchmarks sintéticos, pero evidencia débil en proyectos reales. |
| ZAI GLM 5.3 | Experimental y pin explícito | Falta evaluación externa madura; requiere `allow_experimental=true`. |
| Fable 5, Opus 5 o GPT-5.6 Sol | Escalado explícito | Calidad absoluta alta, pero no son la ruta calidad/precio predeterminada. |

## Límite de ejecución MCP

La integración pertenece al control común del miniproyecto. Para Codex, Claude,
DeepSeek/Qwen/OmniRoute, MiniMax/Qwen Code, Kimi y las rutas locales aprobadas,
`orchestrator_run_task` consume el mismo `TaskSpec`, perfil, decisión, worktree,
gates y telemetría que el CLI. Un pin de modelo es opcional y no habilita
fallback. `validated_apply` está prohibido desde MCP.

`orchestrator_delegate_abacus` envía el prompt por `stdin`, ejecuta el CLI
oficial fijado a 2.6.9 en un directorio temporal `0700`, deshabilita herramientas,
no monta el repositorio, no hereda claves API, limita entrada/salida/tiempo y
devuelve un recibo sin prompt. Sólo admite sensibilidad `public` o `internal`.
Cada invocación es explícita porque puede consumir cuota incluida.

La primera inferencia real queda pendiente de un canario humano aprobado. Las
pruebas automáticas sustituyen el proceso externo y no consumen créditos.
