# Evaluacion del stack multi-modelo — 2026-08-05

## Decision

- **Promptfoo 0.122.0: adoptar.** Es el gate determinista de regresion para
  routing, prompts y contratos. Corre localmente, sin credenciales, sin red,
  sin telemetria/comparticion/cache persistente y conserva solo un recibo de
  conteos, hashes y latencia.
- **DSPy 3.3.0: experimento acotado.** Esta instalado en `.venv-dspy`, aislado
  del runtime. Por ahora solo valida admision; no se autoriza optimizacion real
  hasta disponer de un scorer independiente y un canario de coste cero.
- **LiteLLM 1.95.0: mantener como candidato.** Paso el contrato sintetico, pero
  no sustituye OmniRoute hasta probar los tres proveedores oficiales y mejorar
  materialmente p95 sin regresiones.
- **Bifrost 1.6.8: mantener como candidato ligero.** Su consumo fue mucho menor,
  pero su defensa SSRF bloqueo correctamente el upstream Docker privado. No se
  desactiva esa defensa para fabricar un resultado.
- **Otro gateway: no instalar ahora.** Añadir otra capa antes de terminar esta
  atestacion aumenta superficie, estado y rutas de credenciales sin evidencia
  de una mejora neta.

## Evidencia reproducible

Promptfoo ejecuto 36 casos: 36 exitos, 0 fallos, 0 errores, 1.515 ms. La salida
cruda se destruyo; el recibo privado registra hashes y agregados.

El bake-off uso upstream determinista, payloads sin datos reales, imágenes por
digest, root filesystem de solo lectura, capacidades Linux eliminadas y puertos
solo loopback.

| candidato | contrato | p95 c1 | p95 c4 | p95 c16 | memoria observada | promocion |
|---|---:|---:|---:|---:|---:|---|
| LiteLLM 1.95.0 | 90/90 | 7,2 ms | 25,3 ms | 134,3 ms | 1,07 GiB | No: cobertura no atestiguada |
| Bifrost 1.6.8 | bloqueado por SSRF | — | — | — | 143 MiB | No: contrato/cobertura pendientes |

Estas cifras comparan transporte sintetico, no calidad del modelo ni latencia
del proveedor. Tampoco son comparables contra OmniRoute hasta repetir el mismo
manifiesto con el incumbente levantado en un puerto de laboratorio.

## Almacenamiento y secretos

El auditor encontro:

- `.omniroute/.env` con secreto en texto plano;
- 16 conexiones activas de proveedores no oficiales;
- cinco journals SQLite sin limite;
- 1.290 call logs, con referencias a 1.264 requests y 1.164 responses;
- unos 414,6 MiB recuperables en logs de Codex.

Las mutaciones son `dry-run` por defecto, hacen backup y nunca borran conexiones.
La aplicacion automatica se detuvo por dos motivos correctos:

1. `systemd-creds setup --user` no tiene permiso en este host/sesion;
2. desactivar 16 conexiones compartidas puede afectar otros proyectos y requiere
   aprobacion explicita del propietario.

No se copio ni imprimio ningun valor secreto. No se alteraron bases de datos ni
conexiones durante esta implementacion.

## Siguiente paso

1. Crear un usuario/servicio de laboratorio con credenciales oficiales
   dedicadas y cuotas duras para Kimi, MiniMax y DeepSeek; nunca reutilizar el
   almacen compartido actual.
2. Migrar el secreto de `.omniroute/.env` a credencial del sistema desde una
   sesion de host con privilegios adecuados y comprobar reinicio/rollback.
3. Levantar OmniRoute incumbente, LiteLLM y Bifrost en puertos loopback; para
   Bifrost usar TLS y CA interna o un endpoint oficial de prueba, sin relajar
   SSRF.
4. Ejecutar contratos reales de chat, streaming, tool calls, errores, timeout y
   limites a concurrencia 1/4/16. Atestiguar cada proveedor solo si todos pasan.
5. Promover un reemplazo unicamente si el p95 mediano mejora al menos 20%, el
   peor escenario no empeora mas de 5%, no hay logging de contenido y el costo
   operativo/memoria compensa la migracion.
6. Despues, ejecutar un unico experimento DSPy sobre `tests-v1`: baseline
   congelado, maximo 40 llamadas, conjunto test ciego y Promptfoo como arbitro.

GPT y Claude siguen por sus CLI oficiales (`codex` y `claude`). Los gateways se
limitan a Kimi/MiniMax/DeepSeek con suscripciones oficiales. Ningun fallback de
pago por uso ni proveedor tercero queda autorizado.
