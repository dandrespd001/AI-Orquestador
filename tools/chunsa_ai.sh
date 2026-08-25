#!/usr/bin/env bash
# Entrada acotada al perfil CHUNSA para delegaciones desde Claude Code.
set -euo pipefail

cd "$(dirname "$0")/.."
ORCHESTRATOR_BIN="${CHUNSA_ORCHESTRATOR_BIN:-${AI_ORCHESTRATOR_BIN:-ai-orchestrator}}"

usage() {
  echo "Uso: tools/chunsa_ai.sh route <spec.json>" >&2
  echo "     tools/chunsa_ai.sh run <spec.json> [salida.diff]" >&2
  echo "     tools/chunsa_ai.sh submit <metadata.json>  # prompt por stdin" >&2
  echo "     tools/chunsa_ai.sh artifact <id>" >&2
  echo "     tools/chunsa_ai.sh accept <id> <sha256>" >&2
  echo "     tools/chunsa_ai.sh reject <id> [motivo]" >&2
  echo "     tools/chunsa_ai.sh doctor|doctor-full" >&2
}

[ "$#" -ge 1 ] || { usage; exit 64; }
ACTION="$1"
shift

if ! command -v "$ORCHESTRATOR_BIN" >/dev/null 2>&1 && [ ! -x "$ORCHESTRATOR_BIN" ]; then
  echo "ai-orchestrator no está instalado" >&2
  exit 69
fi

case "$ACTION" in
  doctor)
    [ "$#" -eq 0 ] || { usage; exit 64; }
    exec "$ORCHESTRATOR_BIN" doctor --profile chunsa --check readiness
    ;;
  doctor-full)
    [ "$#" -eq 0 ] || { usage; exit 64; }
    exec "$ORCHESTRATOR_BIN" doctor --profile chunsa --check full
    ;;
  submit)
    [ "$#" -eq 1 ] || { usage; exit 64; }
    [ -f "$1" ] && [ ! -L "$1" ] || {
      echo "Los metadatos deben ser un archivo regular" >&2
      exit 66
    }
    exec "$ORCHESTRATOR_BIN" submit --profile chunsa --metadata-file "$1"
    ;;
  artifact)
    [ "$#" -eq 1 ] || { usage; exit 64; }
    exec "$ORCHESTRATOR_BIN" artifact --id "$1"
    ;;
  accept)
    [ "$#" -eq 2 ] || { usage; exit 64; }
    exec "$ORCHESTRATOR_BIN" accept --id "$1" --sha256 "$2" --approve
    ;;
  reject)
    [ "$#" -eq 1 ] || [ "$#" -eq 2 ] || { usage; exit 64; }
    if [ "$#" -eq 2 ]; then
      exec "$ORCHESTRATOR_BIN" reject --id "$1" --reason "$2" --approve
    fi
    exec "$ORCHESTRATOR_BIN" reject --id "$1" --approve
    ;;
  route|run)
    [ "$#" -ge 1 ] || { usage; exit 64; }
    SPEC_PATH="$1"
    [ -f "$SPEC_PATH" ] && [ ! -L "$SPEC_PATH" ] || {
      echo "El spec debe ser un archivo regular y no un enlace simbólico" >&2
      exit 66
    }
    PIN_ARGS=()
    if [ -n "${CHUNSA_AI_MODEL:-}" ]; then
      PIN_ARGS=(--model "$CHUNSA_AI_MODEL")
    fi
    if [ "$ACTION" = "run" ] && [ "$#" -eq 2 ]; then
      [ ! -L "$2" ] || { echo "La salida no puede ser un enlace simbólico" >&2; exit 66; }
      exec "$ORCHESTRATOR_BIN" run --spec "$SPEC_PATH" --profile chunsa \
        "${PIN_ARGS[@]}" --output "$2"
    fi
    [ "$#" -eq 1 ] || { usage; exit 64; }
    exec "$ORCHESTRATOR_BIN" "$ACTION" --spec "$SPEC_PATH" --profile chunsa \
      "${PIN_ARGS[@]}"
    ;;
  *)
    usage
    exit 64
    ;;
esac
