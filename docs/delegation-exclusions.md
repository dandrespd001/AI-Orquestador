# Delegation exclusions

La implementacion inicial fue realizada directamente por Codex. MiniMax-M3 no
era elegible porque el trabajo cambia almacenamiento global, credenciales,
systemd, routing, trust boundaries y el perfil de un repositorio financiero.
Estas superficies estan expresamente fuera del bridge `$delegate-minimax-api`.

