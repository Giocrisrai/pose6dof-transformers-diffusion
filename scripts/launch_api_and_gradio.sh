#!/usr/bin/env bash
# Lanza API REST (uvicorn) en 8000 y Gradio demo en 7860 en paralelo.
# Usado por docker/api.Dockerfile como CMD.
set -euo pipefail

cd /app

# API en background
uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Gradio en foreground (mantiene el contenedor vivo)
python scripts/demo_charla.py &
GRADIO_PID=$!

# Reenvía señales para cierre limpio
trap "kill -TERM $API_PID $GRADIO_PID 2>/dev/null; wait" SIGTERM SIGINT
wait -n $API_PID $GRADIO_PID
exit $?
