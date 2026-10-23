#!/usr/bin/env sh
cd "$(dirname "$0")" || exit 1
if [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python main.py --studio "$@"
fi
exec python3 main.py --studio "$@"
