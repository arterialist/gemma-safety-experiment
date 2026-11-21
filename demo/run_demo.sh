#!/bin/zsh
# Start the live demo server (about 25 s to load) and open the page.
cd "${0:A:h}/.."
pkill -f "demo/server.py" 2>/dev/null
PYTHONPATH=src .venv/bin/python3 demo/server.py &
sleep 2 && open http://127.0.0.1:8777
wait
