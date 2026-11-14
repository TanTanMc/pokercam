#!/bin/bash
# Double-click this file to start PokerCam.  Extra options go after it in a terminal, e.g.
#   ./start.command --demo        ./start.command --camera 1
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First run: installing (about a minute)..."
  python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python app.py "$@"
