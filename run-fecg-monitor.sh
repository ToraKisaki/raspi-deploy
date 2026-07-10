#!/usr/bin/env bash
# fECG bedside monitor launcher (Raspberry Pi / device side).
# Bundles every command needed to start the monitor. Double-click the
# "fECG Monitor" desktop icon, or run this script directly.
#
# ============ EDIT THESE to match your setup ============
export FECG_SERVER_URL="ws://192.168.79.110:8000/ws/device"   # PC server (ws://<PC-IP>:8000/ws/device)
export FECG_DEVICE_TOKEN="device-secret-001"                # MUST match the server's token
export FECG_PATIENT_ID="P001"                               # which patient this bed records
# =======================================================

REPO="/home/dar/raspi-deploy"
PY="$REPO/.venv/bin/python"
LOG="$HOME/fecg-monitor.log"
export PYTHONPATH="$REPO/device"

# keep a single monitor running: stop any previous instance first
pkill -f "$REPO/device/main.py" 2>/dev/null
sleep 0.3

echo "----- $(date '+%Y-%m-%d %H:%M:%S') launching fECG monitor -> $FECG_SERVER_URL (patient $FECG_PATIENT_ID) -----" >> "$LOG"

# fullscreen monitor (pass --windowed / --sim through if given, e.g. for testing)
exec "$PY" "$REPO/device/main.py" --file "$REPO/examples/test_data" "$@" >> "$LOG" 2>&1
