"""
Central configuration for the Raspberry Pi 5 fECG monitor device.

IMPORTANT WIRING CONSTRAINT
---------------------------
The 3.5" LCD occupies the lower 26 pins of the 40-pin header (physical pins
1..26 -> the original Pi header region, including hardware SPI0). To avoid any
conflict we drive the ADS1293 over a *software* (bit-banged) SPI bus that lives
entirely on the UPPER part of the header (physical pins 27..40).

All pin numbers below are BCM GPIO numbers, but each one is chosen so its
physical position is >= pin 27 (see the table in README.md).

    Signal   BCM     Physical pin
    ------   -----   ------------
    SCLK     GPIO21  40
    MOSI     GPIO20  38
    MISO     GPIO19  35
    CS       GPIO26  37
    DRDY     GPIO16  36   (data-ready interrupt, active low)
    (3V3, GND taken from pins 17 / 39 which are also in the upper region)

If your LCD frees up other pins, edit only this file.
"""

import os

# ----- Software-SPI pin map (BCM numbering, all on physical pins >= 27) -----
PIN_SCLK = 21
PIN_MOSI = 20
PIN_MISO = 19
PIN_CS = 26
PIN_DRDY = 16

# Bit-bang clock half-period (seconds). ADS1293 tolerates slow SCLK fine.
SPI_HALF_PERIOD = 1e-6  # ~500 kHz effective; plenty for ECG sample rates

# ----- Acquisition -----
# Matches the trained UNETR model in ../infer.py: 250 Hz, single abdominal lead.
SAMPLE_RATE_HZ = 250          # samples/sec (model was trained at 250 Hz)
N_INPUT_CHANNELS = 3          # ADS1293 reads 3 channels; model uses ch0 only
ABDOMINAL_CHANNEL = 0         # the channel carrying the mixed mECG+fECG

# ----- AI model / processing -----
# The real model + preprocessing live in ../infer.py and ../models/unetr_int8.onnx.
# These constants mirror SAMPLE_LEN there; do not change without retraining.
WINDOW_SAMPLES = 992          # model input length (~3.97 s @ 250 Hz)
WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE_HZ
HOP_SAMPLES = int(SAMPLE_RATE_HZ * 0.2)  # re-run inference every 200 ms (~5 Hz)

# ----- Backend -----
# The PC runs the server (see the fecg-server repo). Point the Pi at it without
# editing this file by exporting these env vars, e.g.:
#   export FECG_SERVER_URL="ws://192.168.1.50:8000/ws/device"
#   export FECG_DEVICE_TOKEN="shared-secret"   # must match the server
#   export FECG_PATIENT_ID="P001"
SERVER_WS_URL = os.environ.get("FECG_SERVER_URL", "ws://localhost:8000/ws/device")
PATIENT_ID = os.environ.get("FECG_PATIENT_ID", "P001")  # patient this device records
DEVICE_TOKEN = os.environ.get("FECG_DEVICE_TOKEN", "device-secret-001")  # shared secret
UPLOAD_BATCH = 20             # samples per websocket frame

# ----- UI (3.5" screens are typically 480x320) -----
SCREEN_W = 480
SCREEN_H = 320
PLOT_WINDOW_SECONDS = 4.0     # visible time window in the scrolling plot
FULLSCREEN = True
