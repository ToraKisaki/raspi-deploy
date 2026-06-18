# Real-time fECG Monitor + Server

This adds a **live bedside monitor** and a **clinician web backend** on top of
your existing offline pipeline (`infer.py` + `models/unetr_int8.onnx`, unchanged).

```
ADS1293 ──soft-SPI──▶  Pi 5 monitor (PyQt5, 480×320)         ┌─ doctor web dashboard
   │                    ├─ raw abdominal trace (scrolling)   │
   │  single abdominal  ├─ infer.py  → fECG trace (scrolling)├─ live per-patient ECG
   └─ lead, 250 Hz      └─ WebSocket uploader ───────────────▶ FastAPI + SQLite ┘
```

The monitor reuses **your** model maths: `device/model_loader.py` imports
`infer.py` from this folder, so band-pass [3–90 Hz], `/global_var`, the UNETR
INT8 ONNX session, and `fECG = output channel 1` are exactly as you defined them.
Nothing in `infer.py` or `models/` was modified.

New folders:
- **`device/`** — runs on the Raspberry Pi 5 (bedside unit)
- **`server/`** — runs anywhere on the network (backend + web UI)

---

## 1. Wiring — ADS1293 on the upper GPIO pins (LCD keeps pins 1–26)

Your 3.5" LCD uses the lower 26 header pins (incl. hardware SPI0), so the
ADS1293 is driven over a **software SPI bus on physical pins 27–40 only**.

| ADS1293 | Signal | BCM GPIO | Physical pin |
|---------|--------|----------|--------------|
| SCLK    | clock  | GPIO21   | **40** |
| SDI     | MOSI   | GPIO20   | **38** |
| SDO     | MISO   | GPIO19   | **35** |
| CSB     | chip select | GPIO26 | **37** |
| DRDYB   | data ready (active-low) | GPIO16 | **36** |
| VDD     | 3V3    | —        | 17 |
| GND     | ground | —        | 39 |

All signal pins are ≥ pin 27. To change a pin, edit **only** `device/config.py`.
The model uses a single abdominal lead (channel 0); the driver still reads all
three ADS1293 channels if you wire them.

> **Pi 5:** use `lgpio` (`sudo apt install python3-lgpio`). `RPi.GPIO` does not
> work on the Pi 5.

---

## 2. Run the backend (any machine on the network)

```bash
cd server
pip install -r requirements.txt
python seed.py                       # demo doctor + patient P001
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://SERVER_IP:8000/login** → **doctor / password**. The patient list
shows live/idle status; click a patient for the live scrolling raw + fECG traces,
sample counts, and session history. Add clinicians with `db.create_doctor(...)`.

Before any real use: change `DEVICE_TOKEN` (in `device/config.py` and the
server's `FECG_DEVICE_TOKEN` env var) and put the server behind HTTPS/`wss`.

---

## 3. Run the Pi monitor

```bash
cd device
sudo apt install python3-lgpio
pip install -r requirements.txt      # PyQt5, pyqtgraph, onnxruntime, scipy, ...
# point the device at your server: edit SERVER_WS_URL and PATIENT_ID in config.py
python3 main.py                      # fullscreen on the LCD; Esc to quit
```

- Top = raw abdominal signal, bottom = AI-extracted fECG, both scrolling.
- The model runs on a 992-sample (≈4 s) window, re-inferring every 200 ms and
  replaying the freshest samples so the trace stays smooth and ~200 ms behind.
- No ADS1293 / `lgpio`? The app falls back to a realistic **simulated** ECG:
  `python3 main.py --sim --windowed`. If `onnxruntime`/`scipy` are missing, a
  placeholder filter stands in for the model so the UI still runs.

### Test the whole chain without the Pi

From any machine, stream simulated data to the server exactly like the device:

```bash
cd device
pip install numpy websocket-client onnxruntime scipy
python3 stream_sim.py --patient P001 --url ws://SERVER_IP:8000/ws/device
```

Then watch it live in the dashboard.

---

## 4. File map

| File | Role |
|------|------|
| `device/config.py` | pins, 250 Hz / 992-window, server URL, patient/token |
| `device/ads1293.py` | bit-banged SPI driver on the upper GPIO pins (lgpio) |
| `device/acquisition.py` | hardware source + automatic simulator fallback |
| `device/model_loader.py` | imports `infer.py`; placeholder fallback |
| `device/fecg_processor.py` | rolling-window real-time wrapper around the model |
| `device/uploader.py` | non-blocking WebSocket uploader, auto-reconnect |
| `device/main.py` | PyQt5 / pyqtgraph 480×320 dual scrolling monitor |
| `device/stream_sim.py` | headless simulator → server (no GUI / hardware) |
| `server/app.py` | FastAPI: device ingest WS, live fan-out WS, auth, REST |
| `server/database.py` | SQLite (doctors, patients, sessions, samples) |
| `server/static/*.html` | login, patient list, live per-patient ECG view |

## 5. What was tested

End-to-end in a sandbox: auth gating (401/200), device handshake + bad-token
rejection, live WebSocket broadcast to a dashboard subscriber, sample
persistence math, the rolling real-time processor (smooth output via `infer.py`),
and the no-model fallback path.

> Research/educational software — **not a certified medical device.**
