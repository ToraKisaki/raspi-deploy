"""
Streams (timestamp, raw, fecg) batches to the backend over a WebSocket.

Runs in a background thread with an internal queue so acquisition never blocks
on the network. Reconnects automatically. If `websocket-client` isn't installed
or the server is down, samples are dropped (after buffering a little) and the
local monitor keeps working.

    pip install websocket-client
"""

import json
import time
import threading
import queue

from config import (
    SERVER_WS_URL, PATIENT_ID, DEVICE_TOKEN, UPLOAD_BATCH, SAMPLE_RATE_HZ,
)

try:
    import websocket  # websocket-client
    _HAVE_WS = True
except Exception:
    _HAVE_WS = False


class Uploader:
    def __init__(self, url=SERVER_WS_URL, patient_id=PATIENT_ID):
        self.url = url
        self.patient_id = patient_id
        self.q = queue.Queue(maxsize=SAMPLE_RATE_HZ * 10)  # ~10 s buffer
        self._batch = []
        self._stop = threading.Event()
        self.connected = False
        self.enabled = _HAVE_WS
        if self.enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def push(self, ts, raw, fecg):
        if not self.enabled:
            return
        self._batch.append([round(ts, 4), round(raw, 5), round(fecg, 5)])
        if len(self._batch) >= UPLOAD_BATCH:
            try:
                self.q.put_nowait(self._batch)
            except queue.Full:
                pass  # drop oldest by simply discarding this batch
            self._batch = []

    def _run(self):
        while not self._stop.is_set():
            try:
                ws = websocket.create_connection(self.url, timeout=5)
                ws.send(json.dumps({
                    "type": "hello",
                    "patient_id": self.patient_id,
                    "token": DEVICE_TOKEN,
                    "sample_rate": SAMPLE_RATE_HZ,
                }))
                self.connected = True
                print(f"[uploader] connected to {self.url}")
                while not self._stop.is_set():
                    batch = self.q.get()
                    ws.send(json.dumps({
                        "type": "samples",
                        "patient_id": self.patient_id,
                        "data": batch,
                    }))
            except Exception as e:
                self.connected = False
                print(f"[uploader] connection issue: {e}; retrying in 3s")
                time.sleep(3)

    def stop(self):
        self._stop.set()
