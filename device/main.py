#!/usr/bin/env python3
"""
fECG monitor app for Raspberry Pi 5 + 3.5" LCD (480x320).

Two scrolling traces, ECG-monitor style:
    top    = raw abdominal signal from the ADS1293
    bottom = fetal ECG extracted by your AI model (or placeholder filter)

Acquisition runs in a worker thread; the GUI samples the shared ring buffers on
a timer and redraws. Also streams data to the backend via Uploader.

Run on the Pi:
    python3 main.py
Force the simulator anywhere:
    python3 main.py --sim
"""

import sys
import time
import threading
import argparse
import numpy as np

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from config import (
    SCREEN_W, SCREEN_H, SAMPLE_RATE_HZ, PLOT_WINDOW_SECONDS, FULLSCREEN,
    ABDOMINAL_CHANNEL, PATIENT_ID,
)
from acquisition import make_source
from fecg_processor import FecgProcessor
from uploader import Uploader
from hr import (
    estimate_bpm, ema,
    FHR_MIN, FHR_MAX, MHR_MIN, MHR_MAX, FHR_REFRACTORY_S, MHR_REFRACTORY_S,
)

BUF = int(SAMPLE_RATE_HZ * PLOT_WINDOW_SECONDS)

RAW_COLOR = "#00E5FF"
FECG_COLOR = "#FF4081"


class Acquirer(QtCore.QObject):
    """Background acquisition + processing -> shared ring buffers."""
    status = QtCore.pyqtSignal(str)

    def __init__(self, force_sim=False, file_path=None, loop=True):
        super().__init__()
        self.raw = np.zeros(BUF, dtype=np.float32)
        self.fecg = np.zeros(BUF, dtype=np.float32)
        self._lock = threading.Lock()
        self.force_sim = force_sim
        self.file_path = file_path
        self.loop = loop
        self.proc = FecgProcessor()
        self.uploader = Uploader()
        self._stop = threading.Event()
        self.t0 = time.time()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        src = make_source(force_sim=self.force_sim,
                          file_path=self.file_path, loop=self.loop)
        kind = getattr(src, "label", None) or (
            "SIMULATOR" if getattr(src, "is_simulated", False) else "ADS1293")
        self.status.emit(str(kind))
        for ch in src.samples():
            if self._stop.is_set():
                break
            raw = float(ch[ABDOMINAL_CHANNEL])
            fe = self.proc.push(raw)
            with self._lock:
                self.raw[:-1] = self.raw[1:]
                self.raw[-1] = raw
                self.fecg[:-1] = self.fecg[1:]
                self.fecg[-1] = fe
            self.uploader.push(time.time() - self.t0, raw, fe)

    def snapshot(self):
        with self._lock:
            return self.raw.copy(), self.fecg.copy()

    def stop(self):
        self._stop.set()
        self.uploader.stop()


class HRTile(QtWidgets.QWidget):
    """Compact big-number heart-rate readout (label + bpm + units)."""

    def __init__(self, name, color):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 0, 8, 0)
        lay.setSpacing(0)
        self.name = QtWidgets.QLabel(name)
        self.name.setStyleSheet(f"color:{color}; font:bold 10px monospace;")
        self.value = QtWidgets.QLabel("—")
        self.value.setStyleSheet(f"color:{color}; font:bold 34px monospace;")
        self.units = QtWidgets.QLabel("bpm")
        self.units.setStyleSheet("color:#888; font:9px monospace;")
        lay.addWidget(self.name)
        lay.addWidget(self.value)
        lay.addWidget(self.units)

    def set(self, bpm):
        self.value.setText(str(int(round(bpm))) if bpm is not None else "—")


class Monitor(QtWidgets.QWidget):
    # recompute heart rate ~3x/sec (the plot itself redraws at ~30 fps)
    HR_EVERY = 10

    def __init__(self, acq):
        super().__init__()
        self.acq = acq
        self.setWindowTitle("fECG Monitor")
        self.resize(SCREEN_W, SCREEN_H)
        self.setStyleSheet("background:#000;")

        pg.setConfigOptions(antialias=False)  # faster on the Pi
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # header: patient + signal source (no model details)
        self.header = QtWidgets.QLabel(f"Patient {PATIENT_ID}   •   starting…")
        self.header.setStyleSheet("color:#7CFC00; font:bold 11px monospace;")
        layout.addWidget(self.header)

        # heart-rate readouts: FHR (from fECG) + MHR (from raw)
        hr_row = QtWidgets.QHBoxLayout()
        hr_row.setContentsMargins(0, 0, 0, 0)
        hr_row.setSpacing(2)
        self.fhr_tile = HRTile("FHR", FECG_COLOR)
        self.mhr_tile = HRTile("MHR", RAW_COLOR)
        hr_row.addWidget(self.fhr_tile)
        hr_row.addWidget(self.mhr_tile)
        hr_row.addStretch(1)
        layout.addLayout(hr_row)

        # one shared time axis so both traces scroll together in real time
        x = np.linspace(-PLOT_WINDOW_SECONDS, 0, BUF)

        self.p_raw = self._plot("RAW", RAW_COLOR)
        self.c_raw = self.p_raw.plot(x, self.acq.raw, pen=pg.mkPen(RAW_COLOR, width=1))
        layout.addWidget(self.p_raw)

        self.p_fe = self._plot("fECG", FECG_COLOR)
        self.c_fe = self.p_fe.plot(x, self.acq.fecg, pen=pg.mkPen(FECG_COLOR, width=1))
        layout.addWidget(self.p_fe)

        # lock the fECG x-axis to the raw one: identical window, no drift
        self.p_fe.setXLink(self.p_raw)
        self.p_raw.setXRange(-PLOT_WINDOW_SECONDS, 0, padding=0)

        self.x = x
        self.fhr = None
        self.mhr = None
        self._frame = 0
        self.acq.status.connect(self._set_status)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(33)  # ~30 fps

    def _plot(self, title, color):
        p = pg.PlotWidget()
        p.setBackground("#000")
        p.showGrid(x=True, y=True, alpha=0.2)
        p.setMenuEnabled(False)
        p.hideButtons()
        p.setMouseEnabled(False, False)
        p.setTitle(title, color=color, size="8pt")
        p.getAxis("left").setStyle(tickFont=pg.QtGui.QFont("monospace", 6))
        p.getAxis("bottom").setStyle(tickFont=pg.QtGui.QFont("monospace", 6))
        return p

    def _set_status(self, s):
        self.header.setText(f"Patient {PATIENT_ID}   •   {s}")

    def _update(self):
        raw, fe = self.acq.snapshot()
        self.c_raw.setData(self.x, raw)
        self.c_fe.setData(self.x, fe)
        # auto-range Y only (X is fixed & linked, so both stay time-aligned)
        for p, d in ((self.p_raw, raw), (self.p_fe, fe)):
            m = max(0.2, float(np.max(np.abs(d))) * 1.2)
            p.setYRange(-m, m, padding=0)

        self._frame += 1
        if self._frame % self.HR_EVERY == 0:
            self._update_hr(raw, fe)

    def _update_hr(self, raw, fe):
        fs = SAMPLE_RATE_HZ
        f = estimate_bpm(fe, fs, FHR_MIN, FHR_MAX, FHR_REFRACTORY_S)
        m = estimate_bpm(raw, fs, MHR_MIN, MHR_MAX, MHR_REFRACTORY_S)
        if f is not None:
            self.fhr = ema(self.fhr, f)
        if m is not None:
            self.mhr = ema(self.mhr, m)
        self.fhr_tile.set(self.fhr)
        self.mhr_tile.set(self.mhr)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="force simulated source")
    ap.add_argument("--file", default=None,
                    help="replay a .npy recording (1-D, 250 Hz) instead of live/sim")
    ap.add_argument("--no-loop", action="store_true",
                    help="with --file, play once and stop instead of looping")
    ap.add_argument("--windowed", action="store_true", help="do not fullscreen")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    acq = Acquirer(force_sim=args.sim, file_path=args.file, loop=not args.no_loop)
    win = Monitor(acq)
    acq.start()
    if FULLSCREEN and not args.windowed:
        win.showFullScreen()
    else:
        win.show()
    try:
        sys.exit(app.exec_())
    finally:
        acq.stop()


if __name__ == "__main__":
    main()
