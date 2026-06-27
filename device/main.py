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
from collections import deque
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

# trend (mini-CTG): one point per HR recompute (~3/s); ~3 min of history
TREND_LEN = 600
FHR_NORMAL = (110, 160)   # normal fetal band drawn as reference lines


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

        # header: patient + signal source (no model name), plus a model-health
        # badge — we hide the model *name* but never hide a model *failure*:
        # without the real ONNX model the extractor falls back to a filter whose
        # output looks almost like raw, so it must be called out loudly.
        head_row = QtWidgets.QHBoxLayout()
        head_row.setContentsMargins(0, 0, 0, 0)
        head_row.setSpacing(6)
        self.header = QtWidgets.QLabel(f"Patient {PATIENT_ID}   •   starting…")
        self.header.setStyleSheet("color:#7CFC00; font:bold 11px monospace;")
        self.warn = QtWidgets.QLabel("⚠ NO MODEL")
        self.warn.setStyleSheet(
            "color:#000; background:#FF3B30; font:bold 10px monospace; padding:1px 4px;")
        info = getattr(self.acq.proc, "info", "") or ""
        self._model_ok = not ("placeholder" in info.lower() or "unavailable" in info.lower())
        self.warn.setVisible(not self._model_ok)
        head_row.addWidget(self.header)
        head_row.addStretch(1)
        head_row.addWidget(self.warn)
        layout.addLayout(head_row)

        # heart-rate readouts (FHR from fECG, MHR from raw) + a trend graph
        # filling the space to their right (mini cardiotocograph).
        hr_row = QtWidgets.QHBoxLayout()
        hr_row.setContentsMargins(0, 0, 0, 0)
        hr_row.setSpacing(4)
        self.fhr_tile = HRTile("FHR", FECG_COLOR)
        self.mhr_tile = HRTile("MHR", RAW_COLOR)
        hr_row.addWidget(self.fhr_tile)
        hr_row.addWidget(self.mhr_tile)

        self.p_trend = self._trend_plot()
        self.fhr_hist = deque([np.nan] * TREND_LEN, maxlen=TREND_LEN)
        self.mhr_hist = deque([np.nan] * TREND_LEN, maxlen=TREND_LEN)
        self._trend_x = np.arange(TREND_LEN)
        self.t_fhr = self.p_trend.plot([], [], pen=pg.mkPen(FECG_COLOR, width=1),
                                       connect="finite")
        self.t_mhr = self.p_trend.plot([], [], pen=pg.mkPen(RAW_COLOR, width=1),
                                       connect="finite")
        hr_row.addWidget(self.p_trend, 1)   # stretch=1 -> takes the leftover width
        layout.addLayout(hr_row)

        # one shared time axis so both traces scroll together in real time.
        # 0 s = oldest sample (left), PLOT_WINDOW_SECONDS = newest (right edge) —
        # positive seconds across the visible window, no confusing negatives.
        x = np.linspace(0, PLOT_WINDOW_SECONDS, BUF)

        # top trace shares the bottom trace's x-axis, so only label time once
        self.p_raw = self._plot("RAW", RAW_COLOR, show_time=False)
        self.c_raw = self.p_raw.plot(x, self.acq.raw, pen=pg.mkPen(RAW_COLOR, width=1))
        layout.addWidget(self.p_raw)

        self.p_fe = self._plot("fECG", FECG_COLOR, show_time=True)
        self.c_fe = self.p_fe.plot(x, self.acq.fecg, pen=pg.mkPen(FECG_COLOR, width=1))
        layout.addWidget(self.p_fe)

        # lock the fECG x-axis to the raw one: identical window, no drift
        self.p_fe.setXLink(self.p_raw)
        self.p_raw.setXRange(0, PLOT_WINDOW_SECONDS, padding=0)

        self.x = x
        self.fhr = None
        self.mhr = None
        self._frame = 0
        self._source = "starting…"
        self.acq.status.connect(self._set_status)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(33)  # ~30 fps

    def _plot(self, title, color, show_time=True):
        p = pg.PlotWidget()
        p.setBackground("#000")
        p.showGrid(x=True, y=True, alpha=0.2)
        p.setMenuEnabled(False)
        p.hideButtons()
        p.setMouseEnabled(False, False)
        p.setTitle(title, color=color, size="8pt")
        tick = pg.QtGui.QFont("monospace", 6)
        lbl = {"color": "#aaa", "font-size": "7pt"}
        # units in the label text (no units= kwarg -> avoids pyqtgraph's SI
        # auto-prefix turning "mV" into "mmV")
        left = p.getAxis("left")
        left.setStyle(tickFont=tick)
        left.setLabel("amplitude (mV)", **lbl)           # Y axis = signal amplitude
        bottom = p.getAxis("bottom")
        bottom.setStyle(tickFont=tick)
        if show_time:
            bottom.setLabel("time (s)", **lbl)           # X axis = seconds in window
        else:
            # top plot shares the x-axis with the bottom one; keep it clean
            bottom.setStyle(showValues=False)
        return p

    def _trend_plot(self):
        """Mini cardiotocograph: FHR/MHR over the last few minutes (bpm)."""
        p = pg.PlotWidget()
        p.setBackground("#000")
        p.setMenuEnabled(False)
        p.hideButtons()
        p.setMouseEnabled(False, False)
        p.setTitle("TREND (bpm)", color="#aaa", size="7pt")
        p.setMaximumHeight(80)               # don't steal height from the waveforms
        p.showGrid(x=False, y=True, alpha=0.2)
        p.setYRange(50, 210, padding=0)
        p.setXRange(0, TREND_LEN, padding=0)
        left = p.getAxis("left")
        left.setStyle(tickFont=pg.QtGui.QFont("monospace", 6))
        left.setTicks([[(60, "60"), (110, "110"), (160, "160"), (200, "200")]])
        p.getAxis("bottom").setStyle(showValues=False)   # time axis, no clutter
        # faint dashed reference lines for the normal fetal band
        for y in FHR_NORMAL:
            p.addItem(pg.InfiniteLine(
                pos=y, angle=0,
                pen=pg.mkPen("#664455", width=1, style=QtCore.Qt.DashLine)))
        return p

    def _set_status(self, s):
        self._source = s
        self._refresh_header()

    def _refresh_header(self):
        """Compose the header from patient identity (from server) + source."""
        p = getattr(self.acq.uploader, "patient", None) or {}
        bits = [PATIENT_ID]
        name = p.get("name")
        if name and name != PATIENT_ID:
            bits.append(name)
        if p.get("mrn"):
            bits.append(f"MRN {p['mrn']}")
        self.header.setText(f"{' · '.join(bits)}   •   {self._source}")

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

        # append to the trend (NaN where unknown so the line breaks, not dips)
        self.fhr_hist.append(self.fhr if self.fhr is not None else np.nan)
        self.mhr_hist.append(self.mhr if self.mhr is not None else np.nan)
        self.t_fhr.setData(self._trend_x, np.fromiter(self.fhr_hist, dtype=float),
                           connect="finite")
        self.t_mhr.setData(self._trend_x, np.fromiter(self.mhr_hist, dtype=float),
                           connect="finite")

        # patient identity arrives async from the server after connect
        self._refresh_header()

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
