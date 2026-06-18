"""
Rolling-buffer real-time wrapper around the (batch) UNETR model.

The model consumes a full 992-sample window and returns the fECG for that whole
window. For a live monitor we keep a 992-sample ring buffer and re-run inference
every HOP_SAMPLES (200 ms). After each inference we "replay" the freshest
HOP_SAMPLES of the model's output, one per incoming raw sample, so the analyzed
trace scrolls smoothly and stays aligned with the raw trace (latency ~= one hop).

    proc = FecgProcessor()
    fe = proc.push(raw_sample)   # returns the aligned fECG sample
"""

import numpy as np

from config import WINDOW_SAMPLES, HOP_SAMPLES
from model_loader import load_extractor


class FecgProcessor:
    def __init__(self):
        self.extractor, self.info = load_extractor()
        self._buf = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._fecg = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._since_hop = 0
        self._filled = 0
        # index into _fecg we emit next (replays the last HOP fresh samples)
        self._emit = WINDOW_SAMPLES - 1

    def push(self, value):
        # append raw sample to the ring buffer
        self._buf[:-1] = self._buf[1:]
        self._buf[-1] = value
        self._filled = min(self._filled + 1, WINDOW_SAMPLES)
        self._since_hop += 1

        if self._since_hop >= HOP_SAMPLES and self._filled >= WINDOW_SAMPLES:
            self._since_hop = 0
            try:
                out = self.extractor(self._buf)
                if out.shape[0] != WINDOW_SAMPLES:
                    out = np.interp(
                        np.linspace(0, 1, WINDOW_SAMPLES),
                        np.linspace(0, 1, out.shape[0]),
                        out,
                    ).astype(np.float32)
                self._fecg = out
                # start replaying from the first of the freshest HOP samples
                self._emit = WINDOW_SAMPLES - HOP_SAMPLES
            except Exception:
                pass  # keep last window on transient failure

        idx = min(self._emit, WINDOW_SAMPLES - 1)
        self._emit = min(self._emit + 1, WINDOW_SAMPLES - 1)
        return float(self._fecg[idx])
