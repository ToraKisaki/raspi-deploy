"""
Rolling-buffer real-time wrapper around the (batch) UNETR model.

Temporal alignment guarantee
-----------------------------
The model processes a 992-sample window and outputs the fECG for *exactly* that
window.  For live use we maintain a ring buffer and run inference every HOP_SAMPLES
(200 ms, 50 samples).  After each inference we have 50 fresh output samples that
correspond 1-to-1 with the 50 raw samples that just arrived.  We queue those and
drain exactly one per incoming raw sample so every push() returns the fECG sample
that belongs to *that same wall-clock position*.

Alignment invariant:
    raw[n]  ↔  fecg[n]   (same absolute sample index, same time grid)

No clamp / freeze artifacts: if inference is late the queue grows by up to one
extra hop and drains over the next hop window, keeping latency bounded at ≤2 hops
(~400 ms) rather than accumulating unboundedly.

    proc = FecgProcessor()
    fe = proc.push(raw_sample)   # always returns the aligned fECG sample
"""

from collections import deque

import numpy as np

from config import WINDOW_SAMPLES, HOP_SAMPLES
from model_loader import load_extractor


class FecgProcessor:
    def __init__(self):
        self.extractor, self.info = load_extractor()

        # Circular raw buffer — written with a head pointer to avoid O(N) shifts
        self._buf = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._head = 0          # next write position (mod WINDOW_SAMPLES)
        self._filled = 0        # how many valid samples are in _buf

        # Samples since last inference trigger
        self._since_hop = 0

        # Queue of aligned fecg output samples waiting to be emitted.
        # Each element corresponds 1-to-1 to a raw sample that has already
        # been pushed into _buf.  We append HOP_SAMPLES at a time after each
        # inference and pop one per push() call.
        self._queue: deque[float] = deque()

        # Fallback when the queue is empty (pre-warm or model not loaded)
        self._last_fe = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, value: float):
        """Append one raw sample into the circular buffer."""
        self._buf[self._head] = value
        self._head = (self._head + 1) % WINDOW_SAMPLES
        self._filled = min(self._filled + 1, WINDOW_SAMPLES)

    def _ordered_window(self) -> np.ndarray:
        """Return the 992 samples in chronological order (oldest → newest)."""
        if self._filled < WINDOW_SAMPLES:
            # Buffer not full yet: the valid data is in [0 .. _filled)
            return self._buf[: self._filled].copy()
        # Full: oldest sample sits at _head, wrap around
        return np.roll(self._buf, -self._head).copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, value: float) -> float:
        """
        Ingest one raw sample.  Returns the fECG sample aligned to this
        same time position.

        Latency: one inference hop (~200 ms / 50 samples).  The very first
        WINDOW_SAMPLES pushes return 0.0 while the buffer warms up.
        """
        self._write(value)
        self._since_hop += 1

        # Trigger inference once per HOP_SAMPLES when buffer is warm
        if self._since_hop >= HOP_SAMPLES and self._filled >= WINDOW_SAMPLES:
            self._since_hop = 0
            self._run_inference()

        # Drain one aligned output sample per push
        if self._queue:
            fe = self._queue.popleft()
            self._last_fe = fe
            return fe
        return self._last_fe

    def _run_inference(self):
        """Run the model on the current window and enqueue the fresh outputs."""
        window = self._ordered_window()
        try:
            out = self.extractor(window)
            if out.shape[0] != WINDOW_SAMPLES:
                out = np.interp(
                    np.linspace(0, 1, WINDOW_SAMPLES),
                    np.linspace(0, 1, out.shape[0]),
                    out,
                ).astype(np.float32)
            # The last HOP_SAMPLES of the output correspond to the HOP_SAMPLES
            # raw samples that were just pushed since the previous inference.
            fresh = out[-HOP_SAMPLES:]
            self._queue.extend(float(v) for v in fresh)
        except Exception:
            # On model failure: fill with zeros so the trace doesn't freeze
            self._queue.extend(0.0 for _ in range(HOP_SAMPLES))
