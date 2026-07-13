"""
fECG extractor for the live monitor.

This wraps your *existing* trained pipeline in ../infer.py (UNETR INT8 ONNX,
250 Hz, 992-sample windows, Butterworth [3-90] band-pass + global_var
normalization, fECG = output channel 1). There is ONE source of truth for the
model maths: infer.py. We just import it.

    extractor, info = load_extractor()
    mecg_992, fecg_992 = extractor(window_992)

If onnxruntime / scipy / the model file aren't available (e.g. developing on a
laptop without the wheel), we fall back to a lightweight template-subtraction
filter so the whole monitor still runs and can be demoed.
"""

import os
import sys
import numpy as np

from config import WINDOW_SAMPLES

# repo root = parent of this device/ folder; that's where infer.py + models/ live
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class UnetrExtractor:
    """Thin wrapper over infer.py so behaviour is identical to batch inference."""
    def __init__(self):
        import infer  # noqa: your script at repo root
        self._infer = infer
        if infer.SAMPLE_LEN != WINDOW_SAMPLES:
            raise RuntimeError(
                f"config WINDOW_SAMPLES ({WINDOW_SAMPLES}) != infer.SAMPLE_LEN "
                f"({infer.SAMPLE_LEN}); keep them in sync."
            )
        self.session = infer.load_session()
        self.name = f"UNETR INT8 ONNX ({os.path.basename(infer.MODEL_PATH)})"

    def __call__(self, window):
        w = np.asarray(window, dtype=np.float32).ravel()
        if w.shape[0] != WINDOW_SAMPLES:
            # pad/trim defensively
            out = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            n = min(w.shape[0], WINDOW_SAMPLES)
            out[:n] = w[:n]
            w = out
        mecg, fecg = self._infer.run_inference_components(self.session, w)
        return mecg.astype(np.float32), fecg.astype(np.float32)


class PlaceholderExtractor:
    """No-model fallback: template-subtraction estimate (NOT your model)."""
    name = "placeholder (template-subtraction filter — model unavailable)"

    def __init__(self, fs=250):
        self.fs = fs

    def _bandpass(self, x, lo, hi):
        x = np.asarray(x, dtype=np.float32)
        rc_hp = 1.0 / (2 * np.pi * lo)
        a_hp = rc_hp / (rc_hp + 1.0 / self.fs)
        y = np.zeros_like(x)
        px, py = x[0], 0.0
        for i in range(len(x)):
            py = a_hp * (py + x[i] - px)
            px = x[i]
            y[i] = py
        rc_lp = 1.0 / (2 * np.pi * hi)
        a_lp = (1.0 / self.fs) / (rc_lp + 1.0 / self.fs)
        z = np.zeros_like(y)
        acc = y[0]
        for i in range(len(y)):
            acc += a_lp * (y[i] - acc)
            z[i] = acc
        return z

    def __call__(self, window):
        x = np.asarray(window, dtype=np.float32)
        mat = self._bandpass(x, 1.0, 25.0)
        thr = 3.0 * np.std(mat) + 1e-6
        peaks, refr, i = [], int(0.25 * self.fs), 1
        while i < len(mat) - 1:
            if mat[i] > thr and mat[i] >= mat[i - 1] and mat[i] >= mat[i + 1]:
                peaks.append(i)
                i += refr
            else:
                i += 1
        residual = x.copy()
        half = int(0.12 * self.fs)
        segs = [x[p - half:p + half] for p in peaks
                if p - half >= 0 and p + half < len(x)]
        if len(segs) >= 2:
            tmpl = np.mean(np.stack(segs), axis=0)
            for p in peaks:
                a, b = p - half, p + half
                if a >= 0 and b < len(x):
                    residual[a:b] -= tmpl
        fecg = (self._bandpass(residual, 3.0, 45.0) * 2.0).astype(np.float32)
        return mat.astype(np.float32), fecg


def load_extractor():
    try:
        ex = UnetrExtractor()
        return ex, f"Loaded {ex.name}"
    except Exception as e:
        ex = PlaceholderExtractor()
        return ex, f"infer.py/model unavailable ({e}); using {ex.name}"
