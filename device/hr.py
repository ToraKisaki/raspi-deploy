"""
Numpy-only Pan-Tompkins-style heart-rate estimator for the on-device monitor.

Mirrors the detector in ``server/analysis.py`` so the Pi can show FHR/MHR
locally without importing the server package:

    baseline removal -> derivative -> squaring -> moving-window integration ->
    adaptive threshold with a physiologic refractory period.

    bpm = estimate_bpm(window, fs, min_bpm, max_bpm, refractory_s)

Cheap enough to run on the full plot window a few times per second on a Pi.
"""

import numpy as np

# detector search ranges (bpm) — a little past the alarm bands for margin
FHR_MIN, FHR_MAX = 90, 220     # fetal, from the AI-extracted fECG channel
MHR_MIN, MHR_MAX = 45, 140     # maternal, from the dominant QRS in raw
FHR_REFRACTORY_S = 0.25
MHR_REFRACTORY_S = 0.35

MIN_SEC = 2.5                  # need at least this much signal before estimating


def _moving_avg(x, w):
    if w <= 1:
        return x
    k = np.ones(w, dtype=np.float64) / w
    return np.convolve(x, k, mode="same")


def _pick_peaks(x, thr, refractory):
    """Local maxima above ``thr`` with a min spacing of ``refractory`` samples."""
    peaks = []
    n = len(x)
    for i in range(1, n - 1):
        xi = x[i]
        if xi >= thr and xi >= x[i - 1] and xi >= x[i + 1]:
            if peaks and (i - peaks[-1]) < refractory:
                if xi > x[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return np.asarray(peaks, dtype=np.int64)


def estimate_bpm(sig, fs, min_bpm, max_bpm, refractory_s):
    """Estimate heart rate (bpm) from a 1-D ECG-like window, or None."""
    x = np.asarray(sig, dtype=np.float64)
    n = x.size
    if n < int(fs * MIN_SEC):
        return None
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x)

    x = x - _moving_avg(x, max(1, int(0.2 * fs)))     # baseline removal
    d = np.diff(x, prepend=x[:1])                     # derivative
    e = d * d                                         # squaring
    integ = _moving_avg(e, max(1, int(0.04 * fs)))    # ~40 ms integration
    peak_ref = float(np.percentile(integ, 99))
    if peak_ref <= 0:
        return None
    thr = 0.35 * peak_ref
    refr = max(1, int(refractory_s * fs))
    peaks = _pick_peaks(integ, thr, refr)
    if peaks.size < 3:
        return None

    rr = np.diff(peaks) / fs
    lo, hi = 60.0 / max_bpm, 60.0 / min_bpm
    rr = rr[(rr >= lo) & (rr <= hi)]
    if rr.size < 2:
        return None
    return 60.0 / float(np.median(rr))


def ema(prev, new, alpha=0.45):
    return new if prev is None else (alpha * new + (1 - alpha) * prev)
