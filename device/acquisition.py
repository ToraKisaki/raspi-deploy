"""
Acquisition source abstraction.

Tries the real ADS1293 first; if hardware/lgpio is unavailable it transparently
falls back to a synthetic generator that produces a realistic abdominal ECG
(maternal QRS + a smaller, faster fetal QRS + noise). This lets you develop and
demo on any machine, then run unchanged on the Pi.

Usage:
    src = make_source()           # auto: hardware or simulator
    for sample in src.samples():  # yields list[float] mV, one per channel
        ...
"""

import math
import time
import numpy as np

from config import (
    SAMPLE_RATE_HZ, N_INPUT_CHANNELS, ABDOMINAL_CHANNEL,
)


class FileSource:
    """Replays a saved .npy recording at the sample rate, as if it were live.

    The file is treated as a 1-D raw abdominal-lead signal at SAMPLE_RATE_HZ.
    A 2-D array is reduced to one channel (default ABDOMINAL_CHANNEL): shape
    (C, N) -> row, shape (N, C) -> column. Set loop=False to play once and stop.
    Marked is_simulated=True so callers treat it like the simulator, not hardware.
    """
    is_simulated = True

    def __init__(self, path, fs=SAMPLE_RATE_HZ, channel=ABDOMINAL_CHANNEL, loop=True):
        import os
        self.path = path
        self.fs = fs
        self.loop = loop
        arr = np.asarray(np.load(path), dtype=np.float32)
        if arr.ndim == 2:
            if arr.shape[0] <= arr.shape[1]:        # (C, N) -> pick a row
                arr = arr[min(channel, arr.shape[0] - 1)]
            else:                                    # (N, C) -> pick a column
                arr = arr[:, min(channel, arr.shape[1] - 1)]
        self.signal = arr.ravel()
        self.n = int(self.signal.shape[0])
        self.label = f"FILE:{os.path.basename(path)}"

    def samples(self):
        dt = 1.0 / self.fs
        i = 0
        while True:
            if i >= self.n:
                if not self.loop:
                    return
                i = 0
            ch = [0.0] * N_INPUT_CHANNELS
            ch[ABDOMINAL_CHANNEL] = float(self.signal[i])
            i += 1
            yield ch
            time.sleep(dt)


class SimulatedSource:
    is_simulated = True
    label = "SIMULATOR"

    def __init__(self, fs=SAMPLE_RATE_HZ):
        self.fs = fs
        self.t = 0.0
        self.mat_hr = 75.0    # maternal bpm
        self.fet_hr = 140.0   # fetal bpm
        self._rng = np.random.default_rng(7)

    @staticmethod
    def _qrs(phase, width, amp):
        # a simple biphasic spike centred at phase==0 (phase in [-0.5,0.5))
        x = phase / width
        return amp * (1 - x * x) * math.exp(-(x * x) * 4) if abs(x) < 1.5 else 0.0

    def _beat(self, t, hr, amp, width):
        period = 60.0 / hr
        phase = (t % period) / period - 0.0
        # distance to nearest beat
        d = phase if phase < 0.5 else phase - 1.0
        return self._qrs(d * period, width, amp)

    def samples(self):
        dt = 1.0 / self.fs
        while True:
            mat = self._beat(self.t, self.mat_hr, amp=1.0, width=0.05)
            fet = self._beat(self.t, self.fet_hr, amp=0.18, width=0.025)
            baseline = 0.05 * math.sin(2 * math.pi * 0.3 * self.t)  # respiration
            noise = float(self._rng.normal(0, 0.02))
            abd = mat + fet + baseline + noise
            ch = [0.0] * N_INPUT_CHANNELS
            ch[ABDOMINAL_CHANNEL] = abd
            # fill other channels with maternal-dominant leads
            for i in range(N_INPUT_CHANNELS):
                if i != ABDOMINAL_CHANNEL:
                    ch[i] = mat * (0.8 + 0.1 * i) + float(self._rng.normal(0, 0.02))
            self.t += dt
            yield ch
            time.sleep(dt)


class HardwareSource:
    is_simulated = False
    label = "ADS1293"

    def __init__(self, ads):
        self.ads = ads

    def samples(self):
        # busy-wait on DRDY, pace to nominal rate as a safety net
        min_dt = 1.0 / (SAMPLE_RATE_HZ * 4)
        last = 0.0
        while True:
            if self.ads.data_ready():
                yield self.ads.read_sample()
            else:
                now = time.time()
                if now - last < min_dt:
                    time.sleep(min_dt)
                last = now


def make_source(force_sim=False, file_path=None, loop=True):
    if file_path:
        src = FileSource(file_path, loop=loop)
        print(f"[acquisition] FILE playback: {file_path}  "
              f"({src.n} samples, {src.n / SAMPLE_RATE_HZ:.1f}s @ {SAMPLE_RATE_HZ} Hz, "
              f"loop={loop})")
        return src
    if not force_sim:
        try:
            from ads1293 import ADS1293, HardwareUnavailable
            try:
                ads = ADS1293()
                ads.start()
                print("[acquisition] ADS1293 hardware ready")
                return HardwareSource(ads)
            except HardwareUnavailable as e:
                print(f"[acquisition] hardware unavailable: {e}")
        except Exception as e:
            print(f"[acquisition] driver import failed: {e}")
    print("[acquisition] using SIMULATED ECG source")
    return SimulatedSource()
