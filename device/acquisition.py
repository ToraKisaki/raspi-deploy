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
    """Replays a saved recording at the sample rate, as if it were live.

    Supports .npy (1D or 2D) and .npz (looks for 'mix' array).
    Also supports pointing to a directory to loop over all .npy/.npz files.
    Marked is_simulated=True so callers treat it like the simulator, not hardware.
    """
    is_simulated = True

    def __init__(self, path, fs=SAMPLE_RATE_HZ, channel=ABDOMINAL_CHANNEL, loop=True):
        import os
        import glob
        self.path = path
        self.fs = fs
        self.channel = channel
        self.loop = loop
        
        if os.path.isdir(path):
            # Sort files to play them in order
            self.files = sorted(glob.glob(os.path.join(path, "*.npy")) + glob.glob(os.path.join(path, "*.npz")))
            self.label = f"DIR:{os.path.basename(path)}"
        else:
            self.files = [path]
            self.label = f"FILE:{os.path.basename(path)}"
            
        self.file_idx = 0
        self._load_current_file()

    def _load_current_file(self):
        if not self.files:
            self.signal = np.zeros(1000, dtype=np.float32)
            self.n = 1000
            return
            
        curr_path = self.files[self.file_idx]
        data = np.load(curr_path)
        if str(curr_path).endswith('.npz') and 'mix' in data:
            arr = np.asarray(data['mix'], dtype=np.float32)
        else:
            arr = np.asarray(data, dtype=np.float32)

        if arr.ndim == 2:
            if arr.shape[0] <= arr.shape[1]:        # (C, N) -> pick a row
                arr = arr[min(self.channel, arr.shape[0] - 1)]
            else:                                    # (N, C) -> pick a column
                arr = arr[:, min(self.channel, arr.shape[1] - 1)]
        self.signal = arr.ravel()
        self.n = int(self.signal.shape[0])

    def samples(self):
        dt = 1.0 / self.fs
        i = 0
        while True:
            if i >= self.n:
                self.file_idx += 1
                if self.file_idx >= len(self.files):
                    if not self.loop:
                        return
                    self.file_idx = 0
                self._load_current_file()
                i = 0
                continue
            
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
              f"({src.n} samples in current file, {src.n / SAMPLE_RATE_HZ:.1f}s @ {SAMPLE_RATE_HZ} Hz, "
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
