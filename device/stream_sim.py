#!/usr/bin/env python3
"""
Headless device simulator: generates a realistic abdominal ECG, runs the fECG
extractor, and streams to the backend exactly like the real Pi app — but with
no GUI and no hardware. Use it to test the server + web dashboard from any
machine.

    python3 stream_sim.py                 # streams patient P001 to localhost
    python3 stream_sim.py --patient P002 --url ws://SERVER:8000/ws/device
"""
import time
import argparse
from collections import deque

import config
from acquisition import SimulatedSource
from fecg_processor import FecgProcessor
from uploader import Uploader
from hr import (estimate_bpm, ema, FHR_MIN, FHR_MAX, MHR_MIN, MHR_MAX,
                FHR_REFRACTORY_S, MHR_REFRACTORY_S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default=config.PATIENT_ID)
    ap.add_argument("--url", default=config.SERVER_WS_URL)
    ap.add_argument("--seconds", type=float, default=0, help="0 = forever")
    args = ap.parse_args()

    proc = FecgProcessor()
    print(f"[sim] {proc.info}")
    up = Uploader(url=args.url, patient_id=args.patient)
    src = SimulatedSource()
    t0 = time.time()
    n = 0
    window_size = int(config.SAMPLE_RATE_HZ * config.PLOT_WINDOW_SECONDS)
    mecg_window = deque(maxlen=window_size)
    fecg_window = deque(maxlen=window_size)
    fhr = mhr = None
    for ch in src.samples():
        raw = float(ch[config.ABDOMINAL_CHANNEL])
        me, fe = proc.push(raw)
        mecg_window.append(me)
        fecg_window.append(fe)
        if n % 50 == 0 and len(fecg_window) == window_size:
            fetal = estimate_bpm(fecg_window, config.SAMPLE_RATE_HZ,
                                 FHR_MIN, FHR_MAX, FHR_REFRACTORY_S)
            maternal = estimate_bpm(mecg_window, config.SAMPLE_RATE_HZ,
                                    MHR_MIN, MHR_MAX, MHR_REFRACTORY_S)
            if fetal is not None:
                fhr = ema(fhr, fetal)
            if maternal is not None:
                mhr = ema(mhr, maternal)
            up.set_metrics(fhr, mhr)
        up.push(time.time() - t0, raw, fe)
        n += 1
        if n % config.SAMPLE_RATE_HZ == 0:
            print(f"[sim] streamed {n} samples  connected={up.connected}")
        if args.seconds and (time.time() - t0) >= args.seconds:
            break
    up.stop()


if __name__ == "__main__":
    main()
