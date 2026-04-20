"""
Fetal ECG extraction inference for Raspberry Pi 5.
Usage:
  python infer.py --input signal.npy --output fecg.npy
  python infer.py --input signal.npy --plot
"""

import argparse
import numpy as np
import onnxruntime as ort
from scipy.signal import butter, filtfilt
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "unetr_int8.onnx")
SAMPLE_RATE = 250        # Hz
SAMPLE_LEN  = 992        # samples per window (~3.97 s)
GLOBAL_VAR  = 7.2159853  # training normalization constant (var_global)
BANDPASS    = (3.0, 90.0)


def bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal).astype(np.float32)


def preprocess(signal_992: np.ndarray) -> np.ndarray:
    x = np.asarray(signal_992, dtype=np.float32).copy()
    x = bandpass_filter(x, BANDPASS[0], BANDPASS[1], SAMPLE_RATE)
    x = x / np.float32(GLOBAL_VAR)
    return x[None, None, :]  # [1, 1, 992]


def postprocess(output: np.ndarray, scale: float = GLOBAL_VAR) -> np.ndarray:
    # output shape: [1, 2, 992] (dual) or [1, 1, 992] (single)
    if output.shape[1] == 2:
        fecg = output[0, 1, :]  # fECG is channel index 1
    else:
        fecg = output[0, 0, :]
    return (fecg * scale).astype(np.float32)


def load_session(model_path: str = MODEL_PATH) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # RPi 5 has 4 cores
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


def run_inference(session: ort.InferenceSession, signal_992: np.ndarray) -> np.ndarray:
    x = preprocess(signal_992)
    out = session.run(None, {"input": x})[0]
    return postprocess(out)


def sliding_window_inference(session: ort.InferenceSession, signal: np.ndarray, step: int = SAMPLE_LEN) -> np.ndarray:
    """Run inference on a long signal using non-overlapping windows."""
    n = len(signal)
    output = np.zeros(n, dtype=np.float32)
    for start in range(0, n - SAMPLE_LEN + 1, step):
        window = signal[start:start + SAMPLE_LEN]
        output[start:start + SAMPLE_LEN] = run_inference(session, window)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Input .npy file (1D array, 250 Hz)")
    parser.add_argument("--output", default="fecg_out.npy", help="Output .npy path")
    parser.add_argument("--model",  default=MODEL_PATH, help="ONNX model path")
    parser.add_argument("--plot",   action="store_true", help="Plot result (requires matplotlib)")
    args = parser.parse_args()

    signal = np.load(args.input).astype(np.float32).ravel()
    print(f"Input: {len(signal)} samples ({len(signal)/SAMPLE_RATE:.1f} s @ {SAMPLE_RATE} Hz)")

    session = load_session(args.model)
    print(f"Model loaded: {args.model}")

    if len(signal) == SAMPLE_LEN:
        fecg = run_inference(session, signal)
    else:
        fecg = sliding_window_inference(session, signal)

    np.save(args.output, fecg)
    print(f"fECG saved: {args.output}  shape={fecg.shape}")

    if args.plot:
        import matplotlib.pyplot as plt
        t = np.arange(len(signal)) / SAMPLE_RATE
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        ax1.plot(t[:len(signal)], signal, lw=0.7, label="mixture")
        ax1.set_ylabel("Amplitude"); ax1.legend()
        ax2.plot(t[:len(fecg)], fecg, lw=0.7, color="orange", label="fECG (extracted)")
        ax2.set_ylabel("Amplitude"); ax2.set_xlabel("Time (s)"); ax2.legend()
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
