"""
Benchmark SSIM/PSNR and inference latency on test batch.
Usage:
  python benchmark.py
  python benchmark.py --test-mix examples/test_mix.npy --test-fecg examples/test_fecg.npy
  python benchmark.py --model models/unetr_int8.onnx --threads 4
"""

import argparse
import time
import numpy as np
from infer import load_session, run_inference, SAMPLE_LEN


def ssim_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mu_t, mu_p = y_true.mean(), y_pred.mean()
    sigma_t  = np.mean((y_true - mu_t) ** 2)
    sigma_p  = np.mean((y_pred - mu_p) ** 2)
    sigma_tp = np.mean((y_true - mu_t) * (y_pred - mu_p))
    data_range = float(y_true.max() - y_true.min())
    if data_range < 1e-8:
        return 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_t * mu_p + c1) * (2 * sigma_tp + c2)
    den = (mu_t**2 + mu_p**2 + c1) * (sigma_t + sigma_p + c2)
    return float(num / den) if den > 1e-12 else 0.0


def psnr_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = np.mean((y_true - y_pred) ** 2)
    if mse < 1e-12:
        return 100.0
    data_range = float(y_true.max() - y_true.min())
    if data_range < 1e-8:
        return 0.0
    return float(10 * np.log10(data_range**2 / mse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mix",  default="examples/test_mix.npy")
    parser.add_argument("--test-fecg", default="examples/test_fecg.npy")
    parser.add_argument("--model",     default="models/unetr_int8.onnx")
    parser.add_argument("--threads",   type=int, default=4)
    args = parser.parse_args()

    test_mix  = np.load(args.test_mix).astype(np.float32)   # (N, 992)
    test_fecg = np.load(args.test_fecg).astype(np.float32)  # (N, 992)
    N = len(test_mix)
    print(f"Test samples : {N}")
    print(f"Model        : {args.model}")
    print(f"Threads      : {args.threads}")

    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model, sess_options=opts, providers=["CPUExecutionProvider"])

    # Warmup
    for i in range(3):
        run_inference(sess, test_mix[i])

    ssims, psnrs, latencies = [], [], []

    for i in range(N):
        t0 = time.perf_counter()
        pred = run_inference(sess, test_mix[i])
        latencies.append((time.perf_counter() - t0) * 1000)
        ssims.append(ssim_1d(test_fecg[i], pred))
        psnrs.append(psnr_1d(test_fecg[i], pred))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{N}] running SSIM={np.mean(ssims):.4f}  PSNR={np.mean(psnrs):.2f}dB  lat={np.mean(latencies):.1f}ms")

    print()
    print("=" * 50)
    print(f"{'SSIM (mean)':<20} {np.mean(ssims):.6f}")
    print(f"{'SSIM (std)':<20} {np.std(ssims):.6f}")
    print(f"{'PSNR (mean)':<20} {np.mean(psnrs):.2f} dB")
    print(f"{'PSNR (std)':<20} {np.std(psnrs):.2f} dB")
    print(f"{'Latency mean':<20} {np.mean(latencies):.1f} ms")
    print(f"{'Latency median':<20} {np.median(latencies):.1f} ms")
    print(f"{'Latency min':<20} {np.min(latencies):.1f} ms")
    print(f"{'Latency max':<20} {np.max(latencies):.1f} ms")
    print("=" * 50)
    window_s = SAMPLE_LEN / 250.0
    print(f"Real-time factor: {window_s / (np.mean(latencies)/1000):.1f}x  "
          f"(window={window_s:.2f}s, inference={np.mean(latencies):.1f}ms)")


if __name__ == "__main__":
    main()
