# raspi-deploy — Fetal ECG Extraction on Raspberry Pi 5

Real-time non-invasive fetal ECG extraction from single-channel maternal ECG recordings.
Model: UNETR INT8 (29 MB), inference via ONNX Runtime CPU.

---

## Hardware

| Item | Spec |
|------|------|
| Device | Raspberry Pi 5 (4 GB RAM) |
| OS | Raspberry Pi OS 64-bit (Bookworm) |
| Python | 3.10+ |
| ECG sensor | Single-electrode, 250 Hz sampling rate |

---

## Installation

```bash
pip install -r requirements.txt
```

> On RPi 5, `onnxruntime` for aarch64:
> ```bash
> pip install onnxruntime
> ```
> If unavailable on your Python version, use the wheel from:
> https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux

---

## Input Format

The script accepts a **1D NumPy array** (`.npy` file) of raw ECG samples:

| Property | Value |
|----------|-------|
| Sampling rate | **250 Hz** |
| Minimum length | **992 samples** (~3.97 s) |
| Data type | `float32` or `float64` |
| Content | Raw mixture signal (maternal + fetal ECG) |

**Create input from CSV or raw data:**
```python
import numpy as np

# From CSV (one column, no header)
signal = np.loadtxt("ecg_record.csv", dtype=np.float32)
np.save("input.npy", signal)

# From list / array
signal = np.array([...], dtype=np.float32)   # length >= 992
np.save("input.npy", signal)
```

---

## Usage

### Basic — single 992-sample window
```bash
python infer.py --input input.npy
# output saved to: fecg_out.npy
```

### Specify output path
```bash
python infer.py --input input.npy --output results/fecg.npy
```

### Long recording — sliding window (auto-detected)
```bash
# signal longer than 992 samples → processed in non-overlapping 992-sample windows
python infer.py --input long_record.npy --output fecg_long.npy
```

### Use a different model
```bash
python infer.py --input input.npy --model models/other_model.onnx
```

### Visualize result (requires matplotlib)
```bash
python infer.py --input input.npy --plot
```

### All flags
```
--input   PATH    Input .npy file (required)
--output  PATH    Output .npy path (default: fecg_out.npy)
--model   PATH    ONNX model path (default: models/unetr_int8.onnx)
--plot            Show matplotlib plot of mixture vs extracted fECG
```

---

## Output

A `.npy` file containing the extracted fetal ECG signal:
- Same length as the processed portion of the input
- Unit: same as input amplitude (millivolts if input is in mV)
- Sampling rate: 250 Hz

```python
import numpy as np
fecg = np.load("fecg_out.npy")   # shape: (N,), float32
```

---

## Signal Pipeline (inside infer.py)

```
raw input (250 Hz)
    → Butterworth bandpass filter [3–90 Hz, order 3]
    → Divide by global_var (7.2160)          ← training normalization
    → UNETR INT8 inference [1, 1, 992]
    → Multiply by global_var                  ← denormalize
    → fECG output
```

---

## Performance (RPi 5, 4 GB)

| Model | Size | ~Inference / window |
|-------|------|---------------------|
| UNETR INT8 | 29 MB | ~80–120 ms |

One window = 992 samples = 3.97 s of signal.
Throughput ~8–12× real-time on RPi 5.

---

## Model Info

| Property | Value |
|----------|-------|
| Architecture | UNETR (dual-branch, 25.8M params) |
| Quantization | Dynamic INT8 (MatMul/Gemm only) |
| Training data | fecgsyndb 1-subject, 34-channel |
| Input shape | `[B, 1, 992]` |
| Output shape | `[B, 2, 992]` → fECG = channel index 1 |
| Normalize mode | `var_global` (global_var = 7.2160) |
| Val SSIM | ~0.796 (12,800 samples) |
| Val PSNR | ~26.82 dB |
