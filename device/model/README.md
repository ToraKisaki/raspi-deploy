# (Model lives at the repo root, not here)

The live monitor does **not** load a model from this folder. It reuses your
existing trained pipeline directly:

- model:   `../../models/unetr_int8.onnx`
- pipeline: `../../infer.py`  (band-pass [3-90 Hz] → /global_var → UNETR INT8 →
  ×global_var → fECG = output channel 1)

`device/model_loader.py` imports `infer.py` from the repo root so there is a
single source of truth for the model maths. To swap models, point
`infer.MODEL_PATH` at a new `.onnx` (its input must stay `[1, 1, 992]` @ 250 Hz)
— nothing in `device/` needs to change.

If `onnxruntime`/`scipy`/the model can't be loaded (e.g. on a dev laptop), the
monitor automatically falls back to a placeholder filter so it still runs.
