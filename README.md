# EURS @ MaCVi WACV 2027 — team APC_TDC — Track 2 reproduction

This repository reproduces our reported Track 2 testing-phase submission **#19367** (`EURS_t2_sub013`, composite 0.421).
It follows the official template (https://github.com/CeviKle/EURS-MaCVi2027-repro-template). `METHOD.md` describes the
method, all training data, external data and pretrained models.

## Contents
| Path | What it is |
|---|---|
| `export_onnx.py` | `main(model_dir, out_onnx)`: regenerates the leaderboard ONNX from `weights/best.pt` (byte-identical on macOS; same graph, weights and outputs in the Linux Docker image) |
| `src/eurs/` | model code (`model.py`) and export helpers (`export_core.py`) |
| `weights/best.pt` | trained weights (EMA), SHA-256 `a22ad8178279ec390f5253d2e95fdc251354ae42c9a625140d7cb44d053e7acc` |
| `onnx/submitted_model.onnx` | the exact file uploaded for #19367, SHA-256 `dfb217bc31ea9e4e7d9b9480bb0364569e1042253ba1b8e454c34b6d053003e5` |
| `Dockerfile`, `docker-requirements.txt` | the export environment: Python 3.12, PyTorch 2.14.1 (CPU), onnx 1.23.2, onnxscript 0.7.2, onnxslim 0.1.98 |
| `repro.yaml`, `METHOD.md` | hand-in metadata and method description |
| `repro_check.py`, `docker/` | the organizers' checker and base image, unchanged |

## Run
```bash
python repro_check.py . --export-onnx               # regenerates the ONNX and compares it with onnx/submitted_model.onnx
# or, in the pinned environment without network:
docker build -t eurs-t2 .
docker run --rm --network none eurs-t2 python repro_check.py /pkg --export-onnx
```
The export runs on the CPU in a few seconds. A different onnxscript / onnxslim version can rename graph nodes, so use the
pinned environment. On Linux the file header (`producer_version 2.14.1+cpu`) and the order of the Conv attributes differ
from the uploaded file; graph, weights and outputs are identical.
