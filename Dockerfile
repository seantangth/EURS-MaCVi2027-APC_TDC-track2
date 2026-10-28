# Track 2 export environment. Our leaderboard ONNX (EURS_t2_sub013) was exported with PyTorch 2.14.1 and the exact
# onnxscript / onnxslim versions below; with them export_onnx.py regenerates a byte-identical file (checked with
# `python repro_check.py . --export-onnx`). The export runs on the CPU, so the CPU wheel of torch is enough.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 HOME=/tmp XDG_CACHE_HOME=/tmp/.cache TORCHINDUCTOR_CACHE_DIR=/tmp/.inductor
# torchvision (needed by timm) must come from the same CPU index: the PyPI build expects CUDA torch
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
COPY docker-requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN python -c "from importlib.metadata import version as v; import torch, torchvision, timm, onnx, onnxscript, onnxslim; print(*(p + ' ' + v(p) for p in ['torch', 'torchvision', 'timm', 'onnx', 'onnxscript', 'onnxslim']))"
COPY . /pkg
WORKDIR /pkg
