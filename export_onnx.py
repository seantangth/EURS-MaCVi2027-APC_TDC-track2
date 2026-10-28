"""Track 2: regenerate EURS_t2_sub013 (leaderboard submission #19367) from weights/best.pt.

The organizers call   main(model_dir, out_onnx).   Every step between the checkpoint and the uploaded file is here, in the
same order as our original export: build -> load EMA weights -> per-class segmentation logit offsets -> HardSwish/HardSigmoid
rewritten as Mul/Add/Clip -> output-stage detail enhancement (water-type weighted) with portrait detection -> torch.onnx
dynamo export (opset 18, weights embedded) -> onnxslim -> merge duplicate initializers.
"""
import os
import sys

import onnx
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from eurs.model import build  # noqa: E402
from eurs.export_core import _dedup_initializers, basic_acts  # noqa: E402

PRESET = "mnv3l768"
HW = (768, 1024)
SEG_BIAS = "fish:0.5,elasmo:1.5"     # added to the last segmentation-head bias (no extra op, FLOPs unchanged)
SHARPEN = ("g3y", 0.25, 3.0)         # luminance unsharp gain, linear in the predicted water type 1 -> 10
SHARPEN_PORT = (3.0, 1.45)           # fixed gain 3 when the stretched input is detected as a portrait image


def main(model_dir, out_onnx):
    ck = torch.load(os.path.join(model_dir, "best.pt"), map_location="cpu")
    args = ck.get("args", {}) or {}
    model = build(PRESET, refine=bool(args.get("refine", False)), ab_head=bool(args.get("ab_head", False)), rest2=bool(args.get("rest2", False)),
                  rest2_raw=bool(args.get("rest2_raw", False)), rest2_raw2=bool(args.get("rest2_raw2", False)), rest3=bool(args.get("rest3", False)),
                  rest3_half=bool(args.get("rest3_half", False)), seg_cascade=bool(args.get("seg_cascade", False)), seg_half=bool(args.get("seg_half", False)),
                  ms_half=bool(args.get("ms_half", False)))
    model.load_state_dict(ck["ema"])
    names = ["bg", "coral", "fish", "elasmo", "other"]
    with torch.no_grad():
        for kv in SEG_BIAS.split(","):
            k, v = kv.split(":")
            model.head_seg[-1].bias[names.index(k)] += float(v)
    basic_acts(model)
    model.set_sharpen(*SHARPEN)
    model.sharpen_port = SHARPEN_PORT

    model = model.eval().cpu()
    os.makedirs(os.path.dirname(os.path.abspath(out_onnx)), exist_ok=True)
    x = torch.rand(1, 3, *HW)
    prog = torch.onnx.export(model, (x,), dynamo=True, input_names=["input"], output_names=["restored_image", "seg_logits"], opset_version=18)
    prog.save(out_onnx, external_data=False)
    import onnxslim
    m = onnxslim.slim(onnx.load(out_onnx))
    onnx.save_model(m, out_onnx, save_as_external_data=False)
    m = onnx.load(out_onnx)
    if _dedup_initializers(m):
        onnx.save_model(m, out_onnx, save_as_external_data=False)
    return out_onnx


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "weights", sys.argv[2] if len(sys.argv) > 2 else "exported.onnx")
