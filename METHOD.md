# Method description

Team: APC_TDC   Track(s): 2 (this repository; Track 1 has its own repository)   Reported testing-phase submission: #19367 (`EURS_t2_sub013`)

## Approach (one page at most)
One ONNX model (static input 1×3×768×1024 in [0, 1]; 11.32 GFLOPs, 0.992 M parameters) produces both outputs.

- **Segmentation student.** A MobileNetV3-Large encoder (ImageNet-pretrained, timm `mobilenetv3_large_100.ra_in1k`, features at 1/4, 1/8, 1/16) on a per-image, per-channel standardised input, with a light FPN-style decoder and a 5-class head. It was **distilled from a large teacher**: frozen DINOv3 ViT-L/16 (LVD-1689M) with rank-8 LoRA and an FPN decoder, trained on the EURS training scenes plus external data. The student was trained with:
  - hard labels on the EURS training scenes;
  - the teacher's dense soft labels (KL on 1/4-resolution softmax) on external images and on the development-split images;
  - a feature-distillation loss (1×1 projection of the student's 1/16 features to frozen DINOv3 patch features of the clean image, cosine; the projection exists only in training and is not exported).
- **Restoration branch ("rest3").** An independent encoder on the un-normalised image (pixel-unshuffle, inverted-residual stages, FPN back to 1/4) with a small semantic side input from the frozen segmentation features. It predicts:
  - a soft water-type distribution, which weights a table of 3×4 inverse colour transforms;
  - per-pixel transmission / backscatter maps and a per-pixel affine correction, mixed by learned gates;
  - a 1/2-resolution detail residual.

  It was trained after the segmentation network was frozen.
- **Inside the ONNX at the output** (no learned parameters, no extra Conv):
  - luminance unsharp masking whose gain is interpolated from the predicted water type (0.25 → 3), with a fixed gain of 3 when the stretched input is detected as a portrait image;
  - per-class offsets added to the segmentation logits (fish +0.5, elasmobranch +1.5).
- **Export details.** HardSwish / HardSigmoid are rewritten as Mul / Add / Clip (numerically identical) for the Jetson.

## Training data
- **EURS training split:** all 20 training scenes (all 200 degradations each, GT images and GT masks). The water type and level in the training file names (`_<type>_<level>`) supervise the water-type head and the restoration targets. For model selection we used a held-out fold of 5 training scenes; the final model was retrained on all 20 scenes.
- **EURS development split:** the 1,200 degraded development images with **teacher pseudo-labels only** (no ground truth exists for them). The testing-phase images were **never** used for training, pseudo-labelling or model selection. Their only use: one image per test scene (`_1_5`) was hashed (256-bit dHash) to check that no external training image was a near-duplicate of a test scene (nearest distance 50/256, threshold 24, so nothing was removed).
- **External data:** SUIM, UIIS10K, USIS10K, USIS16K and Coralscapes (sources and licences are listed in our Track 1 repository's `METHOD.md`). Their masks were mapped to the 5 EURS classes (unclear classes ignored). Unlabelled pixels were filled with confident predictions (≥ 0.7) of an earlier teacher (ConvNeXt-L, DINOv3-pretrained) trained on 15 EURS training scenes only. The clean images were re-degraded with an image-formation model to create restoration pairs.
- **Pretrained models:**
  - ImageNet MobileNetV3-Large (timm `mobilenetv3_large_100.ra_in1k`, Apache-2.0).
  - DINOv3 ViT-L/16 LVD-1689M (timm `vit_large_patch16_dinov3.lvd1689m`, DINOv3 licence), used for the teacher and for feature distillation in training only; it is not part of the exported model.
  - ConvNeXt-L DINOv3 LVD-1689M (timm `convnext_large.dinov3_lvd1689m`, DINOv3 License): backbone of the earlier teacher that filled unlabelled pixels of the external images (data preparation only).
- **Global values chosen from testing-phase leaderboard scores:** the two segmentation logit offsets were calibrated on held-out training scenes and then kept after comparing our submissions' leaderboard scores (fixed values, identical for every image). Nothing else.

## Anything non-standard at inference
None outside the ONNX graph. The unsharp masking and the logit offsets described above are part of the submitted graph.

## Reproduction notes
- `export_onnx.py` → `main(model_dir, out_onnx)` loads `weights/best.pt` (EMA weights) and regenerates `onnx/submitted_model.onnx` (SHA-256 `dfb217bc31ea9e4e7d9b9480bb0364569e1042253ba1b8e454c34b6d053003e5`). It runs on the CPU in a few seconds.
  - On macOS (Apple silicon), where the leaderboard file was exported, the result is **byte-identical**.
  - In the root `Dockerfile`'s image on Linux x86-64, without network, the graph and weights are the same: only the `producer_version` header (`2.14.1+cpu`) and the order of the attributes inside the Conv nodes differ. onnxruntime gives bit-identical outputs, and `repro_check.py --export-onnx` reports "same graph and weights".
- The root `Dockerfile` pins the export environment: Python 3.12, PyTorch 2.14.1 CPU wheel, and the exact onnx / onnxscript / onnxslim versions that produced the leaderboard file. A different onnxscript / onnxslim version can rename graph nodes.
- `weights/best.pt`: SHA-256 `a22ad8178279ec390f5253d2e95fdc251354ae42c9a625140d7cb44d053e7acc`.
