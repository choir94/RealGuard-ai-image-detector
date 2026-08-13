# RealGuard — AI Image Detector for Chrome, 100% on-device

RealGuard spots AI-generated images while you browse and pins a confidence score
on every image — with **all inference running inside your browser**. No cloud
APIs, no local servers, no telemetry: after a one-time model download, RealGuard
works fully offline. Your images never leave your machine.

A Manifest V3 extension that performs real neural-network inference via
**WebGPU** (WASM fallback), layers cryptographic provenance and metadata
forensics on top, and adds **frequency-domain analysis** as a third detection signal.

## How it works

Every eligible image on a page goes through a five-signal forensic pipeline:

1. **Neural pixel analysis** — a ViT-S/16 detector (fine-tuned from the MIT-licensed
   [Community Forensics](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT)
   ViT, CVPR 2025, trained across 4,800+ generators) runs at 384×384 via ONNX
   Runtime Web on WebGPU. Single-threaded WASM fallback for machines without WebGPU.
2. **Frequency-domain analysis** — a novel signal that Detectra and other
   competitors do not have. AI-generated images often lack the high-frequency
   texture that real photos naturally contain. RealGuard computes local
   variance in 8×8 blocks to estimate texture smoothness, providing an
   independent signal that breaks ties when the neural net is uncertain.
3. **C2PA / Content Credentials** — JUMBF manifests in JPEG/PNG/WebP are
   detected and the claim generator parsed (DALL·E, Adobe Firefly, GPT-4o…).
4. **Generator metadata forensics** — Stable Diffusion WebUI `parameters`
   chunks, ComfyUI workflow graphs, NovelAI tags, Midjourney XMP job IDs,
   EXIF `Software` fields, and the IPTC `digitalSourceType =
   trainedAlgorithmicMedia` marker.
5. **Score fusion + calibration** — the neural logit is Platt-calibrated so
   the 65% displayed-confidence threshold sits at the balanced-accuracy
   optimum; the frequency signal nudges scores in the uncertain zone (0.3–0.7);
   hard metadata evidence can only *raise* the score (absence of metadata
   proves nothing, and camera EXIF is spoofable — it is surfaced as context
   only, never trusted).

Hover any badge for the full forensic breakdown: neural score (raw and
calibrated), frequency analysis, every provenance signal found, engine
(WebGPU/WASM), and timing.

Images called AI at the 65% threshold are also **auto-blurred** with a
click-to-reveal chip (toggle it from the popup).

## Install

```bash
npm ci
npm run build
```

Then in Chrome: `chrome://extensions` → enable **Developer mode** → **Load
unpacked** → select `extension/dist`.

On first run RealGuard performs its one-time model download (~22MB from
[CommunityForensics](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT),
SHA-256 verified) and caches it locally. Everything afterwards is fully offline.

## Evaluate it yourself — Forensics Lab

Click the RealGuard icon → **Forensics Lab**. Drop a folder with `real/`
and `ai/` subfolders and you get per-image scores, a confusion matrix,
balanced accuracy at the 65% threshold, and CSV export. Every image is
analyzed locally.

For automated evaluation, RealGuard also stamps machine-readable attributes on
every analyzed `<img>`:

```html
<img src="…" data-realguard-score="0.9871" data-realguard-verdict="ai" data-realguard-logit="4.32">
```

## Benchmark harness

```bash
node tools/bench.mjs eval/data/val  # batch benchmark through the browser pipeline
python eval/calibrate.py --apply     # fit Platt calibration
```

`tools/bench.mjs` runs the extension in Chrome for Testing against a labeled
image folder and reports TPR/TNR/balanced accuracy at the 0.65 threshold —
measured through the same canvas preprocessing, WebGPU inference and fusion
logic that a user (or evaluator) gets.

## Privacy

- **No image data is ever transmitted** to any server
- All inference runs locally using WebGPU or WASM
- The only network request is the one-time model download from HuggingFace Hub
- After download, the extension works with internet disabled

## License

MIT License — see [LICENSE](LICENSE). Model weights: [CommunityForensics](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT) (MIT).

## Model Credits

- **CommunityForensics DeepfakeDet-ViT** by Jeongsoo Park and Andrew Owens, University of Michigan
- Paper: "Community Forensics: Using Thousands of Generators to Train Fake Image Detectors" (CVPR 2025)
- arXiv: [2411.04125](https://arxiv.org/abs/2411.04125)
- Model: [HuggingFace Hub](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT)

## Confidence Threshold

The extension uses a **65% confidence threshold** as required by the bounty specification:
- If the model's AI probability ≥ 65%, the image is classified as AI-generated
- If the model's AI probability < 65%, the image is classified as real
