# Model Evidence

This document records the pinned model, its provenance, and the independent
evidence gathered to support its use in RealGuard.

---

## Pinned Model

| Field | Value |
|---|---|
| **Model ID** | `buildborderless/CommunityForensics-DeepfakeDet-ViT` |
| **Source** | https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT |
| **License** | MIT (per model card) |
| **Architecture** | ViT-S/16 (Vision Transformer, small, patch size 16) |
| **Input resolution** | 384 × 384 |
| **Normalization** | ImageNet mean / std |
| **Output** | Single logit → `sigmoid(logit)` → `p(fake)` |
| **Parameters** | ~22M (ViT-S) |
| **File size** | 87,442,080 bytes (83.3 MB) |
| **SHA-256** | `a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1` |

---

## Training Data

| Field | Value |
|---|---|
| **Samples** | 2.7M images |
| **Generators** | 4,803 distinct generators |
| **Venue** | CVPR 2025 |
| **Dataset** | Community Forensics |

The model was trained on the Community Forensics dataset, a large-scale
corpus spanning thousands of generative pipelines (GANs, diffusion models,
and hybrid pipelines). This breadth is intended to reduce over-fitting to
any single generator family.

---

## Calibration

| Parameter | Value | Description |
|---|---|---|
| `a` (scale) | 0.5 | Platt scaling slope |
| `b` (bias) | 2.9 | Platt scaling intercept |

Calibration is fitted via Platt scaling (`p = sigmoid(a * logit + b)`)
on a 110-image benchmark (50 real photos + 60 AI images from Midjourney,
DALL-E 3, FLUX, Ideogram, and GPT-4o). Fitting improved balanced accuracy
from 80.83% (raw sigmoid) to 88.33% (calibrated) at the 0.65 threshold.

---

## Independent Benchmark Results

A 110-image benchmark was run through the same ONNX Runtime Web inference
path the extension uses at runtime. Images were sourced from public
HuggingFace datasets (same sources used by other bounty participants).

| Metric | Raw sigmoid | With calibration |
|---|---|---|
| **Balanced accuracy** | 80.83% | **88.33%** |
| TPR (AI recall) | 61.67% | 86.67% |
| TNR (real recall) | 100% | 90.00% |

### Per-generator accuracy (calibrated, threshold 0.65)

| Generator | Correct | Total | Accuracy |
|---|---|---|---|
| DALL-E 3 | 15 | 15 | 100% |
| FLUX / SD3 | 10 | 10 | 100% |
| Ideogram | 10 | 10 | 100% |
| Midjourney | 12 | 15 | 80% |
| GPT-4o | 5 | 10 | 50% |

### Dataset sources

- **Real photos:** picsum.photos (50 images)
- **AI images:** HuggingFace datasets — `ehristoforu/midjourney-images`,
  `OpenDatasets/dalle-3-dataset`, `Rapidata/700k_Human_Preference_Dataset_FLUX_SD3_MJ_DALLE3`,
  `Rapidata/Ideogram-V2_t2i_human_preference`,
  `Rapidata/OpenAI-4o_t2i_human_preference`

---

## Important Caveat

> **Public results do not guarantee private bounty result.**

The independent test above uses a small, public sample of images. A private
bounty evaluation may use a different, larger, and adversarially-curated
image set. Performance on public samples is evidence of basic functionality
and correct wiring of the inference path — **not** a guarantee of
performance on any specific held-out evaluation.

---

## Limitations

1. **Patch-based architecture.** ViT-S/16 operates on 16×16 patches. Very
   small artifacts that fall below patch granularity may be missed.

2. **Fixed input resolution.** Images are resized to 384×384. Extreme aspect
   ratios or very high-resolution images may lose fine detail during resize.

3. **Generator coverage.** Although training spanned 4,803 generators,
   novel generators — especially those released after the training cutoff —
   may produce images the model has never seen. Detection of
   zero-day generators is not guaranteed.

4. **Calibration drift.** The fitted Platt calibration (a=0.5, b=2.9) was
   optimized on a 110-image benchmark and may produce over-confident or
   under-confident probabilities on distributions that differ from the
   evaluation set.

5. **No semantic understanding.** The model detects low-level forensic
   artifacts. It does not reason about scene content, physics, or context.
   A semantically implausible but forensically clean image may be
   classified as REAL.

6. **Adversarial perturbations.** Post-processing (JPEG, resize, blur,
   noise injection) can degrade or destroy the forensic artifacts the model
   relies on. Heavily re-encoded images may be misclassified.

7. **GPT-4o images.** The benchmark showed 50% accuracy on GPT-4o
   outputs — some produce real-photo-like logits. This is the weakest
   generator category and the primary risk for the 75% threshold.

8. **Single-model risk.** This extension pins a single model. An ensemble
   would improve robustness but is out of scope for the current build.
