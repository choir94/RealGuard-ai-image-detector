# Model Evidence

This document records the pinned model, its provenance, and the independent
evidence gathered to support its use in the AI Image Detector extension.

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

| Parameter | Default | Description |
|---|---|---|
| `a` (scale) | 1.0 | Platt scaling slope |
| `b` (bias) | 0.0 | Platt scaling intercept |

By default the calibration is **identity** (`a=1, b=0`): the raw sigmoid
probability is passed through unchanged. A **Platt scaling slot** is
available so that `p = sigmoid(a * logit + b)` can be applied if a
calibrated `(a, b)` pair is measured on a held-out set.

---

## Independent Browser Test Results

A small independent test was performed in a real browser environment using
the bundled ONNX Runtime Web inference path — the same code path the
extension uses at runtime.

| # | Source | Type | Expected | Predicted | Score |
|---|---|---|---|---|---|
| 1 | picsum.photos | Real photo | REAL | REAL | 98% real |
| 2 | picsum.photos | Real photo | REAL | REAL | 100% real |
| 3 | pollinations.ai | AI-generated | AI | AI | 100% fake |
| 4 | pollinations.ai | AI-generated | AI | AI | 100% fake |

**Summary:** 4 / 4 correct.

- Real photos: 2 / 2 correct (recall 1.0)
- AI images: 2 / 2 correct (recall 1.0)
- Balanced accuracy: 1.0

### Test Sources

- **picsum.photos** — Lorem Picsum, serves real photographs.
- **pollinations.ai** — Pollinations.AI, serves diffusion-generated images.

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

4. **Calibration drift.** The default identity calibration may produce
   over-confident or under-confident probabilities on distributions that
   differ from the training set. The Platt scaling slot exists for this
   reason but is not populated by default.

5. **No semantic understanding.** The model detects low-level forensic
   artifacts. It does not reason about scene content, physics, or context.
   A semantically implausible but forensically clean image may be
   classified as REAL.

6. **Adversarial perturbations.** Post-processing (JPEG, resize, blur,
   noise injection) can degrade or destroy the forensic artifacts the model
   relies on. Heavily re-encoded images may be misclassified.

7. **Small evaluation sample.** The independent browser test used only 4
   images. It validates the end-to-end pipeline, not statistical
   generalization.

8. **Single-model risk.** This extension pins a single model. An ensemble
   would improve robustness but is out of scope for the current build.
