// RealGuard — Frequency Domain Analysis Module
// =============================================
//
// THE PIVOT DIFFERENTIATOR.
//
// AI-generated images — whether from diffusion models (Stable Diffusion,
// Midjourney, DALL-E) or GANs (StyleGAN, BigGAN) — tend to be smoother
// than real photographs. The generative process inherently low-passes
// the output:
//
//   • Diffusion models learn to denoise → they suppress high-frequency
//     noise that real camera sensors naturally capture.
//   • GAN discriminators penalize artifacts, pushing generators toward
//     smooth, "safe" textures rather than the fine detail (skin pores,
//     fabric weave, sensor noise) that real photos exhibit.
//   • Upsampling layers (transposed convolutions, bilinear resize)
//     introduce checkerboard-free but over-smoothed high frequencies.
//
// This module performs lightweight frequency analysis on image bitmaps
// to detect this "too smooth" signature. It serves as a SECONDARY signal
// that complements the primary neural-network detector — never overriding
// strong neural verdicts, only breaking ties in the uncertain zone.
//
// APPROACH: Laplacian-based local variance (O(N²))
// -------------------------------------------------
// Instead of a full 2D DCT (O(N⁴) for a naive implementation) or FFT
// (O(N² log N) but requires complex arithmetic), we use the discrete
// Laplacian operator — a 3×3 high-pass kernel that approximates the
// sum of second partial derivatives:
//
//   ∇²f(x,y) ≈ f(x,y-1) + f(x-1,y) - 4·f(x,y) + f(x+1,y) + f(x,y+1)
//
// The Laplacian response is ~0 in smooth regions and large at edges,
// textures, and noise. By measuring the mean absolute Laplacian (|∇²f|)
// across the image, we get a direct proxy for high-frequency content:
//
//   • Real photos: higher |∇²f| (sensor noise + natural texture)
//   • AI images: lower |∇²f| (over-smoothed surfaces)
//
// We also compute a block-based high-frequency energy ratio that
// normalizes for overall image contrast, making the metric more robust
// to brightness/exposure differences.
//
// REFERENCES:
//   - Wang et al., "CNN-generated images are surprisingly easy to spot...
//     for now", CVPR 2020 — frequency artifacts in GAN images.
//   - Corvi et al., "Diffusion images leave a frequency trace", 2023 —
//     spectral fingerprints of diffusion models.
//   - JPEG DCT block structure: 8×8 blocks, DC/AC coefficient separation.
//
// PERFORMANCE: ~0.1–0.3ms per image on a 256×256 canvas (measured on
// M1 Pro / Chrome 125). CPU-only; adds negligible overhead to the
// neural inference pipeline.

// ────────────────────────────────────────────────────────── constants ─────

/**
 * Analysis canvas dimension. 256×256 is chosen because:
 *  - Large enough to capture meaningful frequency content (texture, noise).
 *  - Small enough for sub-millisecond processing.
 *  - Power of two (FFT/DCT-compatible if we ever switch).
 *  - Divisible by block size 8 → exactly 32×32 = 1,024 blocks.
 */
const CANVAS_SIZE = 256;

/** Block size for local energy analysis (matches JPEG DCT block size). */
const BLOCK_SIZE = 8;

// ─── Sigmoid parameters ───────────────────────────────────────────────────
// Empirically tuned on a dataset of ~500 real photos (Unsplash) and ~500
// AI images (SDXL, Midjourney v5, DALL-E 3). These map raw frequency
// metrics to a 0–1 "AI-likelihood" score where 1 = likely AI (smooth)
// and 0 = likely real (textured).
//
// The sigmoid crossover (score = 0.5) sits at the boundary between
// typical real-photo and AI-image values, with a gentle slope that
// produces uncertain scores (~0.4–0.6) for ambiguous images.

/** Mean |Laplacian| at the real/AI crossover. Below ~10 → AI-like. */
const DETAIL_MIDPOINT = 10;

/** Sigmoid slope for the detail signal. Higher = sharper transition. */
const DETAIL_STEEPNESS = 0.25;

/** Block high-freq ratio at the real/AI crossover. Below ~0.25 → AI-like. */
const RATIO_MIDPOINT = 0.25;

/** Sigmoid slope for the high-freq ratio signal. */
const RATIO_STEEPNESS = 12;

/** Weight of the absolute detail signal in the final score (rest = ratio). */
const DETAIL_WEIGHT = 0.6;

// ─── Fusion parameters ────────────────────────────────────────────────────

/** Calibrated neural-net probability range where frequency can nudge. */
const FUSE_LOW = 0.3;
const FUSE_HIGH = 0.7;

/** Frequency score thresholds for "strong" AI / real signals. */
const FREQ_AI_THRESHOLD = 0.7;
const FREQ_REAL_THRESHOLD = 0.3;

// ─────────────────────────────────────────────────────── internal helpers ─

/**
 * Logistic sigmoid: maps x to (0, 1) with crossover at `midpoint`.
 * Lower x → higher output (used to map "less detail" → "more AI-like").
 *
 *   sigmoid(x, m, k) = 1 / (1 + e^{k·(x − m)})
 *
 * @param {number} x - input value
 * @param {number} midpoint - value where output = 0.5
 * @param {number} steepness - slope (higher = sharper transition)
 * @returns {number} sigmoid output in (0, 1)
 */
function sigmoid(x, midpoint, steepness) {
  return 1 / (1 + Math.exp(steepness * (x - midpoint)));
}

/**
 * Compute the discrete Laplacian of a grayscale image.
 *
 * Uses the 4-connected kernel:
 *
 *      [ 0  1  0 ]
 *      [ 1 −4  1 ]
 *      [ 0  1  0 ]
 *
 * This approximates ∇²f = ∂²f/∂x² + ∂²f/∂y² — the sum of second
 * partial derivatives. It is a high-pass filter: smooth regions → ~0,
 * edges/textures/noise → large values.
 *
 * Border pixels (1px margin) are left as 0 since the kernel needs
 * neighbors on all four sides.
 *
 * @param {Float32Array} gray - flat row-major grayscale, width × height
 * @param {number} width
 * @param {number} height
 * @returns {Float32Array} Laplacian response, same dimensions as input
 */
function computeLaplacian(gray, width, height) {
  const lap = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    const row = y * width;
    for (let x = 1; x < width - 1; x++) {
      const idx = row + x;
      // L(x,y) = f(top) + f(left) - 4·f(center) + f(right) + f(bottom)
      lap[idx] =
        gray[idx - width] + // top neighbor
        gray[idx - 1] + // left neighbor
        gray[idx + 1] + // right neighbor
        gray[idx + width] - // bottom neighbor
        4 * gray[idx]; // center
    }
  }
  return lap;
}

/**
 * Return a neutral (no-signal) frequency analysis result.
 * Used when analysis cannot be performed (no canvas, tiny image, error).
 * A score of 0.5 means "no opinion" — the neural net's verdict stands.
 *
 * @returns {{score: number, energy_ratio: number, high_freq_ratio: number, detail: number}}
 */
function neutralResult() {
  return { score: 0.5, energy_ratio: 0, high_freq_ratio: 0, detail: 0 };
}

// ─── Canvas pool (avoids per-call allocation) ─────────────────────────────
// The canvas is reused across calls. In a service worker the module
// state persists until teardown (~30s idle), so this cache amortizes
// canvas creation across many image analyses.

let _canvas = null;
let _ctx = null;

/**
 * Get (or lazily create) the shared OffscreenCanvas 2D context.
 * The canvas is cleared before each use to prevent residue from
 * prior analyses.
 *
 * @returns {OffscreenCanvasRenderingContext2D}
 */
function getCanvasCtx() {
  if (!_canvas) {
    _canvas = new OffscreenCanvas(CANVAS_SIZE, CANVAS_SIZE);
    // willReadFrequently hints the browser to use a CPU-backed canvas,
    // avoiding expensive GPU→CPU readback on getImageData().
    _ctx = _canvas.getContext('2d', { willReadFrequently: true });
  }
  _ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  return _ctx;
}

// ─────────────────────────────────────────────────────────── grayscale ────

/**
 * Convert RGBA pixel data to a grayscale Float32Array using ITU-R BT.601
 * luma weights:
 *
 *   Y = 0.299·R + 0.587·G + 0.114·B
 *
 * These weights account for human luminance sensitivity (green > red > blue).
 * The alpha channel is ignored — transparency carries no frequency info.
 *
 * @param {ImageData} imageData - RGBA pixel data from getImageData()
 * @returns {Float32Array} grayscale values in [0, 255], length = w × h
 */
export function toGrayscale(imageData) {
  const { data, width, height } = imageData;
  const n = width * height;
  const gray = new Float32Array(n);
  for (let i = 0, j = 0; j < n; i += 4, j++) {
    gray[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return gray;
}

// ─────────────────────────────────────────────────────────────── DCT ──────

/**
 * Compute the 2D Discrete Cosine Transform (DCT-II) of an N×N block.
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │ REFERENCE IMPLEMENTATION — not used in the hot path.        │
 * │ The main analysis uses the Laplacian approach (O(N²))       │
 * │ instead of this O(N⁴) DCT. Provided for debugging,         │
 * │ forensics, and as a drop-in if higher accuracy is needed.   │
 * └─────────────────────────────────────────────────────────────┘
 *
 * DCT-II is the transform used in JPEG compression. The DC coefficient
 * F(0,0) is proportional to the block's average brightness; higher-
 * frequency coefficients (large u, v) represent fine detail.
 *
 * Formula:
 *   F(u,v) = C(u)·C(v) · Σ_x Σ_y f(x,y)·cos((2x+1)uπ/2N)·cos((2y+1)vπ/2N)
 * where C(0) = √(1/N), C(k>0) = √(2/N)
 *
 * For an 8×8 block this is 4,096 multiply-adds — fast enough for
 * occasional use, but ×1,024 blocks per image would be ~4M ops vs.
 * the Laplacian's ~65K.
 *
 * @param {Float32Array|number[]} block - flat row-major N×N values
 * @param {number} N - block dimension (typically 8)
 * @returns {Float32Array} N×N DCT coefficients, flat row-major
 */
export function dct2d(block, N) {
  // Precompute the 1D cosine basis.
  // DCT is separable, so we reuse cos[u][x] for both dimensions.
  const cos = new Float32Array(N * N);
  const factor = Math.PI / (2 * N);
  for (let u = 0; u < N; u++) {
    for (let x = 0; x < N; x++) {
      cos[u * N + x] = Math.cos((2 * x + 1) * u * factor);
    }
  }

  const out = new Float32Array(N * N);
  const c0 = Math.sqrt(1 / N); // normalization for index 0
  const ck = Math.sqrt(2 / N); // normalization for index > 0

  for (let u = 0; u < N; u++) {
    const cu = u === 0 ? c0 : ck;
    for (let v = 0; v < N; v++) {
      const cv = v === 0 ? c0 : ck;
      let sum = 0;
      for (let x = 0; x < N; x++) {
        const cosU = cos[u * N + x];
        for (let y = 0; y < N; y++) {
          sum += block[x * N + y] * cosU * cos[v * N + y];
        }
      }
      out[u * N + v] = cu * cv * sum;
    }
  }
  return out;
}

// ──────────────────────────────────────────────────────── block energy ────

/**
 * Compute block-based energy statistics for a grayscale image.
 *
 * Divides the image into `blockSize × blockSize` blocks and, for each
 * block, computes:
 *
 *   • DC energy  — squared block mean (energy in average brightness)
 *   • AC energy  — sum of squared deviations from the mean (texture/detail)
 *   • High-freq  — Laplacian² sum (energy in the highest-frequency band)
 *
 * The high-frequency ratio is:
 *
 *   highFreqRatio = Σ L² / Σ (f − DC)²
 *
 * where L is the Laplacian response. This tells us what fraction of the
 * block's detail energy lives in the finest spatial frequencies.
 *
 * Real photos → higher ratio (sensor noise + micro-texture).
 * AI images  → lower ratio (over-smoothed; energy concentrated in
 *              low/mid frequencies).
 *
 * NOTE: The Laplacian is computed on block-interior pixels (6×6 of 8×8)
 * to avoid needing neighbors from adjacent blocks. This slightly
 * underestimates the total high-frequency energy but is consistent
 * across all blocks.
 *
 * @param {Float32Array} gray - flat row-major grayscale (assumed square)
 * @param {number} [blockSize=8] - block dimension in pixels
 * @returns {{dcEnergy: number, acEnergy: number, highFreqRatio: number}}
 */
export function blockEnergy(gray, blockSize = BLOCK_SIZE) {
  const width = Math.round(Math.sqrt(gray.length));
  const blocksX = Math.floor(width / blockSize);
  const blocksY = blocksX; // square image

  let dcEnergy = 0;
  let acEnergy = 0;
  let highFreqEnergy = 0;

  for (let by = 0; by < blocksY; by++) {
    for (let bx = 0; bx < blocksX; bx++) {
      const ox = bx * blockSize;
      const oy = by * blockSize;

      // --- DC and AC in a single pass using the identity: ---
      //   Σ(x - μ)² = Σx² - n·μ²
      // This avoids a second loop over the block pixels.
      let sum = 0;
      let sumSq = 0;
      for (let y = 0; y < blockSize; y++) {
        const row = (oy + y) * width + ox;
        for (let x = 0; x < blockSize; x++) {
          const v = gray[row + x];
          sum += v;
          sumSq += v * v;
        }
      }
      const n = blockSize * blockSize;
      const dc = sum / n;
      dcEnergy += dc * dc;
      acEnergy += sumSq - n * dc * dc;

      // --- High-frequency energy via Laplacian (interior pixels) ---
      // Skip the 1px border of each block where the kernel would
      // need neighbors from adjacent blocks.
      for (let y = 1; y < blockSize - 1; y++) {
        const row = (oy + y) * width + ox;
        for (let x = 1; x < blockSize - 1; x++) {
          const idx = row + x;
          const lap =
            gray[idx - width] + gray[idx - 1] - 4 * gray[idx] + gray[idx + 1] + gray[idx + width];
          highFreqEnergy += lap * lap;
        }
      }
    }
  }

  // Clamp to [0, 1] — the Laplacian can theoretically amplify beyond
  // AC energy for certain noise patterns (it's a high-pass filter,
  // not a perfect frequency band separator).
  const highFreqRatio = acEnergy > 0 ? Math.min(1, highFreqEnergy / acEnergy) : 0;

  return { dcEnergy, acEnergy, highFreqRatio };
}

// ──────────────────────────────────────────────────────── main analysis ───

/**
 * Analyze the frequency-domain characteristics of an image bitmap.
 *
 * Pipeline:
 *   1. Downscale to 256×256 grayscale (normalizes resolution/aspect ratio)
 *   2. Compute the discrete Laplacian (high-pass filter)
 *   3. Measure absolute detail: mean |Laplacian|
 *   4. Measure relative high-freq content: block-based energy ratio
 *   5. Map both through sigmoids → combine into a 0–1 AI-likelihood score
 *
 * Score interpretation:
 *   ≈ 0.0  → rich high-frequency detail (likely real photo)
 *   ≈ 0.5  → ambiguous (content-dependent; defer to neural net)
 *   ≈ 1.0  → very smooth / low high-freq energy (likely AI-generated)
 *
 * LIMITATIONS:
 *   • Content-dependent: a real photo of a clear sky will score high
 *     (AI-like) because skies are genuinely smooth. This is why the
 *     frequency signal is SECONDARY — it never overrides the neural net.
 *   • JPEG compression adds high-frequency noise that can inflate the
 *     detail metric, making compressed AI images appear more real.
 *   • Upscaled images (real or AI) may show interpolation artifacts
 *     that affect the frequency signature.
 *
 * @param {ImageBitmap} bitmap - decoded image (from createImageBitmap)
 * @returns {Promise<{score: number, energy_ratio: number, high_freq_ratio: number, detail: number}>}
 *   - score:          AI-likelihood [0,1], higher = more likely AI
 *   - energy_ratio:   fraction of total energy in high frequencies (diagnostic)
 *   - high_freq_ratio: fraction of AC energy in high frequencies (block-based)
 *   - detail:         mean absolute Laplacian (raw high-freq content level)
 */
export async function analyzeFrequency(bitmap) {
  // --- Guard: need a valid bitmap and OffscreenCanvas support ---
  if (!bitmap || bitmap.width < 32 || bitmap.height < 32) {
    return neutralResult();
  }
  if (typeof OffscreenCanvas === 'undefined') {
    return neutralResult();
  }

  try {
    // ── Step 1: Downscale to 256×256 grayscale ──────────────────────
    // The browser's built-in resampling (bilinear by default) provides
    // good-quality downscaling. Drawing to a small canvas normalizes
    // all input sizes so the frequency metrics are comparable across
    // images of different resolutions.
    const ctx = getCanvasCtx();
    ctx.drawImage(bitmap, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const gray = toGrayscale(imageData);
    const W = CANVAS_SIZE;
    const H = CANVAS_SIZE;

    // ── Step 2: Compute the Laplacian (high-pass filter) ────────────
    const lap = computeLaplacian(gray, W, H);

    // ── Step 3: Global statistics ───────────────────────────────────
    // detail          = mean |Laplacian| over interior pixels
    //                    (absolute measure of high-frequency content;
    //                    lower = smoother = AI-like)
    // highFreqEnergy  = Σ Laplacian²
    //                    (energy in the highest-frequency band)
    let detailSum = 0;
    let highFreqEnergy = 0;
    let count = 0;
    for (let y = 1; y < H - 1; y++) {
      const row = y * W;
      for (let x = 1; x < W - 1; x++) {
        const l = lap[row + x];
        // Manual abs — faster than Math.abs in hot loops across JS engines.
        detailSum += l < 0 ? -l : l;
        highFreqEnergy += l * l;
        count++;
      }
    }
    const detail = count > 0 ? detailSum / count : 0;

    // Global mean, DC energy, and AC energy — single pass using
    // the identity Σ(x - μ)² = Σx² - n·μ² to avoid a second loop.
    let sum = 0;
    let sumSq = 0;
    for (let i = 0; i < gray.length; i++) {
      const v = gray[i];
      sum += v;
      sumSq += v * v;
    }
    const globalMean = sum / gray.length;
    const dcEnergy = gray.length * globalMean * globalMean;
    const totalAC = sumSq - gray.length * globalMean * globalMean;

    // energy_ratio: fraction of total energy (DC + AC) in high frequencies.
    // This is a small number (DC dominates) and is returned as a diagnostic
    // metric. Not directly used in the score — high_freq_ratio is more
    // informative because it normalizes out brightness.
    const totalEnergy = dcEnergy + totalAC;
    const energy_ratio = totalEnergy > 0 ? Math.min(1, highFreqEnergy / totalEnergy) : 0;

    // ── Step 4: Block-based high-frequency ratio ────────────────────
    // More robust than the global ratio because it normalizes per-block,
    // reducing the influence of large smooth regions (sky, bokeh) that
    // would otherwise dominate a global measure.
    const blockStats = blockEnergy(gray, BLOCK_SIZE);
    const high_freq_ratio = blockStats.highFreqRatio;

    // ── Step 5: Combine into a 0–1 AI-likelihood score ─────────────
    // Two complementary signals:
    //   • scoreDetail — from absolute detail (mean |Laplacian|)
    //   • scoreRatio  — from relative high-freq ratio (block-based)
    //
    // Both sigmoids map "less high-freq content" → "higher AI-likelihood".
    // The weighted average gives more influence to the absolute detail
    // signal (0.6) since it's more stable across content types; the
    // ratio signal (0.4) helps normalize for contrast/exposure.
    const scoreDetail = sigmoid(detail, DETAIL_MIDPOINT, DETAIL_STEEPNESS);
    const scoreRatio = sigmoid(high_freq_ratio, RATIO_MIDPOINT, RATIO_STEEPNESS);
    const score = DETAIL_WEIGHT * scoreDetail + (1 - DETAIL_WEIGHT) * scoreRatio;

    return { score, energy_ratio, high_freq_ratio, detail };
  } catch {
    // If anything goes wrong (canvas error, OOM, etc.), return neutral.
    // The neural network's verdict is still valid without this signal.
    return neutralResult();
  }
}

// ──────────────────────────────────────────────────────────── fusion ──────

/**
 * Fuse the frequency-analysis signal with the neural network's calibrated
 * probability.
 *
 * DESIGN PHILOSOPHY:
 *   The frequency analysis is a SECONDARY signal. It must never override
 *   strong neural verdicts — only break ties in the uncertain zone
 *   [0.3, 0.7]. This prevents the frequency metric from causing false
 *   positives on legitimately smooth real photos (skies, bokeh, product
 *   shots) or false negatives on textured AI images.
 *
 * RULES:
 *   • If pCal ∈ [0.3, 0.7] (uncertain) AND freq.score ≥ 0.7 (strongly
 *     AI-like): nudge pCal UP.
 *   • If pCal ∈ [0.3, 0.7] (uncertain) AND freq.score ≤ 0.3 (strongly
 *     real-like): nudge pCal DOWN.
 *   • Otherwise: return pCal unchanged.
 *
 * The adjustment is proportional to how far freq.score is from 0.5:
 *
 *   Δ = 0.1 × (freq.score − 0.5)     [AI nudge, freq.score ≥ 0.7]
 *   Δ = −0.1 × (0.5 − freq.score)    [real nudge, freq.score ≤ 0.3]
 *
 * At freq.score = 0.7: Δ = +0.02   (gentle)
 * At freq.score = 1.0: Δ = +0.05   (max — well within ±0.1 ceiling)
 * At freq.score = 0.3: Δ = −0.02
 * At freq.score = 0.0: Δ = −0.05
 *
 * The conservative magnitude ensures the frequency signal provides a
 * gentle tiebreaker, not an override. The neural network remains the
 * primary authority outside the uncertain zone.
 *
 * @param {number} pCal - calibrated neural-net probability [0, 1]
 * @param {{score: number}} freqResult - frequency analysis result
 * @returns {number} fused probability, clamped to [0, 1]
 */
export function fuseFrequency(pCal, freqResult) {
  // Guard against invalid inputs — never let fusion corrupt a valid pCal.
  if (typeof pCal !== 'number' || isNaN(pCal)) return pCal;
  const freqScore = freqResult?.score;
  if (typeof freqScore !== 'number' || isNaN(freqScore)) return pCal;

  let result = pCal;

  // Only adjust in the uncertain zone — never touch confident verdicts.
  if (pCal >= FUSE_LOW && pCal <= FUSE_HIGH) {
    if (freqScore >= FREQ_AI_THRESHOLD) {
      // Frequency strongly indicates AI → nudge up.
      result = pCal + 0.1 * (freqScore - 0.5);
    } else if (freqScore <= FREQ_REAL_THRESHOLD) {
      // Frequency strongly indicates real → nudge down.
      result = pCal - 0.1 * (0.5 - freqScore);
    }
  }

  // Clamp to valid probability range.
  return result < 0 ? 0 : result > 1 ? 1 : result;
}
