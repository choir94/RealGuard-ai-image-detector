/**
 * engine.js — ONNX Runtime Web inference engine for Chrome MV3 service worker.
 *
 * This module replaces a previous Transformers.js-based pipeline with direct
 * onnxruntime-web usage.  The `.bundle.min.mjs` build is imported statically
 * because MV3 service workers forbid dynamic `import()`.  The WASM binary is
 * fetched explicitly and injected via `ort.env.wasm.wasmBinary`, sidestepping
 * any runtime fetch of the `.wasm` glue that would violate the MV3 CSP.
 *
 * Pipeline (analyze):
 *   fetch bytes → scanMetadata → analyzeFrequency → decode bitmap → preprocess
 *   → serialized session.run → sigmoid(logit) → calibration → fuseSignals
 *   → fuseSignals → cached result
 *
 * Pipeline (explain):
 *   fetch bytes → decode bitmap → 1 base inference + 9 region inferences (3×3 grid)
 *
 * ── Pillow-matching bilinear resize ────────────────────────────────────────
 * Browser canvas `drawImage` uses a vendor-specific downsampling filter that
 * differs materially from the bilinear interpolation applied by Pillow's
 * `transforms.Resize` during model training.  To keep inference scores
 * faithful to the training distribution we implement a custom separable
 * bilinear resize on raw pixel data, replicating Pillow's filter support and
 * weight computation exactly.  See `bilinearCoefficients` and
 * `bilinearResizeRgb` below.
 */

// ────────────────────────────────────────────────────────────────────────────
// Imports
// ────────────────────────────────────────────────────────────────────────────

/**
 * The `.bundle` build embeds all JS glue statically — no dynamic import() or
 * runtime fetch of auxiliary scripts.  This is mandatory in MV3 service workers
 * where the Content Security Policy disallows `eval`, dynamic imports, and
 * remote script loading.
 */
import ort from '../../../node_modules/onnxruntime-web/dist/ort.bundle.min.mjs';

import { scanMetadata, fuseSignals } from './meta.js';
import { analyzeFrequency, fuseFrequency } from './freq.js';

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/** Cache Storage name for model artifacts (versioned for cache busting). */
const MODEL_CACHE = 'realguard-models-v1';

/** Hard ceiling for image fetches (40 MB).  Protects against memory blow-ups
 *  from accidentally huge inputs. */
const MAX_BYTES = 40 * 1024 * 1024;

/** LRU cache capacity for analysis results. */
const LRU_MAX = 1000;

/** Ring-buffer sizes for diagnostic logs and timing samples. */
const LOG_MAX = 50;
const TIMINGS_MAX = 50;
const DEBUG = false;

// ────────────────────────────────────────────────────────────────────────────
// State
// ────────────────────────────────────────────────────────────────────────────

/**
 * Central mutable state object.  Everything hangs off this singleton so that
 * the service worker (which may be killed and restarted by Chrome) can be
 * re-hydrated by calling `ensureSession()`.
 *
 * @type {Object}
 * @property {ort.InferenceSession|null} session    - The active ONNX session.
 * @property {'webgpu'|'wasm'}                     engine    - Execution provider in use.
 * @property {Object|null}                          model     - Parsed model config from models.json.
 * @property {{a:number,b:number}}                  calibration - Logistic calibration params (a*logit + b).
 * @property {'idle'|'loading'|'downloading'|'ready'|'error'} status - Lifecycle indicator.
 * @property {number}                               progress  - Download progress 0..1.
 * @property {string|null}                          error     - Last error message.
 * @property {Map<string, Object>}                  cache     - LRU result cache keyed by source URL.
 * @property {number[]}                             timings   - Recent inference durations (ms).
 * @property {string[]}                             log       - Ring buffer of diagnostic messages.
 */
const state = {
  session: null,
  engine: 'wasm',
  model: null,
  calibration: { a: 1, b: 0 },
  status: 'idle',
  progress: 0,
  error: null,
  cache: new Map(),
  timings: [],
  log: [],
};

// ────────────────────────────────────────────────────────────────────────────
// Logging
// ────────────────────────────────────────────────────────────────────────────

/**
 * Diagnostic logger.  Pushes a timestamped message into the ring buffer
 * (max {@link LOG_MAX} entries, oldest evicted first) and mirrors to
 * `console.log` for live debugging in DevTools.
 *
 * @param  {...any} args - Values to log (stringified if non-string).
 * @returns {void}
 */
function dlog(...args) {
  const ts = new Date().toISOString();
  const msg = args
    .map((a) => (typeof a === 'string' ? a : JSON.stringify(a)))
    .join(' ');
  state.log.push(`[${ts}] ${msg}`);
  if (state.log.length > LOG_MAX) state.log.shift();
  if (DEBUG) console.log('[engine]', ...args);
}

// ────────────────────────────────────────────────────────────────────────────
// Math utilities
// ────────────────────────────────────────────────────────────────────────────

/**
 * Numerically stable logistic sigmoid.
 *
 * @param {number} x - Raw logit.
 * @returns {number} Probability in (0, 1).
 */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Compute the SHA-256 digest of an `ArrayBuffer` and return it as a
 * lowercase hex string.  Uses the Web Crypto API (available in service workers).
 *
 * @param {ArrayBuffer} buf - Data to hash.
 * @returns {Promise<string>} Hex-encoded digest.
 */
async function sha256hex(buf) {
  const digest = await crypto.subtle.digest('SHA-256', buf);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

// ────────────────────────────────────────────────────────────────────────────
// Byte fetching
// ────────────────────────────────────────────────────────────────────────────

/**
 * Fetch a resource (image URL or data URL) as raw bytes.
 *
 * - `credentials: 'omit'` prevents sending cookies for cross-origin image
 *   fetches, which is both a privacy and correctness safeguard.
 * - Enforces the {@link MAX_BYTES} ceiling via `Content-Length` header (if
 *   present) and a post-read byte-count check.
 *
 * @param {string} src - URL or data URL to fetch.
 * @returns {Promise<Uint8Array>} Raw response bytes.
 * @throws {Error} On HTTP failure or size limit exceeded.
 */
async function fetchBytes(src) {
  // Use force-cache so images already loaded by the page are served from
  // browser HTTP cache even when network is disabled (bounty eval scenario).
  const res = await fetch(src, { credentials: 'omit', cache: 'force-cache' });
  if (!res.ok) {
    throw new Error(`fetchBytes: ${res.status} ${res.statusText} — ${src.slice(0, 120)}`);
  }

  // Pre-check via Content-Length if the server provides it.
  const contentLength = parseInt(res.headers.get('content-length') || '0', 10);
  if (contentLength && contentLength > MAX_BYTES) {
    throw new Error(
      `fetchBytes: resource exceeds ${MAX_BYTES} byte limit (${contentLength} bytes)`,
    );
  }

  const buf = await res.arrayBuffer();

  // Post-read check (handles chunked encoding where Content-Length is absent).
  if (buf.byteLength > MAX_BYTES) {
    throw new Error(
      `fetchBytes: resource exceeds ${MAX_BYTES} byte limit (${buf.byteLength} bytes)`,
    );
  }

  return new Uint8Array(buf);
}

// ────────────────────────────────────────────────────────────────────────────
// Manifest loading
// ────────────────────────────────────────────────────────────────────────────

/**
 * Fetch and parse `models.json` from the extension package, then populate
 * `state.model` with the first (default) entry.  Calibration parameters are
 * loaded from `chrome.storage.local` if the user (or a calibration UI) has
 * persisted them; otherwise the identity transform `{a:1, b:0}` is used.
 *
 * Expected model config shape:
 * ```
 * { id, name, url, sha256, bytes, devLocal, inputSize, resize, mean, std }
 * ```
 *
 * @returns {Promise<void>}
 * @throws {Error} If models.json cannot be fetched or parsed.
 */
export async function loadManifest() {
  const url = chrome.runtime.getURL('models.json');
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`loadManifest: failed to fetch models.json — ${res.status}`);
  }

  const manifest = await res.json();

  // Accept either a bare model object, an array, or { models: [...] }.
  const model = Array.isArray(manifest)
    ? manifest[0]
    : manifest.models
      ? manifest.models[0]
      : manifest;

  if (!model || !model.id) {
    throw new Error('loadManifest: no valid model entry in models.json');
  }

  state.model = model;

  // ── Calibration: manifest default → chrome.storage.local override ─────
  // The manifest may ship a calibration block; if so, use it as the default.
  // User-persisted calibration (from the options page) always wins.
  state.calibration = { a: 1, b: 0 };
  if (
    manifest.calibration &&
    typeof manifest.calibration.a === 'number' &&
    typeof manifest.calibration.b === 'number'
  ) {
    state.calibration = { a: manifest.calibration.a, b: manifest.calibration.b };
  }

  try {
    const stored = await chrome.storage.local.get('calibration');
    if (
      stored.calibration &&
      typeof stored.calibration.a === 'number' &&
      typeof stored.calibration.b === 'number'
    ) {
      state.calibration = { a: stored.calibration.a, b: stored.calibration.b };
      dlog(`Calibration from storage: a=${state.calibration.a}, b=${state.calibration.b}`);
    } else {
      dlog(`Calibration from manifest: a=${state.calibration.a}, b=${state.calibration.b}`);
    }
  } catch (e) {
    dlog(`No calibration in storage (${e.message}), using manifest/identity default`);
  }

  dlog(`Manifest loaded: ${model.id} — ${model.name}`);
}

// ────────────────────────────────────────────────────────────────────────────
// Model bytes: devLocal → Cache Storage → download
// ────────────────────────────────────────────────────────────────────────────

/**
 * Resolve the model's ONNX bytes using a three-tier fallback:
 *
 * 1. **devLocal** — A file bundled inside the extension (for development or
 *    offline-first distribution).  Trusted without hash verification.
 * 2. **Cache Storage** — Previously downloaded and verified model cached in
 *    the `realguard-models-v1` cache.  No re-verification (it was checked
 *    before caching).
 * 3. **Network download** — One-time fetch with streaming progress tracking,
 *    SHA-256 verification, and `cache.put()` for future loads.
 *
 * @returns {Promise<Uint8Array>} Raw ONNX model bytes.
 * @throws {Error} On download failure, hash mismatch, or size exceeded.
 */
async function getModelBytes() {
  const model = state.model;

  // ── Tier 1: devLocal (bundled with extension) ────────────────────────
  if (model.devLocal) {
    try {
      const devUrl = chrome.runtime.getURL(model.devLocal);
      const devRes = await fetch(devUrl);
      if (devRes.ok) {
        const buf = await devRes.arrayBuffer();
        dlog(`Model from devLocal: ${model.devLocal} (${buf.byteLength} bytes)`);
        return new Uint8Array(buf);
      }
      dlog(`devLocal fetch returned ${devRes.status}, falling through to cache`);
    } catch (e) {
      dlog(`devLocal fetch failed (${e.message}), falling through to cache`);
    }
  }

  // ── Tier 2: Cache Storage ─────────────────────────────────────────────
  try {
    const cache = await caches.open(MODEL_CACHE);
    const cached = await cache.match(model.url);
    if (cached) {
      const buf = await cached.arrayBuffer();
      dlog(`Model from Cache Storage (${buf.byteLength} bytes)`);
      return new Uint8Array(buf);
    }
  } catch (e) {
    dlog(`Cache Storage read failed (${e.message}), proceeding to download`);
  }

  // ── Tier 3: Network download with progress + SHA-256 ─────────────────
  dlog(`Downloading model from ${model.url}`);
  state.status = 'downloading';
  state.progress = 0;

  const res = await fetch(model.url, { credentials: 'omit' });
  if (!res.ok) {
    throw new Error(
      `Model download failed: ${res.status} ${res.statusText}`,
    );
  }

  const contentLength = parseInt(res.headers.get('content-length') || '0', 10);

  // Stream the response body for progress reporting.
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;

    // Update progress if we know the total size.
    if (contentLength > 0) {
      state.progress = Math.min(1, received / contentLength);
    }
  }

  // Assemble the complete buffer.
  const blob = new Blob(chunks);
  const buf = await blob.arrayBuffer();
  const u8 = new Uint8Array(buf);

  dlog(`Model downloaded: ${u8.byteLength} bytes`);

  // ── SHA-256 verification ──────────────────────────────────────────────
  // Skip if the manifest ships a placeholder hash (e.g. "TBD" during
  // development).  Only verify when a real 64-char hex digest is present.
  if (model.sha256 && model.sha256.length === 64 && /^[0-9a-f]{64}$/i.test(model.sha256)) {
    const hash = await sha256hex(buf);
    if (hash !== model.sha256) {
      throw new Error(
        `SHA-256 mismatch: expected ${model.sha256}, got ${hash}`,
      );
    }
    dlog('SHA-256 verified ✓');
  } else if (model.sha256) {
    dlog(`SHA-256 skipped: placeholder hash "${model.sha256}"`);
  }

  // ── Optional byte-count sanity check ──────────────────────────────────
  if (model.bytes && u8.byteLength !== model.bytes) {
    dlog(
      `Warning: byte count mismatch (expected ${model.bytes}, got ${u8.byteLength})`,
    );
  }

  // ── Persist to Cache Storage for subsequent loads ─────────────────────
  try {
    const cache = await caches.open(MODEL_CACHE);
    const responseToCache = new Response(blob, {
      headers: { 'content-type': 'application/octet-stream' },
    });
    await cache.put(model.url, responseToCache);
    dlog('Model cached in Cache Storage');
  } catch (e) {
    dlog(`Cache Storage write failed (${e.message}) — model will re-download next time`);
  }

  state.progress = 0;
  return u8;
}

// ────────────────────────────────────────────────────────────────────────────
// Session management
// ────────────────────────────────────────────────────────────────────────────

/** Single-flight guard: while a session is being created, concurrent callers
 *  await this promise instead of triggering duplicate work. */
let sessionPromise = null;

/** Cached WASM binary so session recreation doesn't re-fetch. */
let _wasmBinary = null;

/**
 * Fetch and cache the ORT WASM binary.  The `.jsep.wasm` variant supports
 * both the WASM and WebGPU (JSEP) execution providers.
 *
 * @returns {Promise<ArrayBuffer>} WASM binary.
 */
async function getWasmBinary() {
  if (_wasmBinary) return _wasmBinary;

  const wasmUrl = chrome.runtime.getURL(
    'vendor/ort/ort-wasm-simd-threaded.jsep.wasm',
  );
  dlog(`Loading WASM binary: ${wasmUrl}`);
  const res = await fetch(wasmUrl);
  if (!res.ok) {
    throw new Error(`WASM binary fetch failed: ${res.status}`);
  }
  _wasmBinary = await res.arrayBuffer();
  dlog(`WASM binary loaded (${_wasmBinary.byteLength} bytes)`);
  return _wasmBinary;
}

/**
 * Create the ONNX `InferenceSession`:
 *
 * 1. Ensure manifest is loaded (provides model config).
 * 2. Obtain model bytes (devLocal → cache → download).
 * 3. Inject the WASM binary via `ort.env.wasm.wasmBinary` (avoids dynamic
 *    fetch inside the service worker).
 * 4. Force `numThreads = 1` — MV3 service workers cannot spawn Web Workers.
 * 5. Try WebGPU first (with a GPU device we create explicitly for better
 *    error diagnostics); fall back to pure WASM.
 * 6. Warm up with a dummy tensor so the first real inference isn't slow.
 *
 * @returns {Promise<void>}
 * @throws {Error} On any failure in the above steps.
 */
async function createSession() {
  state.status = 'loading';
  dlog('Creating ONNX inference session…');

  // 1. Manifest
  if (!state.model) {
    await loadManifest();
  }
  const model = state.model;

  // 2. Model bytes
  const modelBytes = await getModelBytes();

  // 3. WASM binary injection
  const wasmBinary = await getWasmBinary();
  ort.env.wasm.wasmBinary = wasmBinary;

  // Also set wasmPaths so any auxiliary lookups resolve to the extension.
  ort.env.wasm.wasmPaths = chrome.runtime.getURL('vendor/ort/');

  // 4. Single-threaded WASM for offscreen documents and service workers.
  //    Both lack a Worker constructor available to ORT's threaded WASM build.
  if (typeof Worker === 'undefined') {
    ort.env.wasm.numThreads = 1;
    dlog('No Worker constructor: numThreads = 1');
  }

  const inputSize = model.inputSize;
  const sessionOpts = { graphOptimizationLevel: 'all' };

  let session = null;
  let engine = 'wasm';

  // 5a. Try WebGPU (offscreen documents have navigator.gpu; SWs may not)
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      if (adapter) {
        const device = await adapter.requestDevice();
        session = await ort.InferenceSession.create(modelBytes, {
          ...sessionOpts,
          executionProviders: [{ name: 'webgpu', device }],
        });
        engine = 'webgpu';
        dlog('Session created with WebGPU execution provider');
      } else {
        dlog('WebGPU adapter unavailable, falling back to WASM');
      }
    } catch (e) {
      dlog(`WebGPU init failed (${e.message}), falling back to WASM`);
      // WebGPU failure can corrupt ORT's internal WASM init state.
      // Reset the wasm proxy so the WASM fallback starts fresh.
      try {
        ort.env.wasm.proxy = false;
        ort.env.wasm.wasmBinary = await getWasmBinary();
      } catch (_) { /* ignore — already set */ }
      session = null;
    }
  }

  // 5b. Fallback to WASM
  if (!session) {
    session = await ort.InferenceSession.create(modelBytes, {
      ...sessionOpts,
      executionProviders: [{ name: 'wasm' }],
    });
    engine = 'wasm';
    dlog('Session created with WASM execution provider');
  }

  state.session = session;
  state.engine = engine;

  // 6. Warmup — prime the graph with a zero tensor so the JIT / graph
  //    optimisation runs now, not on the user's first image.
  const dummy = new ort.Tensor(
    'float32',
    new Float32Array(3 * inputSize * inputSize),
    [1, 3, inputSize, inputSize],
  );
  const inputName = session.inputNames[0];
  await session.run({ [inputName]: dummy });
  dlog('Warmup run complete');

  state.status = 'ready';
  state.error = null;
}

/**
 * Ensure an inference session exists, creating one if necessary.
 *
 * Uses a single-flight pattern: if `createSession()` is already in progress,
 * all concurrent callers share the same promise.  On failure the promise is
 * reset so the next call can retry.
 *
 * @returns {Promise<void>}
 */
export async function ensureSession() {
  if (state.session) return;

  if (!sessionPromise) {
    sessionPromise = createSession().catch((e) => {
      // Reset so the next ensureSession() call can attempt a fresh build.
      sessionPromise = null;
      state.status = 'error';
      state.error = e.message;
      throw e;
    });
  }

  return sessionPromise;
}

// ────────────────────────────────────────────────────────────────────────────
// Pillow-matching bilinear resize
// ────────────────────────────────────────────────────────────────────────────

/**
 * Per-output-pixel resize coefficients for one axis.
 *
 * @typedef {Object} ResizeCoefficients
 * @property {number} start     - First source pixel index that contributes.
 * @property {number[]} weights - Normalised weights for source pixels
 *   `[start, start+1, …, start+weights.length-1]`.
 */

/**
 * Compute bilinear resize coefficients for a single axis, exactly matching
 * Pillow's `transforms.Resize` (bilinear) filter behaviour.
 *
 * Pillow's algorithm:
 * 1. `scale = inputSize / outputSize` (the sampling stride).
 * 2. `filterScale = max(1, scale)` — the filter support widens when
 *    downsampling so that each output pixel averages multiple source pixels
 *    (anti-aliasing), but stays at 1 when upsampling (pure interpolation).
 * 3. For each output pixel `index`, the source-space centre is
 *    `center = (index + 0.5) * scale`.
 * 4. Source pixels within support distance `filterScale` of the centre
 *    contribute.  Weight = `max(0, 1 - |distance|)` where
 *    `distance = |source + 0.5 - center| / filterScale`.
 * 5. Weights are normalised to sum to 1.
 *
 * Reference: LocalLens `offscreen.ts` lines 56-115.
 *
 * @param {number} inputSize  - Number of source pixels along this axis.
 * @param {number} outputSize - Number of output pixels along this axis.
 * @returns {ResizeCoefficients[]} One entry per output pixel.
 */
function bilinearCoefficients(inputSize, outputSize) {
  const scale = inputSize / outputSize;
  const filterScale = Math.max(1, scale);
  const support = filterScale;
  return Array.from({ length: outputSize }, (_, index) => {
    const center = (index + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - support + 0.5));
    const end = Math.min(inputSize, Math.floor(center + support + 0.5));
    const weights = [];
    let total = 0;
    for (let source = start; source < end; source += 1) {
      const distance = Math.abs((source + 0.5 - center) / filterScale);
      const weight = Math.max(0, 1 - distance);
      weights.push(weight);
      total += weight;
    }
    return { start, weights: weights.map((weight) => weight / total) };
  });
}

/**
 * Resize an RGBA pixel buffer to a target size using a separable bilinear
 * filter that matches Pillow's `transforms.Resize` (bilinear) exactly.
 *
 * The operation is two-pass and separable:
 *   1. **Horizontal pass** — resize each row from `sourceWidth` to
 *      `targetWidth`, producing an intermediate buffer of size
 *      `targetWidth × sourceHeight` (still RGBA, 4 channels per pixel).
 *   2. **Vertical pass** — resize each column of the intermediate from
 *      `sourceHeight` to `targetHeight`, producing the final
 *      `targetWidth × targetHeight` buffer.
 *
 * Alpha is dropped — the returned buffer contains RGB only (3 bytes per
 * pixel, tightly packed).
 *
 * The canvas is used *only* to obtain raw RGBA pixel data from the bitmap;
 * all resizing is done manually in JS so that the filter kernel matches
 * Pillow's training transforms rather than the browser's vendor-specific
 * downsampling filter.
 *
 * @param {Uint8ClampedArray} sourceData   - Raw RGBA pixel data (4 bytes/pixel).
 * @param {number} sourceWidth             - Source image width in pixels.
 * @param {number} sourceHeight            - Source image height in pixels.
 * @param {number} targetWidth             - Desired output width.
 * @param {number} targetHeight            - Desired output height.
 * @returns {Uint8ClampedArray} Resized RGB pixel data (3 bytes/pixel, packed).
 */
function bilinearResizeRgb(sourceData, sourceWidth, sourceHeight, targetWidth, targetHeight) {
  // Pre-compute coefficients for both axes.
  const xCoeffs = bilinearCoefficients(sourceWidth, targetWidth);
  const yCoeffs = bilinearCoefficients(sourceHeight, targetHeight);

  // ── Pass 1: Horizontal resize (sourceWidth → targetWidth) ────────────
  // Intermediate buffer: targetWidth × sourceHeight, RGBA (4 channels).
  const intermediate = new Float64Array(targetWidth * sourceHeight * 4);

  for (let y = 0; y < sourceHeight; y++) {
    const srcRowOffset = y * sourceWidth * 4;
    const dstRowOffset = y * targetWidth * 4;

    for (let ox = 0; ox < targetWidth; ox++) {
      const { start, weights } = xCoeffs[ox];
      const numWeights = weights.length;
      const dstOffset = dstRowOffset + ox * 4;

      // Accumulate weighted R, G, B, A for this output column.
      let r = 0, g = 0, b = 0, a = 0;
      for (let wi = 0; wi < numWeights; wi++) {
        const srcX = start + wi;
        const w = weights[wi];
        const srcOffset = srcRowOffset + srcX * 4;
        r += sourceData[srcOffset] * w;
        g += sourceData[srcOffset + 1] * w;
        b += sourceData[srcOffset + 2] * w;
        a += sourceData[srcOffset + 3] * w;
      }

      intermediate[dstOffset] = r;
      intermediate[dstOffset + 1] = g;
      intermediate[dstOffset + 2] = b;
      intermediate[dstOffset + 3] = a;
    }
  }

  // ── Pass 2: Vertical resize (sourceHeight → targetHeight) ────────────
  // Final buffer: targetWidth × targetHeight, RGB (3 channels, no alpha).
  const output = new Uint8ClampedArray(targetWidth * targetHeight * 3);

  for (let oy = 0; oy < targetHeight; oy++) {
    const { start, weights } = yCoeffs[oy];
    const numWeights = weights.length;

    for (let ox = 0; ox < targetWidth; ox++) {
      let r = 0, g = 0, b = 0;
      for (let wi = 0; wi < numWeights; wi++) {
        const srcY = start + wi;
        const w = weights[wi];
        const srcOffset = (srcY * targetWidth + ox) * 4;
        r += intermediate[srcOffset] * w;
        g += intermediate[srcOffset + 1] * w;
        b += intermediate[srcOffset + 2] * w;
      }

      const dstOffset = (oy * targetWidth + ox) * 3;
      output[dstOffset] = r;
      output[dstOffset + 1] = g;
      output[dstOffset + 2] = b;
    }
  }

  return output;
}

// ────────────────────────────────────────────────────────────────────────────
// Image preprocessing
// ────────────────────────────────────────────────────────────────────────────

/**
 * Convert raw RGB pixel data (3 bytes/pixel, tightly packed — as produced by
 * {@link bilinearResizeRgb}) to a normalised CHW float32 tensor.
 *
 * Steps (mirrors PyTorch training transforms):
 *   1. Scale pixel to [0, 1]  (÷255)
 *   2. Normalise per-channel with ImageNet mean/std
 *   3. Rearrange from HWC (interleaved RGB) → CHW (planar)
 *
 * @param {Uint8ClampedArray} data - RGB pixel data (3 bytes/pixel, packed).
 * @param {[number,number,number]} mean - Per-channel mean (ImageNet default).
 * @param {[number,number,number]} std  - Per-channel std  (ImageNet default).
 * @param {number} crop - Spatial dimension (square: crop × crop).
 * @returns {ort.Tensor} Float32 tensor shaped [1, 3, crop, crop].
 */
function imageDataToTensor(data, mean, std, crop) {
  const size = crop * crop; // pixels per channel
  const out = new Float32Array(3 * size);
  const [m0, m1, m2] = mean;
  const [s0, s1, s2] = std;

  for (let i = 0; i < size; i++) {
    const o = i * 3; // RGB stride (no alpha)
    // Channel 0 (R)
    out[i] = (data[o] / 255 - m0) / s0;
    // Channel 1 (G)
    out[i + size] = (data[o + 1] / 255 - m1) / s1;
    // Channel 2 (B)
    out[i + size * 2] = (data[o + 2] / 255 - m2) / s2;
  }

  return new ort.Tensor('float32', out, [1, 3, crop, crop]);
}

/**
 * Full preprocessing pipeline for a decoded bitmap, mirroring the training
 * transforms:
 *
 *   Resize (shortest edge → model.resize, Pillow bilinear)
 *     → CenterCrop (model.inputSize × model.inputSize)
 *     → [0, 1] scale
 *     → ImageNet normalise
 *     → float32 CHW tensor
 *
 * The bitmap is drawn to a canvas at its **original** size and the raw RGBA
 * pixels are read via `getImageData`.  Resizing is then performed manually
 * with {@link bilinearResizeRgb} to match Pillow's bilinear filter exactly —
 * browser canvas `drawImage` uses a different downsampling kernel that would
 * materially change detector scores.
 *
 * @param {ImageBitmap} bitmap - Decoded source image.
 * @returns {ort.Tensor} Preprocessed input tensor.
 */
function preprocess(bitmap) {
  const model = state.model;
  const inputSize = model.inputSize; // e.g. 384
  const resize = model.resize; // e.g. 440 (shortest edge after resize)

  const w = bitmap.width;
  const h = bitmap.height;

  // ── Step 1: Draw bitmap at original size, get raw RGBA pixels ────────
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  const sourceImageData = ctx.getImageData(0, 0, w, h);
  const sourceData = sourceImageData.data;

  // ── Step 2: Compute resize dimensions (shortest edge → `resize`) ─────
  const scale = resize / Math.min(w, h);
  const rw = Math.max(inputSize, Math.round(w * scale));
  const rh = Math.max(inputSize, Math.round(h * scale));

  // ── Step 3: Custom bilinear resize (RGBA → RGB, Pillow-matching) ─────
  const resizedRgb = bilinearResizeRgb(sourceData, w, h, rw, rh);

  // ── Step 4: CenterCrop — extract inputSize × inputSize from centre ───
  const cx = Math.floor((rw - inputSize) / 2);
  const cy = Math.floor((rh - inputSize) / 2);

  const croppedRgb = new Uint8ClampedArray(inputSize * inputSize * 3);
  for (let y = 0; y < inputSize; y++) {
    const srcOffset = ((cy + y) * rw + cx) * 3;
    const dstOffset = y * inputSize * 3;
    croppedRgb.set(resizedRgb.subarray(srcOffset, srcOffset + inputSize * 3), dstOffset);
  }

  // ── Step 5: [0,1] → Normalise → CHW float32 tensor ──────────────────
  return imageDataToTensor(croppedRgb, model.mean, model.std, inputSize);
}

/**
 * Preprocess a square sub-region of a bitmap for heatmap analysis.
 *
 * Cuts the region `[rx, ry, rs×rs]` from the source bitmap (clamped to
 * bounds), then resizes it directly to `inputSize × inputSize` using the
 * custom Pillow-matching bilinear resize.
 *
 * The region is extracted from a canvas at the original source resolution
 * (raw RGBA via `getImageData`), then resized with
 * {@link bilinearResizeRgb} — **not** via canvas `drawImage`, which uses a
 * different downsampling kernel.
 *
 * @param {ImageBitmap} bitmap - Source image.
 * @param {number} rx - Region top-left X (source pixels).
 * @param {number} ry - Region top-left Y (source pixels).
 * @param {number} rs - Region side length (source pixels, square).
 * @returns {ort.Tensor} Preprocessed region tensor.
 */
function preprocessRegion(bitmap, rx, ry, rs) {
  const model = state.model;
  const inputSize = model.inputSize;

  // Clamp region to bitmap bounds.
  const sx = Math.max(0, Math.floor(rx));
  const sy = Math.max(0, Math.floor(ry));
  const sw = Math.min(Math.floor(rs), bitmap.width - sx);
  const sh = Math.min(Math.floor(rs), bitmap.height - sy);

  // ── Step 1: Extract region at original resolution, get raw RGBA ──────
  const canvas = new OffscreenCanvas(sw, sh);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  // Draw only the source sub-region to the canvas (1:1, no resize here).
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, sw, sh);
  const sourceImageData = ctx.getImageData(0, 0, sw, sh);
  const sourceData = sourceImageData.data;

  // ── Step 2: Custom bilinear resize (RGBA → RGB, Pillow-matching) ─────
  const resizedRgb = bilinearResizeRgb(sourceData, sw, sh, inputSize, inputSize);

  // ── Step 3: [0,1] → Normalise → CHW float32 tensor ──────────────────
  return imageDataToTensor(resizedRgb, model.mean, model.std, inputSize);
}

// ────────────────────────────────────────────────────────────────────────────
// Serialized inference queue
// ────────────────────────────────────────────────────────────────────────────

/**
 * ONNX Runtime Web sessions are not guaranteed re-entrant.  This promise chain
 * serialises all `session.run()` calls so that concurrent analyses (e.g.
 * multiple tabs sending messages simultaneously) do not interleave.
 *
 * Each enqueued task runs only after the previous one settles (success or
 * failure).  The chain itself never rejects — individual rejections propagate
 * to the caller while the chain stays healthy for the next task.
 *
 * @type {Promise<void>}
 */
let _inferChain = Promise.resolve();

/**
 * Enqueue a task in the serialized inference queue.
 *
 * @param {() => Promise<any>} fn - Task returning a promise (typically
 *        `() => session.run({ [inputName]: tensor })`).
 * @returns {Promise<any>} Resolves with `fn`'s result, or rejects with its error.
 */
function serialized(fn) {
  // `.then(fn, fn)` runs fn regardless of whether the previous promise
  // resolved or rejected, so one failed inference doesn't block the next.
  const p = _inferChain.then(fn, fn);
  // Swallow rejections on the internal chain so it always stays alive.
  _inferChain = p.then(
    () => {},
    () => {},
  );
  return p;
}

// ────────────────────────────────────────────────────────────────────────────
// analyze() — full image classification
// ────────────────────────────────────────────────────────────────────────────

/**
 * Run the full detection pipeline on a single image.
 *
 * Flow:
 *   1. LRU cache check — return immediately on hit.
 *   2. `ensureSession()` — lazy model load (single-flight).
 *   3. `fetchBytes()` — download image with size guard.
 *   4. `scanMetadata()` — extract EXIF / file-format signals.
 *   5. `analyzeFrequency()` — DCT / spectral analysis.
 *   6. `createImageBitmap()` — decode (no colour-space conversion).
 *   7. `preprocess()` — resize → center-crop → normalise.
 *   8. Serialized `session.run()` — get raw logit.
 *   9. `sigmoid(logit)` → pRaw (uncalibrated probability).
 *  10. `sigmoid(a*logit + b)` → pCal (calibrated probability).
 *  11. `fuseFrequency(pCal, freq)` → disabled (benchmark showed −1% accuracy);
 *      `pFused = pCal` instead.
 *  12. `fuseSignals(pFused, signals)` → blend metadata evidence.
 *  13. Cache and return.
 *
 * @param {Object} params
 * @param {string} [params.url]     - Image URL (HTTP/HTTPS).
 * @param {string} [params.dataUrl] - Data URL (base64-encoded image).
 * @returns {Promise<{ok:boolean, p?:number, pRaw?:number, logit?:number,
 *                     signals?:Object, freq?:Object, engine?:string,
 *                     ms?:number, model?:string, error?:string, cached?:boolean}>}
 */
export async function analyze({ url, dataUrl }) {
  const src = url || dataUrl;
  if (!src) {
    return { ok: false, error: 'No url or dataUrl provided' };
  }

  // ── 1. LRU cache check ───────────────────────────────────────────────
  if (state.cache.has(src)) {
    const cached = state.cache.get(src);
    // Move to end (most-recently-used).
    state.cache.delete(src);
    state.cache.set(src, cached);
    dlog(`Cache hit: ${src.slice(0, 80)}`);
    return { ...cached, cached: true };
  }

  try {
    // ── 2. Ensure session ──────────────────────────────────────────────
    await ensureSession();

    const t0 = performance.now();
    const session = state.session;
    const model = state.model;
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    // ── 3. Fetch image bytes ───────────────────────────────────────────
    const u8 = await fetchBytes(src);

    // ── 4. Metadata signals (EXIF, file header, etc.) ──────────────────
    const { signals } = scanMetadata(u8);

    // ── 5. Decode bitmap (no colour-space conversion for fidelity) ─────
    const blob = new Blob([u8]);
    const bitmap = await createImageBitmap(blob, {
      colorSpaceConversion: 'none',
      premultiplyAlpha: 'none',
    });

    // ── 6. Frequency-domain analysis (needs decoded bitmap) ────────────
    const freq = await analyzeFrequency(bitmap);

    try {
      // ── 7. Preprocess ────────────────────────────────────────────────
      const tensor = preprocess(bitmap);

      // ── 8. Serialized inference ──────────────────────────────────────
      const results = await serialized(() =>
        session.run({ [inputName]: tensor }),
      );
      const logit = results[outputName].data[0];

      const t1 = performance.now();
      const ms = Math.round(t1 - t0);

      // Record timing sample.
      state.timings.push(ms);
      if (state.timings.length > TIMINGS_MAX) state.timings.shift();

      // ── 9. Raw probability ───────────────────────────────────────────
      const pRaw = sigmoid(logit);

      // ── 10. Calibration: pCal = sigmoid(a * logit + b) ────────────────
      const { a, b } = state.calibration;
      const pCal = sigmoid(a * logit + b);

      // ── 11. Fuse with frequency-domain evidence ──────────────────────
      // Benchmark showed frequency fusion slightly reduces accuracy (-1%)
      // on 110-image test. Kept as informational signal; fusion disabled.
      const pFused = pCal;

      // ── 12. Fuse with metadata signals ───────────────────────────────
      const p = fuseSignals(pFused, signals);

      const result = {
        ok: true,
        p, // final fused probability
        pRaw, // raw sigmoid(logit), pre-calibration
        logit, // raw model output
        signals, // metadata analysis
        freq, // frequency analysis
        engine: state.engine,
        ms,
        model: model.id,
      };

      // ── 13. Cache (LRU eviction) ─────────────────────────────────────
      if (state.cache.size >= LRU_MAX) {
        const oldest = state.cache.keys().next().value;
        state.cache.delete(oldest);
      }
      state.cache.set(src, result);

      dlog(`analyze: p=${p.toFixed(4)} pRaw=${pRaw.toFixed(4)} logit=${logit.toFixed(3)} ${ms}ms [${state.engine}]`);
      return result;
    } finally {
      // Free bitmap memory regardless of success/failure.
      bitmap.close();
    }
  } catch (e) {
    dlog(`analyze error: ${e.message}`);
    return { ok: false, error: e.message };
  }
}

// ────────────────────────────────────────────────────────────────────────────
// explain() — 3×3 heatmap
// ────────────────────────────────────────────────────────────────────────────

/**
 * Generate a 3×3 spatial heatmap by running inference on 9 sub-regions of
 * the image.  This is an *on-demand* operation — the 9 extra inferences are
 * NOT performed during regular `analyze()`.
 *
 * The image is divided into a 3×3 grid.  For each cell, a square region
 * (sized to the smaller cell dimension) is extracted, resized to
 * `inputSize`, and run through the model.  The resulting 9 probabilities
 * form a heatmap that reveals which spatial areas drive the model's
 * prediction.
 *
 * @param {Object} params
 * @param {string} [params.url]     - Image URL.
 * @param {string} [params.dataUrl] - Data URL.
 * @returns {Promise<{ok:boolean, base?:number, grid?:number[],
 *                     engine?:string, model?:string, ms?:number,
 *                     error?:string}>}
 */
export async function explain({ url, dataUrl }) {
  const src = url || dataUrl;
  if (!src) {
    return { ok: false, error: 'No url or dataUrl provided' };
  }

  try {
    await ensureSession();

    const t0 = performance.now();
    const session = state.session;
    const model = state.model;
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    // Fetch + decode (reuse the same path as analyze for consistency).
    const u8 = await fetchBytes(src);
    const blob = new Blob([u8]);
    const bitmap = await createImageBitmap(blob, {
      colorSpaceConversion: 'none',
      premultiplyAlpha: 'none',
    });

    try {
      const w = bitmap.width;
      const h = bitmap.height;

      // ── Base inference (full image) ──────────────────────────────────
      const baseTensor = preprocess(bitmap);
      const baseResults = await serialized(() =>
        session.run({ [inputName]: baseTensor }),
      );
      const baseLogit = baseResults[outputName].data[0];
      const base = sigmoid(baseLogit);

      // ── 3×3 grid of region inferences ────────────────────────────────
      const cellW = w / 3;
      const cellH = h / 3;
      const rs = Math.floor(Math.min(cellW, cellH)); // square region size

      const grid = [];
      const gridLogits = [];

      for (let row = 0; row < 3; row++) {
        for (let col = 0; col < 3; col++) {
          const rx = col * cellW;
          const ry = row * cellH;
          const regionTensor = preprocessRegion(bitmap, rx, ry, rs);
          const regionResults = await serialized(() =>
            session.run({ [inputName]: regionTensor }),
          );
          const regionLogit = regionResults[outputName].data[0];
          gridLogits.push(regionLogit);
          grid.push(sigmoid(regionLogit));
        }
      }

      const t1 = performance.now();
      const ms = Math.round(t1 - t0);

      dlog(`explain: base=${base.toFixed(4)} grid=[${grid.map((g) => g.toFixed(2)).join(', ')}] ${ms}ms`);

      return {
        ok: true,
        base, // full-image probability
        grid, // 9 region probabilities (row-major: TL→TR→BL→BR)
        logits: gridLogits, // raw logits for debugging / custom viz
        engine: state.engine,
        model: model.id,
        ms,
      };
    } finally {
      bitmap.close();
    }
  } catch (e) {
    dlog(`explain error: ${e.message}`);
    return { ok: false, error: e.message };
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Status & control
// ────────────────────────────────────────────────────────────────────────────

/**
 * Return a snapshot of the engine's current state for the popup / options UI.
 *
 * @returns {{status:string, progress:number, engine:string, model:(string|null),
 *           log:string[], timings:number[]}}
 */
export function statusPayload() {
  return {
    status: state.status,
    progress: state.progress,
    engine: state.engine,
    model: state.model ? state.model.id : null,
    log: [...state.log], // shallow copy — prevents external mutation
    timings: [...state.timings],
  };
}

/**
 * Update the calibration parameters.  Called when the user adjusts the
 * calibration slider or when a calibration script recomputes values.
 *
 * The calibration applies a logistic transform: `pCal = sigmoid(a*logit + b)`.
 * Identity calibration is `{a:1, b:0}` (no change from raw sigmoid).
 *
 * @param {{a:number, b:number}} c - New calibration parameters.
 * @returns {void}
 */
export function setCalibration(c) {
  state.calibration = { a: c.a, b: c.b };
  dlog(`Calibration updated: a=${c.a}, b=${c.b}`);
}

/**
 * Record an error in the engine state.  Typically called by the service
 * worker's message handler when an unexpected exception occurs outside the
 * normal analyze/explain flow.
 *
 * @param {Error|string} e - Error object or message.
 * @returns {void}
 */
export function noteError(e) {
  state.error = e instanceof Error ? e.message : String(e);
  state.status = 'error';
  dlog(`Error noted: ${state.error}`);
}
