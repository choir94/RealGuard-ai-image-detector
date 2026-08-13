// Unit tests for the frequency-domain analysis module.
// Tests the pure functions (no OffscreenCanvas needed) and the fusion logic.
// Run: node --test tests/**/*.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';

const { toGrayscale, blockEnergy, fuseFrequency, dct2d, analyzeFrequency } =
  await import('../extension/src/lib/freq.js');

// ── toGrayscale ────────────────────────────────────────────────────────────

test('toGrayscale: BT.601 luma weights, alpha ignored', () => {
  // R=100, G=200, B=50, A=255 → Y = 0.299·100 + 0.587·200 + 0.114·50 = 153.0
  const data = new Uint8ClampedArray([100, 200, 50, 255]);
  const gray = toGrayscale({ data, width: 1, height: 1 });
  assert.equal(gray.length, 1);
  assert.ok(Math.abs(gray[0] - 153.0) < 0.01);
});

test('toGrayscale: white → 255, black → 0', () => {
  const data = new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]);
  const gray = toGrayscale({ data, width: 2, height: 1 });
  assert.equal(gray[0], 255);
  assert.equal(gray[1], 0);
});

// ── blockEnergy ────────────────────────────────────────────────────────────

test('blockEnergy: flat block → zero AC, zero high-freq', () => {
  // 8×8 block, all pixels = 128 (flat gray)
  const gray = new Float32Array(64).fill(128);
  const { dcEnergy, acEnergy, highFreqRatio } = blockEnergy(gray, 8);
  assert.ok(dcEnergy > 0, 'DC energy should be positive for non-zero mean');
  assert.equal(acEnergy, 0);
  assert.equal(highFreqRatio, 0);
});

test('blockEnergy: high-contrast checkerboard → high high-freq ratio', () => {
  // 8×8 checkerboard: alternating 0/255 → maximum high-frequency content
  const gray = new Float32Array(64);
  for (let i = 0; i < 64; i++) gray[i] = (i + Math.floor(i / 8)) % 2 ? 255 : 0;
  const { highFreqRatio } = blockEnergy(gray, 8);
  assert.ok(highFreqRatio > 0.5, `checkerboard should have high ratio, got ${highFreqRatio}`);
});

test('blockEnergy: smooth gradient → low high-freq ratio', () => {
  // 8×8 horizontal gradient: each column +1 → very smooth, low high-freq
  const gray = new Float32Array(64);
  for (let y = 0; y < 8; y++)
    for (let x = 0; x < 8; x++) gray[y * 8 + x] = x;
  const { highFreqRatio } = blockEnergy(gray, 8);
  assert.ok(highFreqRatio < 0.3, `smooth gradient should have low ratio, got ${highFreqRatio}`);
});

// ── dct2d ──────────────────────────────────────────────────────────────────

test('dct2d: DC coefficient equals scaled mean for flat block', () => {
  const block = new Float32Array(64).fill(100);
  const coeffs = dct2d(block, 8);
  // F(0,0) = N · mean = 8 · 100 = 800
  assert.ok(Math.abs(coeffs[0] - 800) < 0.5, `DC=${coeffs[0]}, expected ~800`);
  // All AC coefficients should be ~0
  let acMax = 0;
  for (let i = 1; i < 64; i++) acMax = Math.max(acMax, Math.abs(coeffs[i]));
  assert.ok(acMax < 0.5, `max AC=${acMax}, expected ~0`);
});

// ── fuseFrequency ──────────────────────────────────────────────────────────

test('fusion: uncertain + strong AI → nudge up', () => {
  const result = fuseFrequency(0.5, { score: 0.8 });
  const expected = 0.5 + 0.1 * (0.8 - 0.5);
  assert.ok(Math.abs(result - expected) < 1e-9);
  assert.ok(result > 0.5);
});

test('fusion: uncertain + max AI → nudge up (max ±0.05)', () => {
  const result = fuseFrequency(0.5, { score: 1.0 });
  assert.ok(Math.abs(result - 0.55) < 1e-9, `got ${result}`);
});

test('fusion: uncertain + strong real → nudge down', () => {
  const result = fuseFrequency(0.5, { score: 0.2 });
  const expected = 0.5 - 0.1 * (0.5 - 0.2);
  assert.ok(Math.abs(result - expected) < 1e-9);
  assert.ok(result < 0.5);
});

test('fusion: uncertain + max real → nudge down (max ±0.05)', () => {
  const result = fuseFrequency(0.5, { score: 0.0 });
  assert.ok(Math.abs(result - 0.45) < 1e-9, `got ${result}`);
});

test('fusion: neutral freq → no change', () => {
  assert.equal(fuseFrequency(0.5, { score: 0.5 }), 0.5);
});

test('fusion: mid-range freq (0.4–0.6) → no change', () => {
  assert.equal(fuseFrequency(0.5, { score: 0.6 }), 0.5);
  assert.equal(fuseFrequency(0.5, { score: 0.4 }), 0.5);
});

test('fusion: confident AI (pCal=0.8) → never overridden', () => {
  assert.equal(fuseFrequency(0.8, { score: 1.0 }), 0.8);
  assert.equal(fuseFrequency(0.8, { score: 0.0 }), 0.8);
});

test('fusion: confident real (pCal=0.2) → never overridden', () => {
  assert.equal(fuseFrequency(0.2, { score: 1.0 }), 0.2);
  assert.equal(fuseFrequency(0.2, { score: 0.0 }), 0.2);
});

test('fusion: boundary pCal=0.3 and pCal=0.7 are inside uncertain zone', () => {
  // pCal=0.3, freq=0.7 → nudge up
  assert.ok(fuseFrequency(0.3, { score: 0.7 }) > 0.3);
  // pCal=0.7, freq=0.3 → nudge down
  assert.ok(fuseFrequency(0.7, { score: 0.3 }) < 0.7);
});

test('fusion: just outside uncertain zone → no change', () => {
  assert.equal(fuseFrequency(0.29, { score: 0.7 }), 0.29);
  assert.equal(fuseFrequency(0.71, { score: 0.3 }), 0.71);
});

test('fusion: null/undefined freqResult → no change', () => {
  assert.equal(fuseFrequency(0.5, null), 0.5);
  assert.equal(fuseFrequency(0.5, undefined), 0.5);
  assert.equal(fuseFrequency(0.5, {}), 0.5);
});

test('fusion: NaN pCal → returned unchanged', () => {
  assert.ok(Number.isNaN(fuseFrequency(NaN, { score: 0.8 })));
});

// ── analyzeFrequency (degraded path, no OffscreenCanvas in Node) ──────────

test('analyzeFrequency: null bitmap → neutral result', async () => {
  const r = await analyzeFrequency(null);
  assert.equal(r.score, 0.5);
  assert.equal(r.energy_ratio, 0);
  assert.equal(r.high_freq_ratio, 0);
  assert.equal(r.detail, 0);
});

test('analyzeFrequency: tiny bitmap → neutral result', async () => {
  const r = await analyzeFrequency({ width: 16, height: 16 });
  assert.equal(r.score, 0.5);
});
