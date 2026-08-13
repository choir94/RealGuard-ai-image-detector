/**
 * RealGuard — Build Script (v2)
 *
 * Bundles the extension for distribution using esbuild:
 * 1. Bundles background.js + engine.js + meta.js + freq.js into a single module
 * 2. Copies static files (manifest, CSS, HTML, models.json, icons)
 * 3. Copies ONNX Runtime WASM binary for service-worker use
 * 4. Generates placeholder icons
 */

import esbuild from 'esbuild';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname;
const EXT_SRC = path.join(ROOT, 'extension');
const DIST = path.join(EXT_SRC, 'dist');
const NODE_MODULES = path.join(ROOT, 'node_modules');

const STATIC_FILES = [
  ['extension/manifest.json', 'extension/dist/manifest.json'],
  ['extension/static/content.css', 'extension/dist/content.css'],
  ['extension/static/popup.html', 'extension/dist/popup.html'],
  ['extension/static/offscreen.html', 'extension/dist/offscreen.html'],
  ['extension/static/lab.html', 'extension/dist/lab.html'],
  ['extension/static/models.json', 'extension/dist/models.json'],
  ['LICENSE', 'extension/dist/LICENSE'],
];

const STATIC_DIRS = [
  ['extension/static/assets', 'extension/dist/assets'],
  ['extension/static/vendor', 'extension/dist/vendor'],
];

function ensureDir(dir) { if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true }); }

function cp(src, dest) {
  ensureDir(path.dirname(dest));
  if (fs.existsSync(src)) fs.copyFileSync(src, dest);
}

async function build() {
  const watch = process.argv.includes('--watch');
  const zip = process.argv.includes('--zip');
  console.log('[RealGuard] Building extension v2...');

  // Clean dist
  if (fs.existsSync(DIST)) fs.rmSync(DIST, { recursive: true });
  ensureDir(DIST);

  // 1. Bundle background.js (entry point — pulls in engine.js, meta.js, freq.js)
  const bgEntry = path.join(EXT_SRC, 'src', 'background.js');
  await esbuild.build({
    entryPoints: [bgEntry],
    bundle: true,
    format: 'esm',
    target: 'chrome121',
    outfile: path.join(DIST, 'background.js'),
    sourcemap: false,
    minify: false,
    legalComments: 'none',
    logLevel: 'info',
  });
  console.log('  ✓ background.js (bundled)');

  // 2. Bundle content.js (standalone, no imports)
  await esbuild.build({
    entryPoints: [path.join(EXT_SRC, 'src', 'content.js')],
    bundle: true,
    format: 'iife',
    target: 'chrome121',
    outfile: path.join(DIST, 'content.js'),
    sourcemap: false,
    minify: false,
    legalComments: 'none',
    logLevel: 'info',
  });
  console.log('  ✓ content.js');

  // 3. Bundle popup.js
  await esbuild.build({
    entryPoints: [path.join(EXT_SRC, 'src', 'popup.js')],
    bundle: true,
    format: 'esm',
    target: 'chrome121',
    outfile: path.join(DIST, 'popup.js'),
    sourcemap: false,
    minify: false,
    legalComments: 'none',
    logLevel: 'info',
  });
  console.log('  ✓ popup.js');

  // 4. Bundle offscreen.js (hosts ONNX inference engine in offscreen document)
  const offscreenEntry = path.join(EXT_SRC, 'src', 'offscreen.js');
  if (fs.existsSync(offscreenEntry)) {
    await esbuild.build({
      entryPoints: [offscreenEntry],
      bundle: true,
      format: 'esm',
      target: 'chrome121',
      outfile: path.join(DIST, 'offscreen.js'),
      sourcemap: false,
      minify: false,
      legalComments: 'none',
      logLevel: 'info',
    });
    console.log('  ✓ offscreen.js (bundled)');
  }

  // 5. Bundle lab.js
  const labEntry = path.join(EXT_SRC, 'src', 'lab.js');
  if (fs.existsSync(labEntry)) {
    await esbuild.build({
      entryPoints: [labEntry],
      bundle: true,
      format: 'esm',
      target: 'chrome121',
      outfile: path.join(DIST, 'lab.js'),
      sourcemap: false,
      minify: false,
      legalComments: 'none',
      logLevel: 'info',
    });
    console.log('  ✓ lab.js');
  }

  // 6. Copy static files
  for (const [src, dest] of STATIC_FILES) {
    cp(path.join(ROOT, src), path.join(ROOT, dest));
    console.log(`  ✓ ${path.basename(dest)}`);
  }

  // 6. Copy static directories (assets, vendor)
  for (const [src, dest] of STATIC_DIRS) {
    if (fs.existsSync(src)) {
      ensureDir(path.join(ROOT, dest));
      fs.cpSync(path.join(ROOT, src), path.join(ROOT, dest), { recursive: true });
      console.log(`  ✓ ${path.basename(dest)}/`);
    }
  }

  // 8. Copy ONNX Runtime WASM + JSEP glue files
  const ortFiles = [
    ['ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.jsep.wasm'],
    ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.mjs'],
  ];
  for (const [srcName, destName] of ortFiles) {
    const src = path.join(NODE_MODULES, 'onnxruntime-web', 'dist', srcName);
    const dest = path.join(DIST, 'vendor', 'ort', destName);
    ensureDir(path.dirname(dest));
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, dest);
      const size = fs.statSync(dest).size;
      console.log(`  ✓ vendor/ort/${destName} (${(size / 1e6).toFixed(1)}MB)`);
    } else {
      console.warn(`  ⚠ Missing: ${srcName}`);
    }
  }

  // 8. Generate placeholder icons if not present
  ensureDir(path.join(DIST, 'assets'));
  for (const size of [16, 32, 48, 128]) {
    const iconPath = path.join(DIST, 'assets', `icon${size}.png`);
    if (!fs.existsSync(iconPath)) generatePlaceholderPNG(iconPath, size);
  }
  console.log('  ✓ assets/ (icons)');

  // Summary
  const totalSize = getDirSize(DIST);
  console.log(`\n[RealGuard] Build complete! (${(totalSize / 1e6).toFixed(1)}MB total)`);
  console.log(`  Output: ${DIST}`);
  console.log('\nTo install:');
  console.log('  1. Open chrome://extensions');
  console.log('  2. Enable "Developer mode"');
  console.log('  3. Click "Load unpacked"');
  console.log('  4. Select the extension/dist folder');

  if (zip) {
    // TODO: zip the dist folder
    console.log('\n  (zip not implemented yet)');
  }
}

function getDirSize(dir) {
  let size = 0;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) size += getDirSize(p);
    else size += fs.statSync(p).size;
  }
  return size;
}

function generatePlaceholderPNG(filepath, size) {
  const pixels = Buffer.alloc(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      pixels[i] = 37; pixels[i + 1] = 99; pixels[i + 2] = 235; pixels[i + 3] = 255;
    }
  }
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0); ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const rawData = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    rawData[y * (size * 4 + 1)] = 0;
    pixels.copy(rawData, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  }
  const compressedData = zlib.deflateSync(rawData);
  const png = Buffer.concat([signature, makeChunk('IHDR', ihdr), makeChunk('IDAT', compressedData), makeChunk('IEND', Buffer.alloc(0))]);
  fs.writeFileSync(filepath, png);
}

function makeChunk(type, data) {
  const length = Buffer.alloc(4); length.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([length, typeBuf, data, crc]);
}

function crc32(buf) {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) {
    crc ^= buf[i];
    for (let j = 0; j < 8; j++) crc = (crc >>> 1) ^ (0xEDB88320 & -(crc & 1));
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

build().catch((e) => { console.error(e); process.exit(1); });
