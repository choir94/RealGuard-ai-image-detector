/**
 * RealGuard — Build Script
 * 
 * Builds the extension for distribution:
 * 1. Copies Transformers.js library + ONNX Runtime WASM files locally
 * 2. Copies source files
 * 3. Generates icons
 * 4. Creates dist/ folder with complete extension
 */

const fs = require('fs');
const path = require('path');
const https = require('https');

const ROOT = path.join(__dirname, '..');
const DIST_DIR = path.join(ROOT, 'dist');
const NODE_MODULES = path.join(ROOT, 'node_modules');

const COPY_FILES = [
  'manifest.json',
  'background.js',
  'content.js',
  'overlay.css',
  'LICENSE',
  'popup/popup.html',
  'popup/popup.js',
];

function ensureDir(dir) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        downloadFile(response.headers.location, dest).then(resolve).catch(reject);
        return;
      }
      response.pipe(file);
      file.on('finish', () => file.close(resolve));
    }).on('error', (err) => {
      fs.unlink(dest, () => {});
      reject(err);
    });
  });
}

async function build() {
  console.log('[RealGuard] Building extension...');

  // Clean dist
  if (fs.existsSync(DIST_DIR)) {
    fs.rmSync(DIST_DIR, { recursive: true });
  }
  ensureDir(DIST_DIR);
  ensureDir(path.join(DIST_DIR, 'popup'));
  ensureDir(path.join(DIST_DIR, 'lib'));
  ensureDir(path.join(DIST_DIR, 'icons'));

  // Copy static files
  for (const file of COPY_FILES) {
    const src = path.join(ROOT, file);
    const dest = path.join(DIST_DIR, file);
    ensureDir(path.dirname(dest));
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, dest);
      console.log(`  ✓ ${file}`);
    }
  }

  // Copy Transformers.js library from node_modules
  const tfjsSrc = path.join(NODE_MODULES, '@huggingface', 'transformers', 'dist', 'transformers.min.js');
  const tfjsDest = path.join(DIST_DIR, 'lib', 'transformers.min.js');
  if (fs.existsSync(tfjsSrc)) {
    fs.copyFileSync(tfjsSrc, tfjsDest);
    console.log('  ✓ lib/transformers.min.js (from node_modules)');
  } else {
    console.log('  Downloading Transformers.js from CDN...');
    await downloadFile('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0/dist/transformers.min.js', tfjsDest);
    console.log('  ✓ lib/transformers.min.js (from CDN)');
  }

  // Copy ONNX Runtime WASM files (needed for inference)
  const ortDir = path.join(NODE_MODULES, 'onnxruntime-web', 'dist');
  const ortDest = path.join(DIST_DIR, 'lib', 'ort');
  ensureDir(ortDest);

  // We need: the JSEP wasm (for WebGPU), the threaded wasm (for WASM fallback), and the .mjs loader
  const ortFiles = [
    'ort-wasm-simd-threaded.jsep.wasm',   // WebGPU backend
    'ort-wasm-simd-threaded.wasm',         // WASM backend
    'ort-wasm-simd-threaded.jsep.mjs',     // JSEP module loader
  ];

  for (const f of ortFiles) {
    const src = path.join(ortDir, f);
    const dest = path.join(ortDest, f);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, dest);
      const sizeMB = (fs.statSync(dest).size / 1024 / 1024).toFixed(1);
      console.log(`  ✓ lib/ort/${f} (${sizeMB}MB)`);
    } else {
      console.warn(`  ⚠ Missing: ${f}`);
    }
  }

  // Also copy the ort-wasm-simd-threaded.mjs (needed by some code paths)
  const ortMjsSrc = path.join(ortDir, 'ort-wasm-simd-threaded.mjs');
  if (fs.existsSync(ortMjsSrc)) {
    fs.copyFileSync(ortMjsSrc, path.join(ortDest, 'ort-wasm-simd-threaded.mjs'));
    console.log('  ✓ lib/ort/ort-wasm-simd-threaded.mjs');
  }

  // Generate icons
  console.log('  Generating icons...');
  ensureDir(path.join(DIST_DIR, 'icons'));
  for (const size of [16, 48, 128]) {
    const iconPath = path.join(DIST_DIR, 'icons', `icon-${size}.png`);
    generatePlaceholderPNG(iconPath, size);
  }
  console.log('  ✓ icons/');

  // Summary
  const totalSize = getDirSize(DIST_DIR);
  console.log(`\n[RealGuard] Build complete! (${(totalSize / 1024 / 1024).toFixed(1)}MB total)`);
  console.log(`  Output: ${DIST_DIR}`);
  console.log('\nTo install:');
  console.log('  1. Open chrome://extensions');
  console.log('  2. Enable "Developer mode"');
  console.log('  3. Click "Load unpacked"');
  console.log('  4. Select the dist/ folder');
}

function getDirSize(dir) {
  let size = 0;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      size += getDirSize(p);
    } else {
      size += fs.statSync(p).size;
    }
  }
  return size;
}

// Generate a minimal valid PNG file (solid color with shield icon)
function generatePlaceholderPNG(filepath, size) {
  const zlib = require('node:zlib');
  
  const pixels = Buffer.alloc(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      // Blue gradient background
      pixels[i] = 37;      // R
      pixels[i + 1] = 99;  // G  
      pixels[i + 2] = 235; // B
      pixels[i + 3] = 255; // A
    }
  }

  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8;   // bit depth
  ihdr[9] = 6;   // color type (RGBA)
  ihdr[10] = 0;
  ihdr[11] = 0;
  ihdr[12] = 0;

  const rawData = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    rawData[y * (size * 4 + 1)] = 0;
    pixels.copy(rawData, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  }
  const compressedData = zlib.deflateSync(rawData);

  const png = Buffer.concat([
    signature,
    makeChunk('IHDR', ihdr),
    makeChunk('IDAT', compressedData),
    makeChunk('IEND', Buffer.alloc(0)),
  ]);

  fs.writeFileSync(filepath, png);
}

function makeChunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const typeBuf = Buffer.from(type, 'ascii');
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuf, data])), 0);
  return Buffer.concat([length, typeBuf, data, crc]);
}

function crc32(buf) {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) {
    crc ^= buf[i];
    for (let j = 0; j < 8; j++) {
      crc = (crc >>> 1) ^ (0xEDB88320 & -(crc & 1));
    }
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

build().catch(console.error);
