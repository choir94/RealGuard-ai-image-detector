// RealGuard — E2E extension test.
//
// Loads the built extension in headless Chromium, navigates to a test page
// with known real and AI images, and verifies:
//   1. Extension service worker registers
//   2. Offscreen document creates an ONNX session
//   3. Content script discovers images and sets data-realguard-* attributes
//   4. Classification results are correct (real → REAL, AI → AI)
//
// Prerequisites:
//   - npm run build (extension/dist/ must exist)
//   - A Chrome/Chromium binary (set CHROME_PATH env or auto-detect)
//
// Usage:
//   node e2e/extension-test.mjs
//
// Environment:
//   CHROME_PATH  — path to chrome/chromium binary (optional, auto-detected)

import { existsSync, mkdirSync, rmSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const DIST = join(ROOT, 'extension', 'dist');

// ── Test configuration ──────────────────────────────────────────────────────

const TEST_IMAGES = [
  { id: 'real-1', src: 'https://picsum.photos/id/237/400/400', expected: 'real' },
  { id: 'real-2', src: 'https://picsum.photos/id/238/400/400', expected: 'real' },
  { id: 'ai-1', src: 'https://image.pollinations.ai/prompt/a%20beautiful%20sunset%20over%20mountains%20digital%20art', expected: 'ai' },
  { id: 'ai-2', src: 'https://image.pollinations.ai/prompt/a%20futuristic%20cyberpunk%20city%20at%20night%20with%20neon%20lights%20digital%20art%20painting', expected: 'ai' },
];

const TEST_HTML = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RealGuard E2E Test</title>
<style>body{font-family:sans-serif;padding:20px}img{max-width:300px;margin:10px;border:2px solid #ccc}</style>
</head><body>
<h1>RealGuard E2E Test</h1>
${TEST_IMAGES.map((img) => `<img id="${img.id}" src="${img.src}">`).join('\n')}
</body></html>`;

// ── Chrome binary detection ─────────────────────────────────────────────────

function findChrome() {
  if (process.env.CHROME_PATH && existsSync(process.env.CHROME_PATH)) {
    return process.env.CHROME_PATH;
  }
  const candidates = [
    '/root/.cloakbrowser/chromium-146.0.7680.177.5/chrome',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/opt/homebrew/bin/chromium',
  ];
  for (const p of candidates) {
    if (existsSync(p)) return p;
  }
  throw new Error('Chrome/Chromium not found. Set CHROME_PATH env variable.');
}

// ── Playwright runner ───────────────────────────────────────────────────────

async function runE2E() {
  if (!existsSync(DIST)) {
    console.error('✘ extension/dist/ not found. Run "npm run build" first.');
    process.exit(1);
  }

  const chromePath = findChrome();
  const userData = join(process.env.TMPDIR || '/tmp', 'rg-e2e-test');
  if (existsSync(userData)) rmSync(userData, { recursive: true });

  // Write test page
  const http = await import('node:http');
  const testPagePath = '/rg_e2e.html';

  const server = http.createServer((req, res) => {
    if (req.url === testPagePath) {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(TEST_HTML);
    } else {
      res.writeHead(404);
      res.end('Not found');
    }
  });

  await new Promise((resolve) => server.listen(8765, resolve));
  console.log('  Test server: http://127.0.0.1:8765' + testPagePath);

  // Check if playwright is available
  let playwright;
  try {
    playwright = await import('playwright');
  } catch {
    console.error('✘ playwright not installed. Install with: npm install -D playwright');
    server.close();
    process.exit(1);
  }

  const { chromium } = playwright;
  const results = [];

  try {
    const context = await chromium.launchPersistentContext(userData, {
      executablePath: chromePath,
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-dev-shm-usage',
        `--disable-extensions-except=${DIST}`,
        `--load-extension=${DIST}`,
      ],
      ignoreDefaultArgs: ['--disable-extensions'],
    });

    // Wait for service worker
    await new Promise((r) => setTimeout(r, 3000));
    const workers = context.serviceWorkers;

    if (workers.length === 0) {
      console.error('✘ No service worker registered');
      process.exit(1);
    }

    const extId = workers[0].url.split('://')[1].split('/')[0];
    console.log(`  Extension ID: ${extId}`);
    console.log(`  Service worker: ${workers[0].url.split('/').pop()}`);

    const page = await context.newPage();

    // Track errors
    const errors = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`http://127.0.0.1:8765${testPagePath}`, {
      waitUntil: 'networkidle',
      timeout: 60000,
    });

    // Wait for engine ready
    console.log('  Waiting for engine ready...');
    let ready = false;
    for (let i = 0; i < 20; i++) {
      await new Promise((r) => setTimeout(r, 3000));
      try {
        const status = await workers[0].evaluate(async () => {
          try {
            return await chrome.runtime.sendMessage({
              target: 'offscreen',
              type: 'OFFSCREEN_STATUS',
            });
          } catch {
            return null;
          }
        });
        if (status?.status === 'ready') {
          console.log(`  Engine ready (${(i + 1) * 3}s) — ${status.engine}`);
          ready = true;
          break;
        }
        if (status?.status === 'error') {
          console.error(`  Engine error: ${status.error || 'unknown'}`);
          break;
        }
      } catch {
        // SW may have been recycled
      }
    }

    if (!ready) {
      console.error('✘ Engine did not become ready');
      process.exit(1);
    }

    // Wait for content script to process all images
    console.log('  Waiting for image analysis...');
    await new Promise((r) => setTimeout(r, 30000));

    // Collect results
    const imgResults = await page.evaluate(() => {
      const imgs = document.querySelectorAll('img');
      return Array.from(imgs).map((img) => {
        const attrs = {};
        for (const attr of img.attributes) {
          if (attr.name.startsWith('data-realguard')) {
            attrs[attr.name] = attr.value;
          }
        }
        return {
          id: img.id,
          src: img.src.substring(0, 80),
          naturalWidth: img.naturalWidth,
          score: attrs['data-realguard-score'],
          verdict: attrs['data-realguard-verdict'],
          logit: attrs['data-realguard-logit'],
          error: attrs['data-realguard-error'],
        };
      });
    });

    // Verify results
    let passed = 0;
    let failed = 0;

    console.log('\n  ┌─ Results ──────────────────────────────────────────────────');
    console.log('  │ ID       │ Expected │ Verdict │ Score   │ Status');
    console.log('  ├──────────┼──────────┼─────────┼─────────┼───────');

    for (const img of imgResults) {
      const expected = TEST_IMAGES.find((t) => t.id === img.id)?.expected;
      const actual = img.verdict;
      const ok = actual === expected;

      if (ok) passed++;
      else failed++;

      console.log(
        `  │ ${img.id.padEnd(8)} │ ${expected.padEnd(8)} │ ${(actual || 'none').padEnd(7)} │ ${(img.score || 'none').padEnd(7)} │ ${ok ? '✓ pass' : '✘ FAIL'}`,
      );

      results.push({ ...img, expected, passed: ok });
    }
    console.log('  └────────────────────────────────────────────────────────────');

    if (errors.length > 0) {
      console.log(`\n  Page errors (${errors.length}):`);
      for (const e of errors) console.log(`    ${e}`);
    }

    // Take screenshot
    const screenshotPath = join(process.env.TMPDIR || '/tmp', 'rg-e2e-screenshot.png');
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`\n  Screenshot: ${screenshotPath}`);

    await context.close();
  } finally {
    server.close();
    if (existsSync(userData)) rmSync(userData, { recursive: true });
  }

  // Summary
  console.log(`\n  ${passed} passed, ${failed} failed out of ${TEST_IMAGES.length} images`);
  if (failed > 0) {
    console.error('✘ E2E test FAILED');
    process.exit(1);
  }
  console.log('✓ E2E test PASSED');
}

runE2E().catch((err) => {
  console.error('✘ E2E test error:', err.message);
  process.exit(1);
});
