// RealGuard — background service worker: manages the offscreen inference
// document and routes messages between content scripts, popup/lab pages, and
// the ONNX engine living in the offscreen document.
//
// The engine itself (engine.js) no longer runs in the service worker — it
// runs in an offscreen document so that the ONNX session survives
// service-worker teardowns and has a stable WebGPU context.  This worker
// handles offscreen document lifecycle, result caching, and request
// deduplication.

// ────────────────────────────────────────────────────────────────────────────
// Offscreen document lifecycle
// ────────────────────────────────────────────────────────────────────────────

/** Single-flight promise so concurrent callers don't create duplicate documents. */
let creatingOffscreen = null;

async function ensureOffscreenDocument() {
  // Check if an offscreen document already exists.
  const existing = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
  });
  if (existing.length > 0) return;

  // Another caller may have already started creation — piggyback on it.
  if (creatingOffscreen) return creatingOffscreen;

  creatingOffscreen = chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: ['WORKERS'],
    justification: 'Run ONNX image inference with WebGPU/WASM',
  });

  try {
    await creatingOffscreen;
  } finally {
    creatingOffscreen = null;
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Result cache & in-flight deduplication
// ────────────────────────────────────────────────────────────────────────────

/** LRU-style cache for analysis results by URL. Max 750 entries. */
const resultCache = new Map();
const CACHE_MAX = 750;

function cacheGet(url) {
  if (resultCache.has(url)) {
    const val = resultCache.get(url);
    // Move to end (most-recently-used).
    resultCache.delete(url);
    resultCache.set(url, val);
    return val;
  }
  return undefined;
}

function cacheSet(url, result) {
  if (resultCache.has(url)) resultCache.delete(url);
  resultCache.set(url, result);
  if (resultCache.size > CACHE_MAX) {
    // Evict oldest entry (first key in insertion order).
    const oldest = resultCache.keys().next().value;
    resultCache.delete(oldest);
  }
}

/** Deduplicates concurrent analysis requests for the same URL. */
const inFlight = new Map();

// ────────────────────────────────────────────────────────────────────────────
// Engine proxy — sends messages to the offscreen document
// ────────────────────────────────────────────────────────────────────────────

async function analyzeImage(url, dataUrl) {
  // Cache and dedup key — prefer url, fall back to dataUrl for blob: flows.
  const key = url || dataUrl;

  // 1. Check cache first.
  const cached = cacheGet(key);
  if (cached) return { ok: true, ...cached, cached: true };

  // 2. Deduplicate concurrent requests for the same key.
  if (inFlight.has(key)) return inFlight.get(key);

  const promise = (async () => {
    await ensureOffscreenDocument();
    const res = await chrome.runtime.sendMessage({
      target: 'offscreen',
      type: 'OFFSCREEN_ANALYZE',
      url,
      dataUrl,
    });
    if (res?.ok) {
      // Strip the ok flag before caching — we re-add it on response.
      const { ok, ...result } = res;
      cacheSet(key, result);
      return { ok: true, ...result };
    }
    return res || { ok: false, error: 'no response from offscreen' };
  })();

  inFlight.set(key, promise);
  try {
    return await promise;
  } finally {
    inFlight.delete(key);
  }
}

async function explainImage(url, dataUrl) {
  await ensureOffscreenDocument();
  const res = await chrome.runtime.sendMessage({
    target: 'offscreen',
    type: 'OFFSCREEN_EXPLAIN',
    url,
    dataUrl,
  });
  return res || { ok: false, error: 'no response from offscreen' };
}

// ────────────────────────────────────────────────────────────────────────────
// Message routing — from content scripts, popup, and lab
// ────────────────────────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || msg.target !== 'bg') return undefined;

  (async () => {
    switch (msg.type) {
      // New canonical message types
      case 'ANALYZE_IMAGE':
        sendResponse(await analyzeImage(msg.url, msg.dataUrl));
        break;
      case 'EXPLAIN_IMAGE':
        sendResponse(await explainImage(msg.url, msg.dataUrl));
        break;
      case 'ENGINE_STATUS':
        await ensureOffscreenDocument();
        sendResponse(
          await chrome.runtime.sendMessage({ target: 'offscreen', type: 'OFFSCREEN_STATUS' }),
        );
        break;
      case 'SET_CALIBRATION':
        await ensureOffscreenDocument();
        sendResponse(
          await chrome.runtime.sendMessage({
            target: 'offscreen',
            type: 'OFFSCREEN_SET_CALIBRATION',
            calibration: msg.calibration,
          }),
        );
        break;
      case 'OPEN_LAB':
        await chrome.tabs.create({ url: chrome.runtime.getURL('lab.html') });
        sendResponse({ ok: true });
        break;
      // Backward-compatible aliases for existing content.js / popup.js / lab.js
      case 'analyze':
        sendResponse(await analyzeImage(msg.url, msg.dataUrl));
        break;
      case 'explain':
        sendResponse(await explainImage(msg.url, msg.dataUrl));
        break;
      case 'ensure-model':
        await ensureOffscreenDocument();
        sendResponse(
          await chrome.runtime.sendMessage({ target: 'offscreen', type: 'OFFSCREEN_ENSURE_MODEL' }),
        );
        break;
      case 'engine-status':
        await ensureOffscreenDocument();
        sendResponse(
          await chrome.runtime.sendMessage({ target: 'offscreen', type: 'OFFSCREEN_STATUS' }),
        );
        break;
      case 'set-calibration':
        await ensureOffscreenDocument();
        sendResponse(
          await chrome.runtime.sendMessage({
            target: 'offscreen',
            type: 'OFFSCREEN_SET_CALIBRATION',
            calibration: msg.calibration,
          }),
        );
        break;
      case 'open-lab':
        await chrome.tabs.create({ url: chrome.runtime.getURL('lab.html') });
        sendResponse({ ok: true });
        break;
      default:
        sendResponse({ ok: false, error: `unknown message: ${msg.type}` });
    }
  })().catch((e) => {
    sendResponse({ ok: false, error: String(e?.message || e) });
  });

  return true; // async response
});

// ────────────────────────────────────────────────────────────────────────────
// Context menu
// ────────────────────────────────────────────────────────────────────────────

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'realguard-analyze' && tab?.id != null) {
    chrome.tabs
      .sendMessage(tab.id, { type: 'analyze-src', srcUrl: info.srcUrl })
      .catch(() => {});
  }
});

// ────────────────────────────────────────────────────────────────────────────
// Lifecycle hooks
// ────────────────────────────────────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'realguard-analyze',
    title: 'Analyze image with RealGuard',
    contexts: ['image'],
  });
  // Kick off offscreen document creation + model download so the extension
  // is ready (and fully offline-capable) as soon as possible after install.
  ensureOffscreenDocument().catch(() => {});
});

chrome.runtime.onStartup.addListener(() => {
  ensureOffscreenDocument().catch(() => {});
});
