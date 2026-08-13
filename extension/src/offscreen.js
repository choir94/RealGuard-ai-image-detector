// RealGuard — offscreen document: hosts the ONNX inference engine.
//
// The service worker (background.js) can be torn down by Chrome at any time
// (~30s of inactivity), which destroys the ONNX session and any in-flight
// GPU work.  Moving inference into an offscreen document keeps the session
// alive across service-worker restarts and gives WebGPU a stable document
// context that doesn't get starved by background throttling.
//
// This script listens for messages from background.js, dispatches to the
// engine module, and returns results via sendResponse.

import {
  ensureSession,
  analyze,
  explain,
  statusPayload,
  setCalibration,
  noteError,
} from './lib/engine.js';

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || msg.target !== 'offscreen') return undefined;

  (async () => {
    switch (msg.type) {
      case 'OFFSCREEN_ANALYZE':
        sendResponse({ ok: true, ...await analyze({ url: msg.url, dataUrl: msg.dataUrl }) });
        break;
      case 'OFFSCREEN_EXPLAIN':
        sendResponse({ ok: true, ...await explain({ url: msg.url, dataUrl: msg.dataUrl }) });
        break;
      case 'OFFSCREEN_ENSURE_MODEL':
        await ensureSession();
        sendResponse({ ok: true, ...statusPayload() });
        break;
      case 'OFFSCREEN_STATUS':
        sendResponse({ ok: true, ...statusPayload() });
        break;
      case 'OFFSCREEN_SET_CALIBRATION':
        setCalibration(msg.calibration);
        sendResponse({ ok: true });
        break;
      default:
        sendResponse({ ok: false, error: `unknown offscreen message: ${msg.type}` });
    }
  })().catch((e) => {
    noteError(e);
    sendResponse({ ok: false, error: String(e?.message || e) });
  });

  return true; // keep the message channel open for async sendResponse
});

// Auto-boot: start downloading/loading the model immediately so the extension
// is ready to serve analysis requests as soon as possible.
ensureSession().catch(noteError);
