/**
 * RealGuard — Background Service Worker
 * Handles model loading, caching, and inference orchestration.
 * Uses Transformers.js with WebGPU/WASM backend.
 */

import { pipeline, env } from './lib/transformers.min.js';

// Configure Transformers.js for browser extension (offline-capable)
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.remoteHost = 'https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/resolve/main/';
env.useBrowserCache = true;

// Configure ONNX Runtime WASM paths to use bundled files
// This is critical for offline operation
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('lib/ort/');

// Model configuration
const MODEL_ID = 'buildborderless/CommunityForensics-DeepfakeDet-ViT';
const CONFIDENCE_THRESHOLD = 0.65;
let detector = null;
let isInitializing = false;

/**
 * Initialize the AI detection model.
 * Downloads model on first run, uses cache afterward.
 */
async function initModel() {
  if (detector || isInitializing) return detector;
  isInitializing = true;

  try {
    console.log('[RealGuard] Initializing model...');
    const startTime = performance.now();

    // Try WebGPU first, fallback to WASM
    let device = 'wasm';
    try {
      const adapter = await navigator.gpu?.requestAdapter();
      if (adapter) device = 'webgpu';
    } catch (e) {
      console.log('[RealGuard] WebGPU not available, using WASM');
    }

    detector = await pipeline('image-classification', MODEL_ID, {
      device: device,
      dtype: 'q8',  // Use int8 quantized model (~22MB)
    });

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
    console.log(`[RealGuard] Model loaded in ${elapsed}s on ${device}`);

    // Notify content scripts that model is ready
    const tabs = await chrome.tabs.query({});
    for (const tab of tabs) {
      try {
        await chrome.tabs.sendMessage(tab.id, { type: 'MODEL_READY' });
      } catch (e) {
        // Tab might not have content script
      }
    }

    return detector;
  } catch (error) {
    console.error('[RealGuard] Model init failed:', error);
    throw error;
  } finally {
    isInitializing = false;
  }
}

/**
 * Run inference on an image and return confidence score.
 * @param {string} imageDataUrl - Base64 data URL of the image
 * @returns {Promise<{isAI: boolean, confidence: number, rawScore: number}>}
 */
async function detectImage(imageDataUrl) {
  if (!detector) {
    await initModel();
  }

  try {
    const result = await detector(imageDataUrl);

    // Model outputs single sigmoid logit: 0 = real, 1 = fake
    // Transformers.js pipeline returns [{label, score}] format
    let fakeProb;
    if (Array.isArray(result)) {
      // Find the "fake" label or use the raw score
      const fakeResult = result.find(r => 
        r.label?.toLowerCase().includes('fake') || 
        r.label?.toLowerCase().includes('ai') ||
        r.label?.toLowerCase().includes('synthetic')
      );
      if (fakeResult) {
        fakeProb = fakeResult.score;
      } else {
        // Single logit output — result[0].score is the fake probability
        fakeProb = result[0].score;
      }
    } else if (typeof result === 'object' && result.score !== undefined) {
      fakeProb = result.score;
    } else {
      fakeProb = 0.5;
    }

    // Apply confidence threshold
    const isAI = fakeProb >= CONFIDENCE_THRESHOLD;
    const confidence = Math.max(fakeProb, 1 - fakeProb);

    return {
      isAI,
      confidence: Math.round(confidence * 100),
      rawScore: fakeProb,
      aiProbability: Math.round(fakeProb * 100),
    };
  } catch (error) {
    console.error('[RealGuard] Detection error:', error);
    return {
      isAI: false,
      confidence: 0,
      rawScore: 0.5,
      aiProbability: 50,
      error: error.message,
    };
  }
}

// Message handler
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    try {
      switch (message.type) {
        case 'INIT_MODEL':
          await initModel();
          sendResponse({ success: true });
          break;

        case 'DETECT_IMAGE':
          const result = await detectImage(message.imageDataUrl);
          sendResponse(result);
          break;

        case 'GET_STATUS':
          sendResponse({
            modelLoaded: !!detector,
            isInitializing,
            modelId: MODEL_ID,
            threshold: CONFIDENCE_THRESHOLD,
          });
          break;

        default:
          sendResponse({ error: 'Unknown message type' });
      }
    } catch (error) {
      sendResponse({ error: error.message });
    }
  })();

  return true; // Keep channel open for async response
});

// Initialize model on extension install/update
chrome.runtime.onInstalled.addListener(async (details) => {
  console.log('[RealGuard] Extension installed/updated:', details.reason);
  if (details.reason === 'install') {
    // Pre-load model on first install
    await initModel();
  }
});
