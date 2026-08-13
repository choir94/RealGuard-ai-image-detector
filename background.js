/**
 * RealGuard — Background Service Worker
 * Handles model loading, caching, and inference orchestration.
 * Uses Transformers.js with WebGPU/WASM backend.
 * 
 * The CommunityForensics model uses num_labels=1 (sigmoid output):
 * - Raw output is a single logit
 * - sigmoid(logit) = P(fake)
 * - We bypass the pipeline's softmax and apply sigmoid manually
 */

import { pipeline, env, AutoModel, AutoProcessor, RawImage } from './lib/transformers.min.js';

// Configure Transformers.js for browser extension (offline-capable)
env.allowLocalModels = false;
env.allowRemoteModels = true;
env.remoteHost = 'https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/resolve/main/';
env.useBrowserCache = true;

// Configure ONNX Runtime WASM paths to use bundled files
env.backends.onnx.wasm.wasmPaths = chrome.runtime.getURL('lib/ort/');

// Model configuration
const MODEL_ID = 'buildborderless/CommunityForensics-DeepfakeDet-ViT';
const CONFIDENCE_THRESHOLD = 0.65;
let model = null;
let processor = null;
let isInitializing = false;

/**
 * Initialize the AI detection model.
 * Downloads model on first run (~22MB INT8), uses browser cache afterward.
 * Tries WebGPU first, falls back to WASM.
 */
async function initModel() {
  if ((model && processor) || isInitializing) return;
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

    // Load model and processor separately for more control over output
    // Using AutoModel instead of pipeline to bypass softmax on single-label model
    model = await AutoModel.from_pretrained(MODEL_ID, {
      device: device,
      dtype: 'q8',
    });

    processor = await AutoProcessor.from_pretrained(MODEL_ID);

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
  } catch (error) {
    console.error('[RealGuard] Model init failed:', error);
    throw error;
  } finally {
    isInitializing = false;
  }
}

/**
 * Sigmoid function: converts raw logit to probability.
 * @param {number} logit - raw model output
 * @returns {number} probability between 0 and 1
 */
function sigmoid(logit) {
  return 1 / (1 + Math.exp(-logit));
}

/**
 * Run inference on an image and return confidence score.
 * 
 * The CommunityForensics model (num_labels=1) outputs a single logit.
 * sigmoid(logit) = P(fake). We bypass the pipeline's softmax and
 * apply sigmoid manually for correct probability interpretation.
 * 
 * @param {string} imageDataUrl - Base64 data URL of the image
 * @returns {Promise<{isAI: boolean, confidence: number, rawScore: number, aiProbability: number}>}
 */
async function detectImage(imageDataUrl) {
  if (!model || !processor) {
    await initModel();
  }

  try {
    // Load image from data URL
    const image = await RawImage.fromURL(imageDataUrl);

    // Preprocess: processor handles resize (shortest_edge=440), 
    // center crop (384x384), and CLIP normalization
    const inputs = await processor(image);

    // Run inference — model returns {logits} with shape [1, 1]
    const outputs = await model(inputs);

    // Extract the raw logit
    let logit;
    if (outputs.logits) {
      const logitsData = outputs.logits.data || outputs.logits.tolist?.();
      if (logitsData instanceof Float32Array || Array.isArray(logitsData)) {
        logit = Array.isArray(logitsData[0]) ? logitsData[0][0] : logitsData[0];
      } else {
        logit = logitsData;
      }
    } else {
      console.warn('[RealGuard] Unexpected output format:', Object.keys(outputs));
      logit = 0;
    }

    // Apply sigmoid to get P(fake)
    const fakeProb = sigmoid(logit);

    // Apply confidence threshold (65% per bounty spec)
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
            modelLoaded: !!(model && processor),
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
