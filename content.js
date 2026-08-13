/**
 * RealGuard — Content Script
 * Detects <img> elements on the page, sends them for AI detection,
 * and overlays confidence score badges.
 */

const PROCESSED_ATTR = 'data-realguard-processed';
const BADGE_CLASS = 'realguard-badge';
const MIN_IMAGE_SIZE = 64;
const MAX_IMAGE_SIZE = 4096;
const SCAN_INTERVAL_MS = 3000;
const MAX_CONCURRENT_SCANS = 3;

let scanQueue = [];
let activeScans = 0;
let modelReady = false;
let stats = { scanned: 0, aiDetected: 0 };

/**
 * Check if an image element is worth scanning.
 */
function isScannableImage(img) {
  // Skip already processed
  if (img.hasAttribute(PROCESSED_ATTR)) return false;
  if (img.getAttribute('data-realguard-skip') === 'true') return false;

  // Skip hidden images
  if (img.offsetWidth === 0 || img.offsetHeight === 0) return false;

  // Check displayed size (use naturalWidth for actual image size)
  const width = img.naturalWidth || img.offsetWidth;
  const height = img.naturalHeight || img.offsetHeight;

  if (width < MIN_IMAGE_SIZE || height < MIN_IMAGE_SIZE) return false;
  if (width > MAX_IMAGE_SIZE || height > MAX_IMAGE_SIZE) return false;

  // Skip data URIs that are too small (likely icons)
  if (img.src.startsWith('data:')) {
    if (img.src.length < 5000) return false;
  }

  // Skip known non-content images
  const skipPatterns = ['favicon', 'sprite', 'logo', 'icon', 'avatar'];
  const src = (img.src || '').toLowerCase();
  const alt = (img.alt || '').toLowerCase();
  // Only skip if it's clearly a UI icon, not a photo that happens to have "icon" in name
  if (width < 128 && height < 128) {
    for (const pattern of skipPatterns) {
      if (src.includes(pattern) || alt.includes(pattern)) return false;
    }
  }

  return true;
}

/**
 * Convert image element to data URL for inference.
 */
async function imageToDataUrl(img) {
  // If already a data URL, use directly
  if (img.src.startsWith('data:')) {
    return img.src;
  }

  // Use canvas to extract image data
  try {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Cap canvas size for performance
    const maxDim = 512;
    let w = img.naturalWidth || img.width;
    let h = img.naturalHeight || img.height;

    if (w > maxDim || h > maxDim) {
      const ratio = Math.min(maxDim / w, maxDim / h);
      w = Math.round(w * ratio);
      h = Math.round(h * ratio);
    }

    canvas.width = w;
    canvas.height = h;
    ctx.drawImage(img, 0, 0, w, h);

    return canvas.toDataURL('image/jpeg', 0.85);
  } catch (e) {
    // CORS or tainted canvas — try crossOrigin approach
    try {
      const proxyImg = new Image();
      proxyImg.crossOrigin = 'anonymous';
      proxyImg.src = img.src;

      await new Promise((resolve, reject) => {
        proxyImg.onload = resolve;
        proxyImg.onerror = reject;
        setTimeout(reject, 5000);
      });

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const maxDim = 512;
      let w = proxyImg.naturalWidth;
      let h = proxyImg.naturalHeight;

      if (w > maxDim || h > maxDim) {
        const ratio = Math.min(maxDim / w, maxDim / h);
        w = Math.round(w * ratio);
        h = Math.round(h * ratio);
      }

      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(proxyImg, 0, 0, w, h);

      return canvas.toDataURL('image/jpeg', 0.85);
    } catch (e2) {
      console.debug('[RealGuard] Cannot process image:', img.src, e2.message);
      return null;
    }
  }
}

/**
 * Create and attach a badge to an image element.
 */
function createBadge(img, result) {
  const badge = document.createElement('div');
  badge.className = BADGE_CLASS;

  const isAI = result.isAI;
  const confidence = result.confidence;

  if (isAI) {
    badge.classList.add('realguard-ai');
    badge.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/>
        <path d="M2 17l10 5 10-5"/>
        <path d="M2 12l10 5 10-5"/>
      </svg>
      <span>AI ${confidence}%</span>
    `;
  } else {
    badge.classList.add('realguard-real');
    badge.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M9 12l2 2 4-4"/>
        <circle cx="12" cy="12" r="10"/>
      </svg>
      <span>Real ${confidence}%</span>
    `;
  }

  // Wrap image in a relative-positioned container for badge placement
  if (img.parentNode) {
    const wrapper = document.createElement('div');
    wrapper.style.position = 'relative';
    wrapper.style.display = 'inline-block';
    wrapper.className = 'realguard-wrapper';

    img.parentNode.insertBefore(wrapper, img);
    wrapper.appendChild(img);
    wrapper.appendChild(badge);
  }

  // Update stats
  stats.scanned++;
  if (isAI) stats.aiDetected++;

  // Mark image as processed
  img.setAttribute(PROCESSED_ATTR, 'true');
  img.setAttribute('data-realguard-result', isAI ? 'ai' : 'real');
  img.setAttribute('data-realguard-confidence', confidence);
}

/**
 * Process a single image: convert to data URL, send for detection.
 */
async function processImage(img) {
  try {
    const dataUrl = await imageToDataUrl(img);
    if (!dataUrl) {
      img.setAttribute(PROCESSED_ATTR, 'true');
      img.setAttribute('data-realguard-skip', 'true');
      return;
    }

    const result = await chrome.runtime.sendMessage({
      type: 'DETECT_IMAGE',
      imageDataUrl: dataUrl,
    });

    if (result && !result.error) {
      createBadge(img, result);
    } else {
      img.setAttribute(PROCESSED_ATTR, 'true');
      img.setAttribute('data-realguard-skip', 'true');
    }
  } catch (error) {
    console.debug('[RealGuard] Image processing error:', error);
    img.setAttribute(PROCESSED_ATTR, 'true');
    img.setAttribute('data-realguard-skip', 'true');
  } finally {
    activeScans--;
    processQueue();
  }
}

/**
 * Process the scan queue with concurrency limit.
 */
function processQueue() {
  while (scanQueue.length > 0 && activeScans < MAX_CONCURRENT_SCANS) {
    const img = scanQueue.shift();
    if (img && document.body.contains(img) && isScannableImage(img)) {
      activeScans++;
      processImage(img);
    } else {
      img?.setAttribute(PROCESSED_ATTR, 'true');
    }
  }
}

/**
 * Scan the page for unprocessed images.
 */
function scanForImages() {
  if (!modelReady) return;

  const images = document.querySelectorAll('img');
  let newCount = 0;

  for (const img of images) {
    if (isScannableImage(img)) {
      scanQueue.push(img);
      newCount++;
    }
  }

  if (newCount > 0) {
    console.log(`[RealGuard] Found ${newCount} new images to scan`);
    processQueue();
  }
}

/**
 * Observe DOM mutations for dynamically added images.
 */
function setupMutationObserver() {
  const observer = new MutationObserver((mutations) => {
    let hasNewImages = false;

    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType === Node.ELEMENT_NODE) {
          if (node.tagName === 'IMG' && isScannableImage(node)) {
            scanQueue.push(node);
            hasNewImages = true;
          }
          // Check children of added nodes
          const imgs = node.querySelectorAll?.('img');
          if (imgs) {
            for (const img of imgs) {
              if (isScannableImage(img)) {
                scanQueue.push(img);
                hasNewImages = true;
              }
            }
          }
        }
      }
    }

    if (hasNewImages) {
      processQueue();
    }
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });
}

// Listen for messages from background
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'MODEL_READY') {
    modelReady = true;
    console.log('[RealGuard] Model ready, starting scan...');
    scanForImages();
    setupMutationObserver();
    setInterval(scanForImages, SCAN_INTERVAL_MS);
  }

  if (message.type === 'SCAN_NOW') {
    scanForImages();
  }

  if (message.type === 'GET_STATS') {
    sendResponse(stats);
  }
});

// Check if model is already loaded
chrome.runtime.sendMessage({ type: 'GET_STATUS' }, (response) => {
  if (chrome.runtime.lastError) {
    // Service worker might be starting up
    setTimeout(() => {
      chrome.runtime.sendMessage({ type: 'GET_STATUS' }, (response) => {
        if (response?.modelLoaded) {
          modelReady = true;
          scanForImages();
          setupMutationObserver();
          setInterval(scanForImages, SCAN_INTERVAL_MS);
        }
      });
    }, 2000);
    return;
  }

  if (response?.modelLoaded) {
    modelReady = true;
    scanForImages();
    setupMutationObserver();
    setInterval(scanForImages, SCAN_INTERVAL_MS);
  } else {
    // Request model initialization
    chrome.runtime.sendMessage({ type: 'INIT_MODEL' }, () => {
      modelReady = true;
      scanForImages();
      setupMutationObserver();
      setInterval(scanForImages, SCAN_INTERVAL_MS);
    });
  }
});

console.log('[RealGuard] Content script loaded');
