/**
 * RealGuard — Popup Script
 * Shows model status and scanning statistics.
 */

const statusEl = document.getElementById('status');
const statusTextEl = document.getElementById('status-text');
const statScanned = document.getElementById('stat-scanned');
const statAI = document.getElementById('stat-ai');

// Check model status
function checkStatus() {
  chrome.runtime.sendMessage({ type: 'GET_STATUS' }, (response) => {
    if (chrome.runtime.lastError || !response) {
      statusEl.className = 'status error';
      statusTextEl.textContent = 'Service worker starting...';
      setTimeout(checkStatus, 2000);
      return;
    }

    if (response.modelLoaded) {
      statusEl.className = 'status ready';
      statusTextEl.innerHTML = '✓ Model loaded & ready';
    } else if (response.isInitializing) {
      statusEl.className = 'status loading';
      statusTextEl.textContent = 'Loading AI model...';
      setTimeout(checkStatus, 2000);
    } else {
      statusEl.className = 'status loading';
      statusTextEl.textContent = 'Initializing...';
      chrome.runtime.sendMessage({ type: 'INIT_MODEL' }, () => {
        setTimeout(checkStatus, 2000);
      });
    }
  });
}

// Get scan stats from active tab
function getScanStats() {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) {
      chrome.tabs.sendMessage(tabs[0].id, { type: 'GET_STATS' }, (response) => {
        if (chrome.runtime.lastError || !response) return;
        statScanned.textContent = response.scanned || 0;
        statAI.textContent = response.aiDetected || 0;
      });
    }
  });
}

// Listen for stats updates
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'STATS_UPDATE') {
    statScanned.textContent = message.scanned || 0;
    statAI.textContent = message.aiDetected || 0;
  }
});

checkStatus();
getScanStats();
setInterval(getScanStats, 3000);
