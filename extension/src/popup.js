// RealGuard — popup UI logic.

const $ = (id) => document.getElementById(id);

async function refresh() {
  try {
    const res = await chrome.runtime.sendMessage({ target: 'bg', type: 'engine-status' });
    if (!res?.ok) { $('status').textContent = 'error'; return; }
    $('status').textContent = res.status;
    $('engine').textContent = res.engine || '—';
    $('model').textContent = res.model || '—';
    if (res.progress) $('progress').value = res.progress;
    $('progress').style.display = res.status === 'downloading' ? 'block' : 'none';
  } catch { $('status').textContent = 'offline'; }
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) {
      const stats = await chrome.tabs.sendMessage(tab.id, { type: 'page-stats' });
      if (stats?.ok) {
        $('ai-count').textContent = stats.ai;
        $('real-count').textContent = stats.real;
        $('unsure-count').textContent = stats.unsure;
        $('pending-count').textContent = stats.pending;
        $('total-count').textContent = stats.total;
        $('stats').style.display = 'block';
      }
    }
  } catch { $('stats').style.display = 'none'; }
}

document.addEventListener('DOMContentLoaded', () => {
  refresh();
  $('rescan').addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) { await chrome.tabs.sendMessage(tab.id, { type: 'rescan' }); refresh(); }
  });
  $('lab').addEventListener('click', () => {
    chrome.runtime.sendMessage({ target: 'bg', type: 'open-lab' });
  });
  const blurToggle = $('blur-toggle');
  chrome.storage.local.get('blurAI', ({ blurAI = true }) => { blurToggle.checked = blurAI; });
  blurToggle.addEventListener('change', () => {
    chrome.storage.local.set({ blurAI: blurToggle.checked });
    chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
      if (tab?.id) chrome.tabs.sendMessage(tab.id, { type: 'set-blur', blurAI: blurToggle.checked });
    });
  });
  setInterval(refresh, 2000);
});
