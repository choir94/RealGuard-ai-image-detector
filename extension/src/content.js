// RealGuard — content script.
// Discovers images on the page, requests analysis from the engine, and renders
// non-intrusive overlay badges with a confidence score + hover forensics panel.
// All UI lives in a shadow root in a zero-size overlay container so page CSS
// can't touch it and we can't break page layout.

const MIN_NATURAL = 64;
const MIN_DISPLAY = 40;
const AI_THRESHOLD = 0.65;
const REAL_THRESHOLD = 0.35;

const tracked = new Map();
let recId = 0;
let blurAI = true;
const byUrl = new Map();
let overlayRoot = null;
let enabled = true;
let rafPending = false;

// ------------------------------------------------------------ overlay ------

function ensureOverlay() {
  if (overlayRoot) return overlayRoot;
  const host = document.createElement('realguard-overlay');
  host.style.cssText = 'position:absolute;top:0;left:0;width:0;height:0;z-index:2147483646;';
  overlayRoot = host.attachShadow({ mode: 'closed' });
  const style = document.createElement('style');
  style.textContent = BADGE_CSS;
  overlayRoot.appendChild(style);
  document.documentElement.appendChild(host);
  return overlayRoot;
}

function makeBadge(rec) {
  const root = ensureOverlay();
  const el = document.createElement('div');
  el.className = 'badge scanning';
  el.innerHTML = `<span class="dot"></span><span class="txt">scanning</span>`;
  el.addEventListener('mouseenter', () => showPanel(rec));
  el.addEventListener('mouseleave', () => hidePanel());
  root.appendChild(el);
  return el;
}

function labelFor(p) {
  if (p >= AI_THRESHOLD) return ['ai', `AI ${Math.round(p * 100)}%`];
  if (p <= REAL_THRESHOLD) return ['real', `Real ${Math.round((1 - p) * 100)}%`];
  return ['unsure', `Unsure ${Math.round(p * 100)}%`];
}

function paintBadge(rec) {
  const el = rec.badge;
  if (!el) return;
  if (rec.status === 'done') {
    const [cls, txt] = labelFor(rec.result.p);
    el.className = `badge ${cls}`;
    el.querySelector('.txt').textContent = txt;
    applyBlur(rec);
  } else if (rec.status === 'error') {
    el.className = 'badge err';
    el.querySelector('.txt').textContent = '–';
    el.title = rec.error || 'analysis failed';
  }
}

// ----------------------------------------------------------- auto-blur -----

function applyBlur(rec) {
  if (rec.isBg) return;
  const shouldBlur = blurAI && !rec.revealed && rec.result && rec.result.p >= AI_THRESHOLD;
  if (shouldBlur && !rec.blurred) {
    rec.prevFilter = rec.img.style.filter || '';
    rec.img.style.setProperty('filter', 'blur(18px) saturate(.85)', 'important');
    rec.blurred = true;
    if (!rec.reveal) {
      const chip = document.createElement('div');
      chip.className = 'reveal';
      chip.textContent = 'AI image — click to reveal';
      chip.addEventListener('click', () => {
        rec.revealed = true;
        applyBlur(rec);
      });
      ensureOverlay().appendChild(chip);
      rec.reveal = chip;
    }
    positionReveal(rec);
  } else if (!shouldBlur && rec.blurred) {
    rec.img.style.setProperty('filter', rec.prevFilter);
    if (!rec.prevFilter) rec.img.style.removeProperty('filter');
    rec.blurred = false;
    rec.reveal?.remove();
    rec.reveal = null;
  }
}

function positionReveal(rec) {
  if (!rec.reveal) return;
  const r = rec.img.getBoundingClientRect();
  const visible = r.width >= MIN_DISPLAY && r.bottom > 0 && r.top < innerHeight && rec.img.isConnected;
  rec.reveal.style.display = visible ? 'flex' : 'none';
  if (!visible) return;
  rec.reveal.style.transform = `translate(${r.left + scrollX + r.width / 2}px, ${r.top + scrollY + r.height / 2}px)`;
}

function position(rec) {
  const r = rec.img.getBoundingClientRect();
  const el = rec.badge;
  if (!el) return;
  const visible =
    enabled && r.width >= MIN_DISPLAY && r.height >= MIN_DISPLAY &&
    r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth &&
    rec.img.isConnected;
  el.style.display = visible ? 'flex' : 'none';
  if (!visible) return;
  el.style.transform = `translate(${r.right + scrollX - el.offsetWidth - 6}px, ${r.top + scrollY + 6}px)`;
}

function repositionAll() {
  rafPending = false;
  for (const rec of tracked.values()) {
    position(rec);
    positionReveal(rec);
  }
}
function scheduleReposition() {
  if (!rafPending) {
    rafPending = true;
    requestAnimationFrame(repositionAll);
  }
}

// ------------------------------------------------------- hover panel -------

let panelEl = null;
let heatEl = null;

async function showHeatmap(rec) {
  if (rec.heat === 'pending') return;
  if (!rec.heat) {
    rec.heat = 'pending';
    try {
      let payload;
      if (rec.url.startsWith('blob:')) {
        const blob = await (await fetch(rec.url)).blob();
        payload = { dataUrl: await blobToDataUrl(blob) };
      } else {
        payload = { url: rec.url };
      }
      const res = await chrome.runtime.sendMessage({ target: 'bg', type: 'explain', ...payload });
      rec.heat = res?.ok ? res : null;
    } catch {
      rec.heat = null;
    }
  }
  if (!rec.heat || rec.heat === 'pending' || !panelEl || panelEl.dataset.for !== String(rec.id)) return;
  const r = rec.img.getBoundingClientRect();
  heatEl = document.createElement('div');
  heatEl.className = 'heat';
  heatEl.style.transform = `translate(${r.left + scrollX}px, ${r.top + scrollY}px)`;
  heatEl.style.width = `${r.width}px`;
  heatEl.style.height = `${r.height}px`;
  const n = rec.heat.n;
  heatEl.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
  for (const p of rec.heat.grid) {
    const cell = document.createElement('div');
    const alpha = Math.abs(p - 0.5) * 0.7;
    cell.style.background = p >= 0.5 ? `rgba(244,63,94,${alpha})` : `rgba(52,211,153,${alpha})`;
    heatEl.appendChild(cell);
  }
  ensureOverlay().appendChild(heatEl);
}

function showPanel(rec) {
  if (!rec.result) return;
  const root = ensureOverlay();
  hidePanel();
  const { p, pRaw, signals = [], freq, engine, ms, model, cached } = rec.result;
  panelEl = document.createElement('div');
  panelEl.className = 'panel';
  const rows = signals
    .map((s) => `<div class="sig ${s.kind}"><span class="k">${s.kind === 'ai' ? '⚠' : s.kind === 'real' ? '✓' : 'ℹ'}</span><div><b>${esc(s.label)}</b><small>${esc(s.detail || '')}</small></div></div>`)
    .join('');
  const freqRow = freq ? `<div class="pr sub"><span>freq analysis (smoothness)</span><span>${(freq.score * 100).toFixed(1)}% AI-like</span></div>` : '';
  panelEl.innerHTML = `
    <div class="ph"><b>RealGuard forensics</b><span>${esc(model || '')}</span></div>
    <div class="meter"><div class="fill" style="width:${Math.round(p * 100)}%"></div></div>
    <div class="pr"><span>AI likelihood</span><b>${(p * 100).toFixed(1)}%</b></div>
    <div class="pr sub"><span>neural net (uncalibrated)</span><span>${(pRaw * 100).toFixed(1)}%</span></div>
    ${freqRow}
    ${rows || '<div class="sig info"><span class="k">ℹ</span><div><b>No provenance metadata</b><small>verdict from pixel analysis alone</small></div></div>'}
    <div class="pf">${engine === 'webgpu' ? '⚡ WebGPU' : '🧮 WASM'} · ${cached ? 'cached' : `${ms}ms`} · on-device</div>`;
  panelEl.dataset.for = String(rec.id);
  root.appendChild(panelEl);
  showHeatmap(rec);
  const r = rec.img.getBoundingClientRect();
  const x = Math.min(r.right + scrollX - 280, scrollX + innerWidth - 296);
  panelEl.style.transform = `translate(${Math.max(scrollX + 8, x)}px, ${r.top + scrollY + 34}px)`;
}
function hidePanel() {
  panelEl?.remove();
  panelEl = null;
  heatEl?.remove();
  heatEl = null;
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

// ---------------------------------------------------------- analysis -------

async function analyzeRecord(rec) {
  rec.status = 'analyzing';
  const url = rec.url;
  try {
    let payload;
    if (url.startsWith('blob:')) {
      const blob = await (await fetch(url)).blob();
      if (blob.size > 20 * 1024 * 1024) throw new Error('image too large');
      payload = { dataUrl: await blobToDataUrl(blob) };
    } else {
      payload = { url };
    }
    let res = byUrl.get(url);
    if (!res) {
      try {
        res = await chrome.runtime.sendMessage({ target: 'bg', type: 'analyze', ...payload });
      } catch (e) {
        // Fallback: if fetch fails (e.g. network disabled), extract pixel
        // data from the already-loaded <img> element via canvas.
        if (rec.img && rec.img.tagName === 'IMG' && rec.img.complete) {
          const dataUrl = imgToDataUrl(rec.img);
          if (dataUrl) {
            res = await chrome.runtime.sendMessage({ target: 'bg', type: 'analyze', dataUrl });
          }
        }
        if (!res) throw e;
      }
    }
    if (!res?.ok) throw new Error(res?.error || 'no response');
    byUrl.set(url, res);
    rec.result = res;
    rec.status = 'done';
    rec.img.setAttribute('data-realguard-score', res.p.toFixed(4));
    const [verdict] = labelFor(res.p);
    rec.img.setAttribute('data-realguard-verdict', verdict);
    if (typeof res.logit === 'number') rec.img.setAttribute('data-realguard-logit', res.logit.toFixed(4));
  } catch (e) {
    rec.status = 'error';
    rec.error = String(e?.message || e);
    rec.img.setAttribute('data-realguard-error', rec.error);
  }
  paintBadge(rec);
}

/**
 * Extract pixel data from an already-loaded <img> element via canvas.
 * Used as fallback when fetch() fails (e.g. network disabled during eval).
 * May return null for cross-origin images that taint the canvas.
 */
function imgToDataUrl(img) {
  try {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    if (!canvas.width || !canvas.height) return null;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.9);
  } catch {
    // Cross-origin images will throw a SecurityError (canvas tainted).
    return null;
  }
}
function blobToDataUrl(blob) {
  return new Promise((res, rej) => {
    const fr = new FileReader();
    fr.onload = () => res(fr.result);
    fr.onerror = rej;
    fr.readAsDataURL(blob);
  });
}

// ---------------------------------------------------------- discovery ------

function consider(img) {
  if (tracked.has(img)) return;
  const start = () => {
    if (tracked.has(img)) return;
    if (img.naturalWidth < MIN_NATURAL || img.naturalHeight < MIN_NATURAL) return;
    const url = img.currentSrc || img.src;
    if (!url || url.startsWith('chrome')) return;
    const rec = { img, url, id: ++recId, status: 'pending', result: null, badge: null, heat: null };
    tracked.set(img, rec);
    rec.badge = makeBadge(rec);
    position(rec);
    const cached = byUrl.get(url);
    if (cached) {
      rec.result = cached;
      rec.status = 'done';
      paintBadge(rec);
    } else {
      analyzeRecord(rec);
    }
  };
  if (img.complete && img.naturalWidth > 0) start();
  else img.addEventListener('load', start, { once: true });
}

/** Collect <img> elements including those inside open shadow roots. */
function* allImages(root = document) {
  const walker = root.querySelectorAll ? root.querySelectorAll('*') : [];
  if (root === document) yield* document.images;
  for (const el of walker) {
    if (el.shadowRoot) {
      for (const img of el.shadowRoot.querySelectorAll('img')) yield img;
      yield* allImages(el.shadowRoot);
    }
  }
}

const bgChecked = new WeakSet();

/** Track large elements rendered via CSS background-image (badge only, no blur). */
function scanBackgrounds() {
  const els = document.querySelectorAll('body *');
  const cap = Math.min(els.length, 5000);
  for (let i = 0; i < cap; i++) {
    const el = els[i];
    if (bgChecked.has(el) || tracked.has(el) || el instanceof HTMLImageElement) continue;
    bgChecked.add(el);
    const r = el.getBoundingClientRect();
    if (r.width < 100 || r.height < 100) continue;
    const bg = getComputedStyle(el).backgroundImage;
    const m = bg && bg !== 'none' ? bg.match(/url\(["']?(https?:[^"')]+|data:image[^"')]+)["']?\)/) : null;
    if (!m) continue;
    const rec = { img: el, url: m[1], id: ++recId, status: 'pending', result: null, badge: null, heat: null, isBg: true };
    tracked.set(el, rec);
    rec.badge = makeBadge(rec);
    position(rec);
    const cached = byUrl.get(rec.url);
    if (cached) {
      rec.result = cached;
      rec.status = 'done';
      paintBadge(rec);
    } else {
      analyzeRecord(rec);
    }
  }
}

function sweep() {
  if (!enabled) return;
  for (const img of allImages()) consider(img);
  scanBackgrounds();
}

function detach(img) {
  const rec = tracked.get(img);
  if (rec) {
    rec.badge?.remove();
    rec.reveal?.remove();
    if (rec.blurred) {
      rec.img.style.setProperty('filter', rec.prevFilter || '');
      if (!rec.prevFilter) rec.img.style.removeProperty('filter');
    }
    tracked.delete(img);
  }
}

// ------------------------------------------------------------- wiring -------

let sweepTimer = null;
const scheduleSweep = () => {
  clearTimeout(sweepTimer);
  sweepTimer = setTimeout(sweep, 250);
};

const mo = new MutationObserver((muts) => {
  let dirty = false;
  for (const m of muts) {
    if (m.type === 'attributes' && m.target instanceof HTMLImageElement) {
      const rec = tracked.get(m.target);
      if (rec && (m.target.currentSrc || m.target.src) !== rec.url) {
        detach(m.target);
      }
      dirty = true;
    } else if (m.addedNodes.length || m.removedNodes.length) {
      for (const n of m.removedNodes) {
        if (n instanceof HTMLImageElement) detach(n);
        else if (n.querySelectorAll) n.querySelectorAll('img').forEach(detach);
      }
      dirty = true;
    }
  }
  if (dirty) {
    scheduleSweep();
    scheduleReposition();
  }
});

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type === 'analyze-src' && msg.srcUrl) {
    for (const img of document.images) {
      if ((img.currentSrc || img.src) === msg.srcUrl) consider(img);
    }
    sendResponse({ ok: true });
  } else if (msg?.type === 'page-stats') {
    let ai = 0, real = 0, unsure = 0, pending = 0;
    for (const rec of tracked.values()) {
      if (rec.status !== 'done') { pending++; continue; }
      const p = rec.result.p;
      if (p >= AI_THRESHOLD) ai++;
      else if (p <= REAL_THRESHOLD) real++;
      else unsure++;
    }
    sendResponse({ ok: true, total: tracked.size, ai, real, unsure, pending });
  } else if (msg?.type === 'rescan') {
    for (const img of [...tracked.keys()]) detach(img);
    byUrl.clear();
    sweep();
    sendResponse({ ok: true });
  } else if (msg?.type === 'set-enabled') {
    enabled = msg.enabled;
    if (!enabled) for (const img of [...tracked.keys()]) detach(img);
    else sweep();
    sendResponse({ ok: true });
  } else if (msg?.type === 'set-blur') {
    blurAI = msg.blurAI;
    for (const rec of tracked.values()) applyBlur(rec);
    scheduleReposition();
    sendResponse({ ok: true });
  }
  return false;
});

async function init() {
  const { siteDisabled = {}, blurAI: storedBlur } = await chrome.storage.local.get(['siteDisabled', 'blurAI']);
  if (typeof storedBlur === 'boolean') blurAI = storedBlur;
  if (siteDisabled[location.hostname]) enabled = false;
  if (!enabled) return;
  sweep();
  mo.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['src', 'srcset'],
  });
  addEventListener('scroll', scheduleReposition, { passive: true, capture: true });
  addEventListener('resize', scheduleReposition, { passive: true });
  setInterval(scheduleReposition, 1500);
  setInterval(scheduleSweep, 4000);
}
init();

// Styles for the shadow DOM (kept here so content.css stays a no-op shell).
const BADGE_CSS = `
:host { all: initial; }
.badge {
  position: absolute; top: 0; left: 0; display: flex; align-items: center; gap: 5px;
  padding: 3px 9px 3px 7px; border-radius: 999px; cursor: default; user-select: none;
  font: 600 11.5px/1.4 -apple-system, system-ui, "Segoe UI", sans-serif;
  color: #fff; background: rgba(20,22,28,.82); backdrop-filter: blur(6px);
  box-shadow: 0 1px 4px rgba(0,0,0,.35), inset 0 0 0 1px rgba(255,255,255,.14);
  transition: background .25s ease; pointer-events: auto; white-space: nowrap;
}
.badge .dot { width: 7px; height: 7px; border-radius: 50%; background: #9aa4b2; }
.badge.scanning .dot { background: #7dd3fc; animation: pulse 1s ease-in-out infinite; }
.badge.ai { background: linear-gradient(135deg, rgba(190,24,60,.92), rgba(136,19,55,.92)); }
.badge.ai .dot { background: #fda4af; }
.badge.real { background: linear-gradient(135deg, rgba(5,102,54,.9), rgba(6,78,59,.9)); }
.badge.real .dot { background: #6ee7b7; }
.badge.unsure { background: linear-gradient(135deg, rgba(146,101,7,.92), rgba(120,73,10,.92)); }
.badge.unsure .dot { background: #fcd34d; }
.badge.err { opacity: .5; }
@keyframes pulse { 50% { opacity: .25; transform: scale(.7); } }
.panel {
  position: absolute; top: 0; left: 0; width: 280px; padding: 12px 14px;
  border-radius: 14px; background: rgba(17,19,24,.96); color: #e5e9f0;
  font: 12px/1.45 -apple-system, system-ui, "Segoe UI", sans-serif;
  box-shadow: 0 12px 40px rgba(0,0,0,.5), inset 0 0 0 1px rgba(255,255,255,.1);
  pointer-events: none; z-index: 1;
}
.panel .ph { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
.panel .ph b { font-size: 12.5px; }
.panel .ph span { color: #8b93a3; font-size: 10.5px; }
.panel .meter { height: 6px; border-radius: 4px; background: #2a2f3a; overflow: hidden; margin: 2px 0 6px; }
.panel .meter .fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #34d399, #fcd34d, #f43f5e); }
.panel .pr { display: flex; justify-content: space-between; align-items: baseline; margin: 3px 0; }
.panel .pr b { font-size: 15px; }
.panel .pr.sub, .panel .pr.sub span { color: #8b93a3; font-size: 11px; }
.panel .sig { display: flex; gap: 6px; margin: 5px 0; padding: 4px 6px; border-radius: 6px; }
.panel .sig.ai { background: rgba(244,63,94,.1); }
.panel .sig.real { background: rgba(52,211,153,.1); }
.panel .sig.info { background: rgba(125,211,252,.08); }
.panel .sig .k { width: 16px; text-align: center; font-size: 13px; }
.panel .sig b { font-size: 11px; }
.panel .sig small { display: block; color: #8b93a3; font-size: 10px; }
.panel .pf { margin-top: 8px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,.08); color: #8b93a3; font-size: 10.5px; }
.heat {
  position: absolute; top: 0; left: 0; display: grid; gap: 0; pointer-events: none;
  z-index: 0; opacity: .65; border-radius: 4px; overflow: hidden;
}
.reveal {
  position: absolute; top: 0; left: 0; transform-origin: center;
  padding: 6px 14px; border-radius: 8px; cursor: pointer;
  background: rgba(190,24,60,.85); color: #fff; font: 600 12px system-ui;
  pointer-events: auto; display: flex; align-items: center; gap: 4px;
  box-shadow: 0 2px 8px rgba(0,0,0,.3);
}
.reveal:hover { background: rgba(190,24,60,.95); }
`;
