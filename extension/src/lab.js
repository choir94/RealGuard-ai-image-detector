// RealGuard — Forensics Lab: evaluate detector on labeled image folders.
// Click the popup → "Forensics Lab" → drag a folder with real/ and ai/ subfolders.

const THRESHOLD = 0.65;

const $ = (id) => document.getElementById(id);

function log(msg) {
  $('log').textContent += msg + '\n';
  $('log').scrollTop = $('log').scrollHeight;
}

async function analyzeFile(file) {
  const dataUrl = await new Promise((res, rej) => {
    const fr = new FileReader();
    fr.onload = () => res(fr.result);
    fr.onerror = rej;
    fr.readAsDataURL(file);
  });
  const res = await chrome.runtime.sendMessage({ target: 'bg', type: 'analyze', dataUrl });
  return res;
}

async function processFolder(files) {
  const aiFiles = [], realFiles = [];
  for (const f of files) {
    const path = f.webkitRelativePath || f.name;
    if (/\/ai\//i.test(path) || /^ai\//i.test(path)) aiFiles.push(f);
    else if (/\/real\//i.test(path) || /^real\//i.test(path)) realFiles.push(f);
  }
  if (!aiFiles.length && !realFiles.length) {
    log('No labeled images found. Create real/ and ai/ subfolders.');
    return;
  }
  log(`Found ${aiFiles.length} AI images, ${realFiles.length} real images.`);
  log('Analyzing... (this runs through the real extension pipeline)');

  const all = [...aiFiles.map(f => ({ f, label: 'ai' })), ...realFiles.map(f => ({ f, label: 'real' }))];
  const rows = [];
  let correct = 0;

  for (let i = 0; i < all.length; i++) {
    const { f, label } = all[i];
    try {
      const res = await analyzeFile(f);
      if (res?.ok) {
        const verdict = res.p >= THRESHOLD ? 'ai' : 'real';
        const isCorrect = verdict === label;
        if (isCorrect) correct++;
        rows.push({ name: f.name, label, score: res.p, logit: res.logit, verdict, correct: isCorrect });
        $('progress').value = (i + 1) / all.length;
        $('progress-text').textContent = `${i + 1}/${all.length}`;
      } else {
        rows.push({ name: f.name, label, score: null, error: res?.error || 'failed' });
      }
    } catch (e) {
      rows.push({ name: f.name, label, score: null, error: String(e.message) });
    }
  }

  // Metrics
  const scored = rows.filter(r => r.score != null);
  const aiScored = scored.filter(r => r.label === 'ai');
  const realScored = scored.filter(r => r.label === 'real');
  const tpr = aiScored.length ? aiScored.filter(r => r.score >= THRESHOLD).length / aiScored.length : 0;
  const tnr = realScored.length ? realScored.filter(r => r.score < THRESHOLD).length / realScored.length : 0;
  const ba = (tpr + tnr) / 2;

  log(`\n=== Results (threshold ${THRESHOLD}) ===`);
  log(`Scored: ${scored.length}/${rows.length}`);
  log(`TPR: ${(tpr * 100).toFixed(2)}%  TNR: ${(tnr * 100).toFixed(2)}%`);
  log(`Balanced accuracy: ${(ba * 100).toFixed(2)}%`);
  log(`Correct: ${correct}/${scored.length}`);

  // Render table
  const tbody = $('results-body');
  tbody.innerHTML = '';
  for (const r of scored.sort((a, b) => b.score - a.score)) {
    const tr = document.createElement('tr');
    const cells = [r.name, r.label, `${(r.score * 100).toFixed(1)}%`, r.verdict, r.correct ? '✓' : '✗'];
    for (const cell of cells) {
      const td = document.createElement('td');
      td.textContent = cell;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  $('results').style.display = 'block';

  // CSV export
  $('export').onclick = () => {
    const esc = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
    const csv = 'name,label,score,logit,verdict,correct\n' + rows.map(r =>
      [r.name, r.label, r.score ?? '', r.logit ?? '', r.verdict ?? '', r.correct ?? ''].map(esc).join(',')
    ).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'realguard_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };
}

document.addEventListener('DOMContentLoaded', () => {
  $('folder-input').addEventListener('change', (e) => {
    const files = [...e.target.files].filter(f => /\.(png|jpe?g|webp|gif|bmp)$/i.test(f.name));
    if (files.length) processFolder(files);
  });
  // Drag & drop
  const drop = $('drop-zone');
  drop.addEventListener('click', () => $('folder-input').click());
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('over');
    const files = [...e.dataTransfer.files].filter(f => /\.(png|jpe?g|webp|gif|bmp)$/i.test(f.name));
    if (files.length) processFolder(files);
  });
});
