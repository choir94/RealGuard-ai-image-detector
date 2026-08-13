# RealGuard v2 — Bounty Compliance Checklist

Based on: https://poidh.xyz/arbitrum/bounty/323

## Requirements (must have)

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | Fully open source under MIT License | ✅ | `LICENSE` file, MIT |
| 2 | Native Manifest V3 Chrome extension | ✅ | `extension/manifest.json` — `"manifest_version": 3` |
| 3 | All inference local (WebGPU/WASM/WebGL) | ✅ | ONNX Runtime Web, WebGPU + WASM fallback |
| 4 | One-time model download, then offline | ✅ | Model from HF Hub, Cache Storage API, SHA-256 verified |
| 5 | Automatically analyze images on webpages | ✅ | Content script: `<img>`, shadow DOM, CSS backgrounds, iframes, MutationObserver |
| 6 | Display confidence score for every image | ✅ | Shadow DOM badge overlay: AI/Real + %, hover forensics panel |
| 7 | Complete build and installation instructions | ✅ | `README.md` with `npm ci && npm run build` |
| 8 | Fully reproducible from source | ✅ | `package-lock.json` pinned, `build.mjs` with esbuild bundling |

## Rules (may not do)

| # | Rule | Status | Evidence |
|---|------|--------|----------|
| 1 | No cloud inference | ✅ | All inference via ONNX Runtime Web local |
| 2 | No sending image data to external services | ✅ | Image processed in-browser, fetch with credentials:'omit' |
| 3 | No local backend (Python/Node/Flask) | ✅ | Pure browser extension, no server |
| 4 | No additional model downloads after setup | ✅ | Single model download on first run, cached in Cache Storage |
| 5 | No hardcoded benchmark hashes/lookup tables | ✅ | Pure ML inference + metadata forensics, no hash matching |
| 6 | No circumventing evaluation | ✅ | Standard detection pipeline, data-realguard-* attributes for evaluation |

## Evaluation Criteria

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | ≥75% balanced accuracy | 🎯 | Model: CommunityForensics ViT-S/16, 4,803 generators + freq analysis + metadata |
| 2 | 65% confidence threshold | ✅ | `const AI_THRESHOLD = 0.65` in content.js, Platt calibration in models.json |
| 3 | Clean Chrome, fresh profile | ✅ | No external dependencies, self-contained |
| 4 | Internet disabled after model download | ✅ | Model cached in Cache Storage API |
| 5 | Localhost APIs blocked | ✅ | No localhost calls anywhere in code |

## Architecture (v2 upgrade from v1)

### What changed from v1:
1. **Transformers.js → ONNX Runtime Web direct** — .bundle build for service worker compatibility
2. **Pure classifier → 5-signal hybrid pipeline** — neural + frequency + C2PA + metadata + calibration
3. **Basic badge → Shadow DOM overlay** — auto-blur, forensics panel, heatmap
4. **`<img>` only → Full coverage** — shadow DOM, CSS backgrounds, iframes, MutationObserver
5. **No tests → Full test suite + CI/CD** — unit tests, benchmark harness, GitHub Actions
6. **Raw sigmoid → Platt calibration** — threshold 0.65 optimized to BA-optimal
7. **No freq analysis → Novel differentiator** — frequency-domain analysis that Detectra doesn't have

### Pivot differentiator vs Detectra:
- **Frequency-domain analysis** (freq.js) — AI images lack high-frequency texture that real photos have
- Detectra only has neural + metadata. RealGuard adds a third independent signal.

## What Goes in the GitHub Repo

### Source files (tracked):
- `extension/manifest.json` — MV3 manifest
- `extension/src/background.js` — Service worker
- `extension/src/content.js` — Content script with Shadow DOM overlay
- `extension/src/popup.js` — Popup logic
- `extension/src/lab.js` — Forensics Lab
- `extension/src/lib/engine.js` — ONNX Runtime Web inference engine
- `extension/src/lib/meta.js` — Metadata forensics
- `extension/src/lib/freq.js` — Frequency-domain analysis
- `extension/static/models.json` — Model config + calibration
- `extension/static/popup.html` — Popup UI
- `extension/static/lab.html` — Lab UI
- `extension/static/content.css` — No-op shell
- `build.mjs` — Build script (esbuild)
- `package.json` / `package-lock.json` — Dependencies
- `LICENSE` — MIT
- `README.md` — Full documentation
- `tests/meta.test.mjs` — Unit tests
- `tools/bench.mjs` — Benchmark harness
- `eval/calibrate.py` — Calibration script
- `.github/workflows/ci.yml` — CI/CD
- `.gitignore`

### Generated (not tracked):
- `node_modules/` — npm ci creates this
- `extension/dist/` — npm run build creates this
