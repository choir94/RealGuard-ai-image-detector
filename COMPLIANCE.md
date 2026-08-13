# RealGuard — Bounty Compliance Checklist

Based on: https://poidh.xyz/arbitrum/bounty/323

## Requirements (must have)

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| 1 | Fully open source under MIT License | ✅ | `LICENSE` file, MIT |
| 2 | Native Manifest V3 Chrome extension | ✅ | `manifest.json` — `"manifest_version": 3` |
| 3 | All inference local (WebGPU/WASM/WebGL) | ✅ | Transformers.js with WebGPU + WASM fallback, no cloud calls |
| 4 | One-time model download, then offline | ✅ | Model from HF Hub, cached via Browser Cache API, no further downloads |
| 5 | Automatically analyze images on webpages | ✅ | Content script detects `<img>`, MutationObserver for dynamic content |
| 6 | Display confidence score for every image | ✅ | Badge overlay shows AI/Real + percentage |
| 7 | Complete build and installation instructions | ✅ | `README.md` with full build + install steps |
| 8 | Fully reproducible from source | ✅ | `npm install && npm run build` → `dist/`, `package-lock.json` pinned |

## Rules (may not do)

| # | Rule | Status | Evidence |
|---|------|--------|----------|
| 1 | No cloud inference | ✅ | All inference via Transformers.js local |
| 2 | No sending image data to external services | ✅ | Image processed in-browser, no network calls for inference |
| 3 | No local backend (Python/Node/Flask) | ✅ | Pure browser extension, no server |
| 4 | No additional model downloads after setup | ✅ | Single model download on first run, cached after |
| 5 | No hardcoded benchmark hashes/lookup tables | ✅ | Pure ML inference, no hash matching |
| 6 | No circumventing evaluation | ✅ | Standard detection pipeline |

## Evaluation Criteria

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | ≥75% balanced accuracy | 🎯 | Model: 97.2% claimed, 4,803 generators coverage |
| 2 | 65% confidence threshold | ✅ | Hardcoded in `background.js`: `const THRESHOLD = 0.65` |
| 3 | Clean Chrome, fresh profile | ✅ | No external dependencies, self-contained |
| 4 | Internet disabled after model download | ✅ | Model cached in Browser Cache API |
| 5 | Localhost APIs blocked | ✅ | No localhost calls anywhere in code |

## What Goes in the GitHub Repo

### Must be tracked (source files):
- `manifest.json` — MV3 manifest
- `background.js` — Service worker (model loading, inference)
- `content.js` — Content script (image detection, badge overlay)
- `overlay.css` — Badge styling
- `popup/popup.html` — Extension popup UI
- `popup/popup.js` — Popup logic
- `scripts/build.js` — Build script (reproducible)
- `package.json` — npm dependencies
- `package-lock.json` — Pinned dependency versions
- `LICENSE` — MIT License
- `README.md` — Build & installation instructions
- `.gitignore` — Ignore node_modules, dist, etc.

### Must NOT be tracked (generated):
- `node_modules/` — npm install creates this
- `dist/` — npm run build creates this
- `icons/*.png` — Build script generates these

### Model weights:
- NOT included in repo — downloaded from HuggingFace Hub on first run
- URL: `https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT`
- Cached in browser after download, works offline
