# RealGuard — AI Image Detector for Chrome

Detect AI-generated images directly in your browser. 100% local, 100% private.

## What It Does

RealGuard is a Chrome Extension (Manifest V3) that automatically scans images on any webpage and displays a confidence score indicating whether each image is AI-generated or real.

- **No cloud inference** — all processing happens in your browser
- **No external APIs** — no data ever leaves your device
- **No backend servers** — pure browser extension
- **Works offline** after initial model download

## How It Works

1. Install the extension
2. Visit any webpage with images
3. RealGuard automatically scans each image
4. A badge appears on each image showing:
   - **AI 92%** (red badge) — likely AI-generated
   - **Real 81%** (green badge) — likely a real photo

## Technology

- **Model**: [CommunityForensics DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT)
  - Vision Transformer (ViT-Small), 21.8M parameters
  - Trained on 2.7M samples across 4,803 AI generators
  - Published at CVPR 2025
  - MIT License
- **Inference**: [Transformers.js](https://github.com/huggingface/transformers.js) v4
  - WebGPU acceleration with WASM fallback
  - INT8 quantized model (~22MB)
- **Extension**: Chrome Manifest V3
  - Content script for image detection
  - Service worker for model management
  - Browser cache for offline model storage

## Installation

### From Source (Build Instructions)

```bash
# Clone the repository
git clone https://github.com/yourusername/realguard-ai-image-detector.git
cd realguard-ai-image-detector

# Install dependencies
npm install

# Build the extension
npm run build
```

This creates a `dist/` folder with the complete extension.

### Load in Chrome

1. Open `chrome://extensions`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `dist/` folder
5. The extension will download the AI model on first run (~22MB, one-time)
6. After model download, the extension works fully offline

### Verification

- The extension icon in the toolbar shows model status
- Click the icon to see scanning statistics
- Visit any webpage with images — badges appear automatically

## Build Requirements

- Node.js 18+
- npm

## Reproducibility

The build is fully reproducible:
1. `npm install` installs exact dependencies (package-lock.json pinned)
2. `npm run build` creates the extension from source
3. The model is downloaded from HuggingFace Hub during first extension load
4. After download, the model is cached in the browser and works offline

## Privacy

- **No image data is ever transmitted** to any server
- All inference runs locally using WebGPU or WASM
- The only network request is the one-time model download from HuggingFace Hub
- After download, the extension works with internet disabled

## License

MIT License — see [LICENSE](LICENSE)

## Model Credits

- **CommunityForensics DeepfakeDet-ViT** by Jeongsoo Park and Andrew Owens, University of Michigan
- Paper: "Community Forensics: Using Thousands of Generators to Train Fake Image Detectors" (CVPR 2025)
- arXiv: [2411.04125](https://arxiv.org/abs/2411.04125)
- Model: [HuggingFace Hub](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT)

## Confidence Threshold

The extension uses a **65% confidence threshold** as required by the bounty specification:
- If the model's AI probability ≥ 65%, the image is classified as AI-generated
- If the model's AI probability < 65%, the image is classified as real
