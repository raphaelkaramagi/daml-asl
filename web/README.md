# ASL Alphabet Recognition — Web Demo

Interactive browser-based demo for the ASL Alphabet Recognition project. All ML inference runs entirely client-side using TensorFlow.js and MediaPipe — no backend or server required.

**Live:** [daml-asl.vercel.app](https://daml-asl.vercel.app)

---

## Features

| Section | Description |
|---|---|
| **Live Prediction** | Webcam feed with real-time MediaPipe hand skeleton overlay + ResNet50 and Landmark NN predictions side by side |
| **Image Upload** | Drag-and-drop or click to upload any image for prediction |
| **Training Replay** | Animated epoch-by-epoch charts replaying the real training metrics from both models |
| **Micro Training** | Train a small neural network live in the browser with TF.js |
| **Model Comparison** | Architecture, metrics, strengths and weaknesses for each approach |
| **Sample Gallery** | Browse all 29 ASL classes, click any sample image to get predictions |
| **Settings** | Toggle models on/off, adjust MediaPipe detection confidence |

---

## Development

```bash
cd web
npm install
npm run dev   # http://localhost:3000
```

### Requirements
- Node.js 18+
- npm 9+

---

## Preparing Models & Assets

Before the first run (or after retraining), prepare the static assets from the **repo root**:

```bash
# 1. Convert trained Keras models to TF.js format
#    Outputs: web/public/models/landmark-nn/ and web/public/models/resnet-graph/
python scripts/convert_models.py

# 2. Resize and copy sample images for the gallery
#    Outputs: web/public/samples/ + web/public/samples/manifest.json
python scripts/prepare_samples.py

# 3. Export training metrics for the Training Replay section
#    Output: web/public/training-data.json
python scripts/extract_training_data.py
```

> These scripts require the Python environment from the main project (`conda activate daml-asl`).

### Model files (already committed)

The converted model files are committed to the repo under `web/public/models/`:

| Path | Format | Size | Notes |
|---|---|---|---|
| `models/landmark-nn/` | TF.js layers model | ~72 KB weights | Used for landmark NN inference |
| `models/resnet-graph/` | TF.js graph model | ~23 MB (uint8 quantized) | Used for ResNet50 inference |
| `models/preprocessing.json` | JSON | <1 KB | StandardScaler mean/scale + class names |

---

## Tech Stack

| Library | Purpose |
|---|---|
| Next.js 16 (App Router, TypeScript) | Framework, static export |
| Tailwind CSS | Styling |
| Framer Motion | Animations |
| TensorFlow.js | In-browser model inference and micro training |
| MediaPipe Tasks Vision | Hand landmark detection |
| Recharts | Training curve charts |
| Zustand | Global state management |

---

## Architecture

```
web/src/
├── app/
│   ├── layout.tsx       # Root layout, metadata, Open Graph, PWA manifest
│   ├── page.tsx         # Single-page app with sticky nav
│   └── globals.css
├── components/
│   ├── Hero.tsx                # Landing, model loading progress
│   ├── LivePredictor.tsx       # Orchestrates webcam + upload + predictions
│   ├── WebcamFeed.tsx          # Camera stream + landmark overlay + prediction loop
│   ├── ImageUploader.tsx       # Drag-and-drop image input
│   ├── LandmarkVisualizer.tsx  # Canvas-based hand skeleton renderer
│   ├── PredictionDisplay.tsx   # Side-by-side model results + confidence bars
│   ├── TrainingReplay.tsx      # Animated training charts with playback controls
│   ├── MicroTraining.tsx       # In-browser TF.js training UI
│   ├── ModelComparison.tsx     # Architecture cards and metrics
│   ├── SampleGallery.tsx       # Class grid + image selector + predictions
│   ├── SettingsPanel.tsx       # Slide-out settings drawer
│   └── ui/                     # Card, Section, ConfidenceBar, LoadingSpinner
├── hooks/
│   ├── useModels.ts     # Async model loading orchestration (MediaPipe + TF.js)
│   ├── usePrediction.ts # Image/canvas → hand detection → model inference pipeline
│   └── useWebcam.ts     # Camera stream lifecycle
├── lib/
│   ├── models.ts        # loadLandmarkModel, loadResnetModel, predict functions
│   ├── landmarks.ts     # MediaPipe HandLandmarker init + detection
│   ├── micro-trainer.ts # TF.js in-browser training logic
│   └── constants.ts     # CLASS_NAMES, MODEL_PATHS, HAND_CONNECTIONS
└── store/
    └── app-store.ts     # Zustand store (model loaded state, settings)
```

---

## Deployment (Vercel)

### 1. Push to GitHub

Make sure everything including `web/public/models/` is committed and pushed:

```bash
git add -f web/public/models/
git add -A
git commit -m "your message"
git push
```

### 2. Import on Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New Project**
2. Import the `raphaelkaramagi/daml-asl` repository
3. Set **Root Directory** to `web`
4. Framework: **Next.js** (auto-detected)
5. Click **Deploy**

The `vercel.json` in `web/` sets 1-year immutable CDN caching on `/models/` and `/samples/`, so model files are cached after the first load.

### 3. Build locally

```bash
npm run build   # Outputs static site to out/
```

---

## Notes

- All inference is client-side — no API routes or server functions are used
- MediaPipe WASM and the hand landmark model are loaded from CDN on first use
- The ResNet50 graph model (~23 MB) is quantized to uint8, reducing the original 208 MB by ~9×
- Detection confidence defaults to 0.3; adjustable in the Settings panel
- Small images (< 300px) are upscaled before MediaPipe detection to improve reliability
