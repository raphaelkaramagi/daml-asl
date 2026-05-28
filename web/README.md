# ASL Alphabet Recognition — Web Demo

Interactive browser-based demo for the ASL Alphabet Recognition project. All ML inference runs entirely client-side using TensorFlow.js and MediaPipe — no backend or server required.

**Live:** [daml-asl.vercel.app](https://daml-asl.vercel.app)

---

## Features

| Section | Description |
|---|---|
| **Live Prediction** | Webcam feed with real-time MediaPipe hand skeleton overlay + ResNet50 and Landmark NN predictions side by side |
| **Image Upload** | Drag-and-drop or click to upload any image for prediction |
| **Model Comparison** | Evaluation banner, architecture, metrics, strengths and weaknesses for each approach |
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

Before the first run (or after retraining), prepare assets from the **repo root**:

```bash
conda activate daml-asl

# Optional: retrain with improved pipeline
python scripts/reextract_landmarks.py      # improved landmark CSV
python scripts/train_landmark_nn.py        # landmark classifier
python scripts/generate_hand_crops.py      # hand-cropped ResNet training set
python scripts/train_resnet_improved.py    # 3-phase ResNet training (~4-8 hrs GPU)

# Convert and export for web
python scripts/convert_models.py           # → web/public/models/resnet-graph/ (float32 graph)
python scripts/evaluate_models.py        # → results/evaluation_results.json (+ web sync)
python scripts/update_results_doc.py     # → docs/RESULTS.md
python scripts/prepare_samples.py        # → web/public/samples/
```

Landmark-only retrain (no ResNet work):

```bash
python scripts/run_landmark_pipeline.py
```

### Webcam inference notes

**Do not mirror webcam pixels before feature extraction.** Training photos are unmirrored; mirroring the video/canvas before MediaPipe or ResNet changes hand geometry relative to training and degrades Landmark NN live accuracy. The UI mirrors the preview with CSS (`scaleX(-1)`) and flips landmark x-coordinates in the overlay only.

Gallery/upload inference:

- Single-pass IMAGE-mode detection (`detectHandFromImage`)
- Small images (&lt;300px) upscaled before MediaPipe detection
- Configurable detection confidence (Settings panel, default **0.3**)

### Model files (committed)

| Path | Format | Size | Notes |
|---|---|---|---|
| `models/landmark-nn/` | TF.js layers model | ~72 KB | Original weights (~18K params) for live parity |
| `models/resnet-graph/` | TF.js graph model | ~91 MB | Retrained ResNet50, float32 (browser-compatible) |
| `models/preprocessing.json` | JSON | <1 KB | StandardScaler mean/scale + class names |

---

## Tech Stack

| Library | Purpose |
|---|---|
| Next.js 16 (App Router, TypeScript) | Framework, static export |
| Tailwind CSS | Styling |
| Framer Motion | Animations |
| TensorFlow.js | In-browser model inference |
| MediaPipe Tasks Vision | Hand landmark detection |
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
│   ├── EvaluationSummary.tsx   # Deployed model evaluation banner
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
- ResNet50 graph model is ~91 MB float32 (reliable browser loading); first load may take a moment on slow connections
- Detection confidence defaults to **0.3**; adjustable in the Settings panel
- Small images (&lt;300px) are upscaled before MediaPipe detection on upload/gallery paths
