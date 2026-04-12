# ASL Alphabet Recognition - Web Demo

Interactive web demo for the ASL Alphabet Recognition project. All inference runs entirely in the browser using TensorFlow.js and MediaPipe.

## Features

- **Live Prediction** - Upload images or use webcam for real-time ASL hand sign classification
- **Two Models** - Side-by-side comparison of ResNet50 (pixel-based) and Landmark NN (geometry-based)
- **Training Replay** - Animated visualization of the real training process with actual metrics
- **Micro Training** - Train a small neural network live in your browser
- **Model Comparison** - Detailed architecture and performance comparison
- **Sample Gallery** - Browse and predict on sample images from each class

## Development

```bash
npm install
npm run dev
```

## Preparing Models

Run the Python scripts from the project root to convert models and prepare assets:

```bash
python scripts/convert_models.py       # Convert Keras models to TF.js format
python scripts/prepare_samples.py      # Resize and copy sample images
python scripts/extract_training_data.py # Extract training metrics JSON
```

## Deployment

Deployed via Vercel. Set the root directory to `web/` in Vercel project settings.

```bash
npm run build   # Static export to out/
```

## Tech Stack

Next.js, TypeScript, Tailwind CSS, TensorFlow.js, MediaPipe, Recharts, Framer Motion, Zustand
