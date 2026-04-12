'use client';

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Section from './ui/Section';
import Card from './ui/Card';
import WebcamFeed from './WebcamFeed';
import ImageUploader from './ImageUploader';
import PredictionDisplay from './PredictionDisplay';
import LandmarkVisualizer from './LandmarkVisualizer';
import { usePrediction, type PredictionResult } from '@/hooks/usePrediction';

type InputMode = 'webcam' | 'upload';

export default function LivePredictor() {
  const [mode, setMode] = useState<InputMode>('upload');
  const { result, loading, predict, clear } = usePrediction();
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [webcamResult, setWebcamResult] = useState<PredictionResult | null>(null);

  const handleImage = useCallback(
    (img: HTMLImageElement) => {
      predict(img);
    },
    [predict]
  );

  const handleWebcamPrediction = useCallback((r: PredictionResult) => {
    setWebcamResult(r);
  }, []);

  const activeResult = mode === 'webcam' ? webcamResult : result;

  return (
    <Section
      id="predictor"
      title="Live Prediction"
      subtitle="Upload an image or use your webcam to classify ASL hand signs with both models simultaneously."
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-4">
          <div className="flex gap-2 p-1 bg-zinc-800/50 rounded-lg">
            {(['upload', 'webcam'] as const).map((m) => (
              <button
                key={m}
                onClick={() => {
                  setMode(m);
                  if (m === 'upload') {
                    setWebcamResult(null);
                  } else {
                    clear();
                    setPreviewUrl(null);
                  }
                }}
                className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                  mode === m
                    ? 'bg-zinc-700 text-white'
                    : 'text-zinc-400 hover:text-zinc-300'
                }`}
              >
                {m === 'upload' ? 'Upload Image' : 'Webcam'}
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            {mode === 'upload' ? (
              <motion.div
                key="upload"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
              >
                <ImageUploader
                  onImage={handleImage}
                  previewUrl={previewUrl}
                  setPreviewUrl={(url) => {
                    setPreviewUrl(url);
                    if (!url) clear();
                  }}
                />
              </motion.div>
            ) : (
              <motion.div
                key="webcam"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
              >
                <WebcamFeed onPrediction={handleWebcamPrediction} />
              </motion.div>
            )}
          </AnimatePresence>

          {activeResult?.handDetection && mode === 'upload' && previewUrl && (
            <Card>
              <p className="text-xs uppercase tracking-wider text-zinc-500 mb-3">
                Detected Hand Landmarks
              </p>
              <div className="relative w-full aspect-square max-w-[200px] mx-auto bg-zinc-800 rounded-lg overflow-hidden">
                <LandmarkVisualizer
                  landmarks={activeResult.handDetection.landmarks}
                  width={200}
                  height={200}
                  showLabels
                />
              </div>
            </Card>
          )}
        </div>

        <div>
          <PredictionDisplay result={activeResult ?? null} loading={loading} />
        </div>
      </div>

      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
        {[
          {
            title: 'ResNet50',
            desc: 'Analyzes raw 96x96 pixel images',
            detail: 'Transfer learning from ImageNet',
          },
          {
            title: 'MediaPipe',
            desc: '21 hand landmarks detected',
            detail: '63 features (x, y, z per landmark)',
          },
          {
            title: 'Landmark NN',
            desc: 'Classifies hand geometry',
            detail: 'Dense 128 → 64 → 29 network',
          },
        ].map((item) => (
          <Card key={item.title}>
            <p className="text-sm font-semibold text-zinc-300">{item.title}</p>
            <p className="text-xs text-zinc-500 mt-1">{item.desc}</p>
            <p className="text-[10px] text-zinc-600 mt-0.5">{item.detail}</p>
          </Card>
        ))}
      </div>
    </Section>
  );
}
