'use client';

import { useState, useCallback } from 'react';
import {
  predictWithLandmarkNN,
  predictWithResnet,
  getTopPredictions,
  loadPreprocessing,
  isLandmarkModelLoaded,
  isResnetModelLoaded,
  type Prediction,
} from '@/lib/models';
import { detectHandFromImage, type HandDetectionResult } from '@/lib/landmarks';
import { CLASS_NAMES } from '@/lib/constants';
import { useAppStore } from '@/store/app-store';

export interface PredictionResult {
  resnet?: Prediction;
  landmark?: Prediction;
  handDetection: HandDetectionResult | null;
  resnetTop3?: { label: string; confidence: number }[];
  landmarkTop3?: { label: string; confidence: number }[];
}

export function usePrediction() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);

  const predict = useCallback(
    async (source: HTMLImageElement | HTMLCanvasElement) => {
      setLoading(true);

      try {
        const { enableResnet, enableLandmark } = useAppStore.getState();
        const resnetReady = isResnetModelLoaded();
        const landmarkReady = isLandmarkModelLoaded();

        const srcW = source instanceof HTMLImageElement ? source.naturalWidth : source.width;
        const srcH = source instanceof HTMLImageElement ? source.naturalHeight : source.height;

        // Upscale small images before MediaPipe detection for better reliability
        let detectionSource: HTMLImageElement | HTMLCanvasElement = source;
        const MIN_DIM = 300;
        if (srcW < MIN_DIM || srcH < MIN_DIM) {
          const scale = Math.max(MIN_DIM / srcW, MIN_DIM / srcH, 1);
          const upCanvas = document.createElement('canvas');
          upCanvas.width = Math.round(srcW * scale);
          upCanvas.height = Math.round(srcH * scale);
          const upCtx = upCanvas.getContext('2d')!;
          upCtx.drawImage(source, 0, 0, upCanvas.width, upCanvas.height);
          detectionSource = upCanvas;
        }

        const handResult = detectHandFromImage(detectionSource);

        let resnetPred: Prediction | undefined;
        let landmarkPred: Prediction | undefined;
        let resnetTop3: { label: string; confidence: number }[] | undefined;
        let landmarkTop3: { label: string; confidence: number }[] | undefined;

        if (enableResnet && resnetReady) {
          const canvas = document.createElement('canvas');
          canvas.width = srcW;
          canvas.height = srcH;
          const ctx = canvas.getContext('2d')!;
          ctx.drawImage(source, 0, 0);
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          resnetPred = predictWithResnet(imageData);
          resnetTop3 = getTopPredictions(resnetPred.allConfidences, [...CLASS_NAMES]);
        }

        if (enableLandmark && landmarkReady && handResult) {
          landmarkPred = predictWithLandmarkNN(handResult.features);
          const preprocessing = await loadPreprocessing();
          landmarkTop3 = getTopPredictions(landmarkPred.allConfidences, preprocessing.classes);
        }

        setResult({
          resnet: resnetPred,
          landmark: landmarkPred,
          handDetection: handResult,
          resnetTop3,
          landmarkTop3,
        });
      } catch (err) {
        console.error('Prediction error:', err);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  const clear = useCallback(() => setResult(null), []);

  return { result, loading, predict, clear };
}
