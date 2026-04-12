'use client';

import { useState, useCallback } from 'react';
import { predictWithLandmarkNN, predictWithResnet, getTopPredictions, type Prediction } from '@/lib/models';
import { detectHandFromImage, type HandDetectionResult } from '@/lib/landmarks';
import { CLASS_NAMES } from '@/lib/constants';
import { useAppStore } from '@/store/app-store';

export interface PredictionResult {
  resnet?: Prediction;
  landmark?: Prediction;
  handDetection: HandDetectionResult | null;
  resnetTop3?: { label: string; confidence: number }[];
  landmarkTop3?: { label: string; confidence: number }[];
  inputImageUrl?: string;
}

export function usePrediction() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const { enableResnet, enableLandmark, resnetLoaded, landmarkLoaded } = useAppStore();

  const predict = useCallback(
    async (source: HTMLImageElement | HTMLCanvasElement) => {
      setLoading(true);

      try {
        const handResult = detectHandFromImage(source);

        let resnetPred: Prediction | undefined;
        let landmarkPred: Prediction | undefined;
        let resnetTop3: { label: string; confidence: number }[] | undefined;
        let landmarkTop3: { label: string; confidence: number }[] | undefined;

        if (enableResnet && resnetLoaded) {
          const canvas = document.createElement('canvas');
          canvas.width = source instanceof HTMLImageElement ? source.naturalWidth : source.width;
          canvas.height = source instanceof HTMLImageElement ? source.naturalHeight : source.height;
          const ctx = canvas.getContext('2d')!;
          ctx.drawImage(source, 0, 0);
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          resnetPred = predictWithResnet(imageData);
          resnetTop3 = getTopPredictions(resnetPred.allConfidences, [...CLASS_NAMES]);
        }

        if (enableLandmark && landmarkLoaded && handResult) {
          landmarkPred = predictWithLandmarkNN(handResult.features);
          const classes = (await import('@/lib/models').then((m) => m.loadPreprocessing())).classes;
          landmarkTop3 = getTopPredictions(landmarkPred.allConfidences, classes);
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
    [enableResnet, enableLandmark, resnetLoaded, landmarkLoaded]
  );

  const clear = useCallback(() => setResult(null), []);

  return { result, loading, predict, clear };
}
