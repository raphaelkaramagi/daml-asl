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
import { detectHandFromImage, setRunningMode } from '@/lib/landmarks';
import {
  canvasToResnetImageData,
  drawSourceToCanvas,
  getPaddedBBox,
  SMALL_HAND_AREA_THRESHOLD,
} from '@/lib/image-utils';
import { useAppStore } from '@/store/app-store';

export interface PredictionResult {
  resnet?: Prediction;
  landmark?: Prediction;
  handDetection: import('@/lib/landmarks').HandDetectionResult | null;
  resnetTop3?: { label: string; confidence: number }[];
  landmarkTop3?: { label: string; confidence: number }[];
  handTooSmall?: boolean;
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

        setRunningMode('IMAGE');
        const handResult = detectHandFromImage(source);
        const canvas = drawSourceToCanvas(source);

        let handTooSmall = false;
        if (handResult?.landmarks) {
          const bbox = getPaddedBBox(handResult.landmarks);
          handTooSmall = bbox.area < SMALL_HAND_AREA_THRESHOLD;
        }

        let resnetPred: Prediction | undefined;
        let landmarkPred: Prediction | undefined;
        let resnetTop3: { label: string; confidence: number }[] | undefined;
        let landmarkTop3: { label: string; confidence: number }[] | undefined;

        const preprocessing = await loadPreprocessing();
        const classes = preprocessing.classes;

        if (enableResnet && resnetReady) {
          const imageData = canvasToResnetImageData(
            canvas,
            handResult?.landmarks
          );
          resnetPred = predictWithResnet(imageData);
          resnetTop3 = getTopPredictions(resnetPred.allConfidences, classes);
        }

        if (enableLandmark && landmarkReady && handResult) {
          landmarkPred = predictWithLandmarkNN(handResult.features);
          landmarkTop3 = getTopPredictions(landmarkPred.allConfidences, classes);
        }

        setResult({
          resnet: resnetPred,
          landmark: landmarkPred,
          handDetection: handResult,
          resnetTop3,
          landmarkTop3,
          handTooSmall,
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
