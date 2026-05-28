'use client';

import { useCallback, useEffect, useRef } from 'react';
import { useAppStore } from '@/store/app-store';
import {
  loadLandmarkModel,
  loadResnetModel,
  loadPreprocessing,
  resetResnetModel,
} from '@/lib/models';
import { initHandLandmarker } from '@/lib/landmarks';

export function useModels() {
  const initialized = useRef(false);
  const store = useAppStore();

  const retryResnet = useCallback(async () => {
    store.setResnetLoadError(null);
    store.setResnetLoaded(false);
    store.setResnetProgress(0);
    store.setLoadingModel('resnet');

    try {
      resetResnetModel();
      await loadResnetModel((p) => store.setResnetProgress(p));
      store.setResnetLoaded(true);
      store.setResnetLoadError(null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to load ResNet model';
      store.setResnetLoadError(message);
      console.error('Failed to load resnet model:', err);
    } finally {
      store.setLoadingModel(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const initModels = useCallback(async () => {
    if (initialized.current) return;
    initialized.current = true;

    try {
      store.setLoadingModel('preprocessing');
      await loadPreprocessing();
    } catch (err) {
      console.error('Failed to load preprocessing:', err);
    }

    try {
      store.setLoadingModel('mediapipe');
      await initHandLandmarker(store.detectionConfidence);
    } catch (err) {
      console.error('Failed to init MediaPipe:', err);
    }

    try {
      store.setLoadingModel('landmark');
      await loadLandmarkModel((p) => store.setLandmarkProgress(p));
      store.setLandmarkLoaded(true);
    } catch (err) {
      console.error('Failed to load landmark model:', err);
    }

    try {
      store.setLoadingModel('resnet');
      await loadResnetModel((p) => store.setResnetProgress(p));
      store.setResnetLoaded(true);
      store.setResnetLoadError(null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to load ResNet model';
      store.setResnetLoadError(message);
      console.error('Failed to load resnet model:', err);
    }

    store.setLoadingModel(null);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    initModels();
  }, [initModels]);

  return {
    landmarkReady: store.landmarkLoaded,
    resnetReady: store.resnetLoaded,
    resnetLoadError: store.resnetLoadError,
    loading: store.loadingModel,
    landmarkProgress: store.landmarkProgress,
    resnetProgress: store.resnetProgress,
    retryResnet,
  };
}
