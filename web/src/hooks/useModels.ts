'use client';

import { useCallback, useEffect, useRef } from 'react';
import { useAppStore } from '@/store/app-store';
import {
  loadLandmarkModel,
  loadResnetModel,
  loadPreprocessing,
} from '@/lib/models';
import { initHandLandmarker } from '@/lib/landmarks';

export function useModels() {
  const initialized = useRef(false);
  const store = useAppStore();

  const initModels = useCallback(async () => {
    if (initialized.current) return;
    initialized.current = true;

    try {
      store.setLoadingModel('preprocessing');
      await loadPreprocessing();

      store.setLoadingModel('mediapipe');
      await initHandLandmarker(store.detectionConfidence);

      store.setLoadingModel('landmark');
      await loadLandmarkModel((p) => store.setLandmarkProgress(p));
      store.setLandmarkLoaded(true);

      store.setLoadingModel('resnet');
      await loadResnetModel((p) => store.setResnetProgress(p));
      store.setResnetLoaded(true);

      store.setLoadingModel(null);
    } catch (err) {
      console.error('Model loading error:', err);
      store.setLoadingModel(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    initModels();
  }, [initModels]);

  return {
    landmarkReady: store.landmarkLoaded,
    resnetReady: store.resnetLoaded,
    loading: store.loadingModel,
    landmarkProgress: store.landmarkProgress,
    resnetProgress: store.resnetProgress,
  };
}
