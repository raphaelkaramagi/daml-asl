import { create } from 'zustand';

interface AppState {
  // Model loading
  landmarkLoaded: boolean;
  resnetLoaded: boolean;
  landmarkProgress: number;
  resnetProgress: number;
  loadingModel: string | null;

  // Settings
  detectionConfidence: number;
  enableResnet: boolean;
  enableLandmark: boolean;
  darkMode: boolean;

  // Actions
  setLandmarkLoaded: (loaded: boolean) => void;
  setResnetLoaded: (loaded: boolean) => void;
  setLandmarkProgress: (p: number) => void;
  setResnetProgress: (p: number) => void;
  setLoadingModel: (model: string | null) => void;
  setDetectionConfidence: (c: number) => void;
  setEnableResnet: (e: boolean) => void;
  setEnableLandmark: (e: boolean) => void;
  toggleDarkMode: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  landmarkLoaded: false,
  resnetLoaded: false,
  landmarkProgress: 0,
  resnetProgress: 0,
  loadingModel: null,

  detectionConfidence: 0.3,
  enableResnet: true,
  enableLandmark: true,
  darkMode: true,

  setLandmarkLoaded: (loaded) => set({ landmarkLoaded: loaded }),
  setResnetLoaded: (loaded) => set({ resnetLoaded: loaded }),
  setLandmarkProgress: (p) => set({ landmarkProgress: p }),
  setResnetProgress: (p) => set({ resnetProgress: p }),
  setLoadingModel: (model) => set({ loadingModel: model }),
  setDetectionConfidence: (c) => set({ detectionConfidence: c }),
  setEnableResnet: (e) => set({ enableResnet: e }),
  setEnableLandmark: (e) => set({ enableLandmark: e }),
  toggleDarkMode: () => set((s) => ({ darkMode: !s.darkMode })),
}));
