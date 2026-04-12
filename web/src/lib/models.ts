import * as tf from '@tensorflow/tfjs';
import { MODEL_PATHS, CLASS_NAMES, RESNET_INPUT_SIZE, LANDMARK_FEATURES } from './constants';

export interface Prediction {
  label: string;
  confidence: number;
  allConfidences: number[];
}

interface PreprocessingData {
  scaler: { mean: number[]; scale: number[] };
  classes: string[];
}

let landmarkModel: tf.LayersModel | null = null;
let resnetModel: tf.LayersModel | null = null;
let preprocessingData: PreprocessingData | null = null;

export async function loadPreprocessing(): Promise<PreprocessingData> {
  if (preprocessingData) return preprocessingData;
  const resp = await fetch(MODEL_PATHS.preprocessing);
  preprocessingData = await resp.json();
  return preprocessingData!;
}

export async function loadLandmarkModel(
  onProgress?: (fraction: number) => void
): Promise<tf.LayersModel> {
  if (landmarkModel) return landmarkModel;
  landmarkModel = await tf.loadLayersModel(MODEL_PATHS.landmarkNN, {
    onProgress: onProgress ?? (() => {}),
  });
  return landmarkModel;
}

export async function loadResnetModel(
  onProgress?: (fraction: number) => void
): Promise<tf.LayersModel> {
  if (resnetModel) return resnetModel;
  resnetModel = await tf.loadLayersModel(MODEL_PATHS.resnet, {
    onProgress: onProgress ?? (() => {}),
  });
  return resnetModel;
}

export function scaleLandmarks(landmarks: number[]): number[] {
  if (!preprocessingData) throw new Error('Preprocessing data not loaded');
  const { mean, scale } = preprocessingData.scaler;
  return landmarks.map((v, i) => (v - mean[i]) / scale[i]);
}

export function predictWithLandmarkNN(landmarks: number[]): Prediction {
  if (!landmarkModel || !preprocessingData) {
    throw new Error('Landmark model or preprocessing not loaded');
  }

  const scaled = scaleLandmarks(landmarks);
  const input = tf.tensor2d([scaled], [1, LANDMARK_FEATURES]);
  const output = landmarkModel.predict(input) as tf.Tensor;
  const probs = output.dataSync() as Float32Array;
  input.dispose();
  output.dispose();

  const allConfidences = Array.from(probs);
  const maxIdx = allConfidences.indexOf(Math.max(...allConfidences));

  return {
    label: preprocessingData.classes[maxIdx],
    confidence: allConfidences[maxIdx],
    allConfidences,
  };
}

export function predictWithResnet(imageData: ImageData): Prediction {
  if (!resnetModel) throw new Error('ResNet model not loaded');

  const tensor = tf.tidy(() => {
    const img = tf.browser.fromPixels(imageData);
    const resized = tf.image.resizeBilinear(img, [RESNET_INPUT_SIZE, RESNET_INPUT_SIZE]);
    const normalized = resized.div(255.0);
    return normalized.expandDims(0);
  });

  const output = resnetModel.predict(tensor) as tf.Tensor;
  const probs = output.dataSync() as Float32Array;
  tensor.dispose();
  output.dispose();

  const allConfidences = Array.from(probs);
  const maxIdx = allConfidences.indexOf(Math.max(...allConfidences));

  return {
    label: CLASS_NAMES[maxIdx],
    confidence: allConfidences[maxIdx],
    allConfidences,
  };
}

export function getTopPredictions(
  confidences: number[],
  classes: string[],
  k = 3
): { label: string; confidence: number }[] {
  const indexed = confidences.map((c, i) => ({ label: classes[i], confidence: c }));
  indexed.sort((a, b) => b.confidence - a.confidence);
  return indexed.slice(0, k);
}

export function isLandmarkModelLoaded() {
  return landmarkModel !== null;
}

export function isResnetModelLoaded() {
  return resnetModel !== null;
}
