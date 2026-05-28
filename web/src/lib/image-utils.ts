import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import { RESNET_INPUT_SIZE } from './constants';

export const MIN_DETECTION_DIM = 300;
export const DETECTION_SCALES = [1, 1.5, 2] as const;
export const BBOX_PADDING = 0.2;
/** Normalized bbox area below this suggests hand is too far from camera */
export const SMALL_HAND_AREA_THRESHOLD = 0.08;

export interface NormalizedBBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  area: number;
}

export function getPaddedBBox(
  landmarks: NormalizedLandmark[],
  padding = BBOX_PADDING
): NormalizedBBox {
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (const lm of landmarks) {
    minX = Math.min(minX, lm.x);
    minY = Math.min(minY, lm.y);
    maxX = Math.max(maxX, lm.x);
    maxY = Math.max(maxY, lm.y);
  }
  const w = maxX - minX;
  const h = maxY - minY;
  const padX = w * padding;
  const padY = h * padding;
  minX = Math.max(0, minX - padX);
  minY = Math.max(0, minY - padY);
  maxX = Math.min(1, maxX + padX);
  maxY = Math.min(1, maxY + padY);
  return {
    minX,
    minY,
    maxX,
    maxY,
    area: (maxX - minX) * (maxY - minY),
  };
}

export function getSourceDimensions(
  source: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement
): { width: number; height: number } {
  if (source instanceof HTMLVideoElement) {
    return { width: source.videoWidth, height: source.videoHeight };
  }
  if (source instanceof HTMLImageElement) {
    return { width: source.naturalWidth, height: source.naturalHeight };
  }
  return { width: source.width, height: source.height };
}

/** Draw source to canvas; optionally mirror horizontally (webcam preview parity). */
export function drawSourceToCanvas(
  source: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
  options?: { mirror?: boolean; width?: number; height?: number }
): HTMLCanvasElement {
  const { width: srcW, height: srcH } = getSourceDimensions(source);
  const width = options?.width ?? srcW;
  const height = options?.height ?? srcH;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  if (options?.mirror) {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(source, 0, 0, width, height);
  return canvas;
}

/** Upscale canvas so smallest dimension is at least minDim. */
export function upscaleCanvas(
  canvas: HTMLCanvasElement,
  minDim = MIN_DETECTION_DIM
): HTMLCanvasElement {
  if (canvas.width >= minDim && canvas.height >= minDim) {
    return canvas;
  }
  const scale = Math.max(minDim / canvas.width, minDim / canvas.height, 1);
  const up = document.createElement('canvas');
  up.width = Math.round(canvas.width * scale);
  up.height = Math.round(canvas.height * scale);
  const ctx = up.getContext('2d')!;
  ctx.drawImage(canvas, 0, 0, up.width, up.height);
  return up;
}

/** Scale canvas by factor (for multi-scale detection retry). */
export function scaleCanvas(
  canvas: HTMLCanvasElement,
  factor: number
): HTMLCanvasElement {
  if (factor === 1) return canvas;
  const scaled = document.createElement('canvas');
  scaled.width = Math.round(canvas.width * factor);
  scaled.height = Math.round(canvas.height * factor);
  const ctx = scaled.getContext('2d')!;
  ctx.drawImage(canvas, 0, 0, scaled.width, scaled.height);
  return scaled;
}

/** Crop hand region from canvas using normalized landmark bbox. */
export function cropCanvasToHand(
  canvas: HTMLCanvasElement,
  landmarks: NormalizedLandmark[],
  padding = BBOX_PADDING
): HTMLCanvasElement {
  const { minX, minY, maxX, maxY } = getPaddedBBox(landmarks, padding);
  const x1 = Math.floor(minX * canvas.width);
  const y1 = Math.floor(minY * canvas.height);
  const x2 = Math.max(x1 + 1, Math.ceil(maxX * canvas.width));
  const y2 = Math.max(y1 + 1, Math.ceil(maxY * canvas.height));

  const cropped = document.createElement('canvas');
  cropped.width = x2 - x1;
  cropped.height = y2 - y1;
  const ctx = cropped.getContext('2d')!;
  ctx.drawImage(canvas, x1, y1, x2 - x1, y2 - y1, 0, 0, x2 - x1, y2 - y1);
  return cropped;
}

/** Prepare ResNet input: full frame or hand crop, resized to model input size. */
export function canvasToResnetImageData(
  canvas: HTMLCanvasElement,
  landmarks?: NormalizedLandmark[] | null
): ImageData {
  const source =
    landmarks && landmarks.length > 0
      ? cropCanvasToHand(canvas, landmarks)
      : canvas;

  const resized = document.createElement('canvas');
  resized.width = RESNET_INPUT_SIZE;
  resized.height = RESNET_INPUT_SIZE;
  const ctx = resized.getContext('2d')!;
  ctx.drawImage(source, 0, 0, RESNET_INPUT_SIZE, RESNET_INPUT_SIZE);
  return ctx.getImageData(0, 0, RESNET_INPUT_SIZE, RESNET_INPUT_SIZE);
}
