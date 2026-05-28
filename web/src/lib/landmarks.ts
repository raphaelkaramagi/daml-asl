import {
  FilesetResolver,
  HandLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import {
  DETECTION_SCALES,
  drawSourceToCanvas,
  scaleCanvas,
  upscaleCanvas,
} from './image-utils';

const MEDIAPIPE_VISION_VERSION = '0.10.34';
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VISION_VERSION}/wasm`;

let handLandmarker: HandLandmarker | null = null;
let initPromise: Promise<HandLandmarker> | null = null;
let currentConfidence = 0.2;
let currentTrackingConfidence = 0.5;

/** Temporal hold: keep last detection briefly when video frame misses */
let lastHandResult: HandDetectionResult | null = null;
let missedFrames = 0;
const MAX_MISSED_FRAMES = 3;

export interface HandDetectionResult {
  landmarks: NormalizedLandmark[];
  worldLandmarks: NormalizedLandmark[];
  features: number[];
  /** True if result is held from a previous frame */
  held?: boolean;
}

export async function initHandLandmarker(
  minConfidence = 0.2,
  minTrackingConfidence = 0.5
): Promise<HandLandmarker> {
  currentConfidence = minConfidence;
  currentTrackingConfidence = minTrackingConfidence;

  if (handLandmarker) return handLandmarker;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE);

    const createLandmarker = async (delegate: 'GPU' | 'CPU') =>
      HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate,
        },
        runningMode: 'IMAGE',
        numHands: 1,
        minHandDetectionConfidence: minConfidence,
        minHandPresenceConfidence: minConfidence,
        minTrackingConfidence: minTrackingConfidence,
      });

    try {
      handLandmarker = await createLandmarker('GPU');
    } catch {
      handLandmarker = await createLandmarker('CPU');
    }
    return handLandmarker;
  })();

  return initPromise;
}

export function updateHandLandmarkerConfidence(
  minConfidence: number,
  minTrackingConfidence = currentTrackingConfidence
): void {
  currentConfidence = minConfidence;
  currentTrackingConfidence = minTrackingConfidence;
  if (handLandmarker) {
    handLandmarker.setOptions({
      minHandDetectionConfidence: minConfidence,
      minHandPresenceConfidence: minConfidence,
      minTrackingConfidence: minTrackingConfidence,
    });
  }
}

export function extractFeaturesFromLandmarks(
  landmarks: NormalizedLandmark[]
): number[] {
  const wrist = landmarks[0];
  const features: number[] = [];
  for (const lm of landmarks) {
    features.push(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z);
  }
  return features;
}

function parseDetectionResult(
  result: ReturnType<HandLandmarker['detect']>
): HandDetectionResult | null {
  if (!result.landmarks || result.landmarks.length === 0) {
    return null;
  }
  const landmarks = result.landmarks[0];
  const worldLandmarks = result.worldLandmarks?.[0] ?? landmarks;
  return {
    landmarks,
    worldLandmarks,
    features: extractFeaturesFromLandmarks(landmarks),
  };
}

function detectOnCanvas(canvas: HTMLCanvasElement): HandDetectionResult | null {
  if (!handLandmarker) return null;
  const result = handLandmarker.detect(canvas);
  return parseDetectionResult(result);
}

/** Multi-scale IMAGE detection with optional mirror (matches Python training pipeline). */
export function detectHandFromImage(
  image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
  options?: { mirror?: boolean }
): HandDetectionResult | null {
  if (!handLandmarker) return null;

  const baseCanvas = drawSourceToCanvas(image, { mirror: options?.mirror });
  const upCanvas = upscaleCanvas(baseCanvas);

  for (const scale of DETECTION_SCALES) {
    const scaled = scaleCanvas(upCanvas, scale);
    const detected = detectOnCanvas(scaled);
    if (detected) {
      return detected;
    }
  }
  return null;
}

/** Webcam path: multi-scale IMAGE detection with temporal hold on brief misses. */
export function detectHandWithHold(
  image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
  options?: { mirror?: boolean }
): HandDetectionResult | null {
  const detected = detectHandFromImage(image, options);
  if (detected) {
    lastHandResult = detected;
    missedFrames = 0;
    return detected;
  }

  if (lastHandResult && missedFrames < MAX_MISSED_FRAMES) {
    missedFrames += 1;
    return { ...lastHandResult, held: true };
  }

  missedFrames += 1;
  return null;
}

export function resetTemporalHold(): void {
  lastHandResult = null;
  missedFrames = 0;
}

export function getLastHandResult(): HandDetectionResult | null {
  return lastHandResult;
}
