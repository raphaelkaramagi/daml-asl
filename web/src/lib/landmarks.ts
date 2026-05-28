import {
  FilesetResolver,
  HandLandmarker,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';

const MEDIAPIPE_VISION_VERSION = '0.10.34';
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VISION_VERSION}/wasm`;

let handLandmarker: HandLandmarker | null = null;
let initPromise: Promise<HandLandmarker> | null = null;
let currentConfidence = 0.5;

export interface HandDetectionResult {
  landmarks: NormalizedLandmark[];
  worldLandmarks: NormalizedLandmark[];
  features: number[];
}

export async function initHandLandmarker(
  minConfidence = 0.5
): Promise<HandLandmarker> {
  currentConfidence = minConfidence;

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

export function updateHandLandmarkerConfidence(minConfidence: number): void {
  currentConfidence = minConfidence;
  if (handLandmarker) {
    handLandmarker.setOptions({
      minHandDetectionConfidence: minConfidence,
      minHandPresenceConfidence: minConfidence,
    });
  }
}

export function setRunningMode(mode: 'IMAGE' | 'VIDEO') {
  if (handLandmarker) {
    handLandmarker.setOptions({ runningMode: mode });
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

/** Simple single-pass detection for gallery/upload (matches pre-session behavior). */
export function detectHandFromImage(
  image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement
): HandDetectionResult | null {
  if (!handLandmarker) return null;

  const result = handLandmarker.detect(image);

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

/** Webcam path: raw unmirrored video pixels (matches training data). */
export function detectHandFromVideo(
  video: HTMLVideoElement,
  timestampMs: number
): HandDetectionResult | null {
  if (!handLandmarker) return null;

  const result = handLandmarker.detectForVideo(video, timestampMs);

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
