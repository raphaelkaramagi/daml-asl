import { FilesetResolver, HandLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision';

let handLandmarker: HandLandmarker | null = null;
let initPromise: Promise<HandLandmarker> | null = null;

export interface HandDetectionResult {
  landmarks: NormalizedLandmark[];
  worldLandmarks: NormalizedLandmark[];
  features: number[];  // 63-dim wrist-relative vector
}

export async function initHandLandmarker(
  minConfidence = 0.5
): Promise<HandLandmarker> {
  if (handLandmarker) return handLandmarker;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'IMAGE',
      numHands: 1,
      minHandDetectionConfidence: minConfidence,
      minHandPresenceConfidence: minConfidence,
    });
    return handLandmarker;
  })();

  return initPromise;
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
  const features = extractFeaturesFromLandmarks(landmarks);

  return { landmarks, worldLandmarks, features };
}

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
  const features = extractFeaturesFromLandmarks(landmarks);

  return { landmarks, worldLandmarks, features };
}
