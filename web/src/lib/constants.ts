export const CLASS_NAMES = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
  'del', 'nothing', 'space',
] as const;

export type ClassName = (typeof CLASS_NAMES)[number];

export const NUM_CLASSES = CLASS_NAMES.length; // 29
export const NUM_LANDMARKS = 21;
export const LANDMARK_FEATURES = NUM_LANDMARKS * 3; // 63
export const RESNET_INPUT_SIZE = 96;

const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

export const MODEL_PATHS = {
  landmarkNN: `${BASE}/models/landmark-nn/model.json`,
  resnet: `${BASE}/models/resnet-graph/model.json`,
  preprocessing: `${BASE}/models/preprocessing.json`,
} as const;

export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // index
  [0, 9], [9, 10], [10, 11], [11, 12],  // middle
  [0, 13], [13, 14], [14, 15], [15, 16],// ring
  [0, 17], [17, 18], [18, 19], [19, 20],// pinky
  [5, 9], [9, 13], [13, 17],            // palm
];
