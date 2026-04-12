import * as tf from '@tensorflow/tfjs';

export interface TrainingConfig {
  learningRate: number;
  epochs: number;
  batchSize: number;
  hiddenUnits: number;
}

export interface EpochResult {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
}

const DEFAULT_CONFIG: TrainingConfig = {
  learningRate: 0.01,
  epochs: 30,
  batchSize: 16,
  hiddenUnits: 64,
};

export function generateSyntheticData(
  numSamplesPerClass: number,
  numClasses: number,
  numFeatures: number
): { xs: tf.Tensor2D; ys: tf.Tensor1D } {
  const data: number[][] = [];
  const labels: number[] = [];

  for (let c = 0; c < numClasses; c++) {
    for (let i = 0; i < numSamplesPerClass; i++) {
      const sample: number[] = [];
      for (let f = 0; f < numFeatures; f++) {
        const center = ((c * numFeatures + f) % 100) / 100;
        sample.push(center + (Math.random() - 0.5) * 0.3);
      }
      data.push(sample);
      labels.push(c);
    }
  }

  const indices = Array.from({ length: data.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const shuffledData = indices.map((i) => data[i]);
  const shuffledLabels = indices.map((i) => labels[i]);

  return {
    xs: tf.tensor2d(shuffledData),
    ys: tf.tensor1d(shuffledLabels, 'int32'),
  };
}

export function createMicroModel(
  numFeatures: number,
  numClasses: number,
  hiddenUnits: number,
  learningRate: number
): tf.Sequential {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [numFeatures],
      units: hiddenUnits,
      activation: 'relu',
    })
  );
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(
    tf.layers.dense({
      units: Math.floor(hiddenUnits / 2),
      activation: 'relu',
    })
  );
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(
    tf.layers.dense({
      units: numClasses,
      activation: 'softmax',
    })
  );

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

export async function trainMicroModel(
  config: Partial<TrainingConfig> = {},
  onEpochEnd?: (result: EpochResult) => void,
  signal?: AbortSignal
): Promise<{ model: tf.Sequential; history: EpochResult[] }> {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const numClasses = 10; // A-J for demo
  const numFeatures = 63;
  const samplesPerClass = 50;

  const { xs, ys } = generateSyntheticData(samplesPerClass, numClasses, numFeatures);

  const splitIdx = Math.floor(xs.shape[0] * 0.8);
  const xTrain = xs.slice(0, splitIdx);
  const yTrain = ys.slice(0, splitIdx);
  const xVal = xs.slice(splitIdx);
  const yVal = ys.slice(splitIdx);

  const model = createMicroModel(numFeatures, numClasses, cfg.hiddenUnits, cfg.learningRate);
  const history: EpochResult[] = [];

  await model.fit(xTrain, yTrain, {
    epochs: cfg.epochs,
    batchSize: cfg.batchSize,
    validationData: [xVal, yVal],
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (signal?.aborted) {
          model.stopTraining = true;
          return;
        }
        const result: EpochResult = {
          epoch: epoch + 1,
          loss: logs?.loss ?? 0,
          accuracy: logs?.acc ?? 0,
          valLoss: logs?.val_loss ?? 0,
          valAccuracy: logs?.val_acc ?? 0,
        };
        history.push(result);
        onEpochEnd?.(result);
      },
    },
  });

  xs.dispose();
  ys.dispose();
  xTrain.dispose();
  yTrain.dispose();
  xVal.dispose();
  yVal.dispose();

  return { model, history };
}

export function predictWithMicroModel(
  model: tf.Sequential,
  features: number[]
): { classIdx: number; confidence: number; allConfidences: number[] } {
  const input = tf.tensor2d([features], [1, features.length]);
  const output = model.predict(input) as tf.Tensor;
  const probs = Array.from(output.dataSync());
  input.dispose();
  output.dispose();

  const maxIdx = probs.indexOf(Math.max(...probs));
  return { classIdx: maxIdx, confidence: probs[maxIdx], allConfidences: probs };
}
