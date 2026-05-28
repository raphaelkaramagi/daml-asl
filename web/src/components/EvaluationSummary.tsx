'use client';

import { useEffect, useState } from 'react';
import Card from './ui/Card';

interface EvaluationData {
  testImages: number;
  detectionRate: number;
  resnetEndToEnd: number;
  resnetCorrect: number;
  landmarkAccuracyGivenDetection: number;
  landmarkEndToEnd: number;
  landmarkCorrect: number;
}

export default function EvaluationSummary() {
  const [data, setData] = useState<EvaluationData | null>(null);

  useEffect(() => {
    fetch('/evaluation-results.json')
      .then((r) => r.json())
      .then(setData)
      .catch(() => {});
  }, []);

  if (!data) return null;

  const metrics = [
    {
      label: 'ResNet end-to-end',
      value: `${data.resnetEndToEnd.toFixed(1)}%`,
      detail: `${data.resnetCorrect}/${data.testImages} test photos`,
      highlight: true,
    },
    {
      label: 'Landmark end-to-end',
      value: `${data.landmarkEndToEnd.toFixed(1)}%`,
      detail: `${data.landmarkCorrect}/${data.testImages} test photos`,
    },
    {
      label: 'Detection rate',
      value: `${data.detectionRate.toFixed(1)}%`,
      detail: 'MediaPipe hand detection',
    },
    {
      label: 'Given detection',
      value: `${data.landmarkAccuracyGivenDetection.toFixed(1)}%`,
      detail: 'Landmark NN classifier',
    },
  ];

  return (
    <Card className="mb-6">
      <p className="text-xs uppercase tracking-wider text-zinc-500 mb-3">
        Deployed model evaluation ({data.testImages}-photo test set)
      </p>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        {metrics.map((m) => (
          <div
            key={m.label}
            className={`rounded-md p-3 text-center ${
              m.highlight ? 'bg-blue-500/10 border border-blue-500/20' : 'bg-zinc-800/50'
            }`}
          >
            <div className="text-lg font-bold text-white font-mono">{m.value}</div>
            <div className="text-[10px] text-zinc-500">{m.label}</div>
            <div className="text-[9px] text-zinc-600 mt-0.5">{m.detail}</div>
          </div>
        ))}
      </div>
      <div className="space-y-2 text-xs text-zinc-400 leading-relaxed">
        <p>
          <span className="text-blue-400 font-medium">ResNet50</span> is the recommended
          primary model for live demo — it uses hand-cropped pixels and always produces a
          prediction ({data.resnetEndToEnd.toFixed(1)}% end-to-end).
        </p>
        <p>
          <span className="text-emerald-400 font-medium">Landmark NN</span> is lightweight
          and classifies hand geometry with {data.landmarkAccuracyGivenDetection.toFixed(0)}%
          accuracy when MediaPipe detects a hand, but end-to-end accuracy is capped by the{' '}
          {data.detectionRate.toFixed(1)}% detection rate.
        </p>
        <p className="text-[10px] text-zinc-600">
          Tip: use the detection confidence slider in Settings if hands are missed. Stop the
          webcam before using the gallery to avoid competing detection loops.
        </p>
      </div>
    </Card>
  );
}
