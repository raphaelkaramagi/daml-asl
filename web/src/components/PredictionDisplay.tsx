'use client';

import { motion } from 'framer-motion';
import Card from './ui/Card';
import ConfidenceBar from './ui/ConfidenceBar';
import type { PredictionResult } from '@/hooks/usePrediction';

interface PredictionDisplayProps {
  result: PredictionResult | null;
  loading?: boolean;
}

function ModelResultCard({
  title,
  badge,
  prediction,
  top3,
  color,
  noHand,
}: {
  title: string;
  badge: string;
  prediction?: { label: string; confidence: number };
  top3?: { label: string; confidence: number }[];
  color: string;
  noHand?: boolean;
}) {
  return (
    <Card className="flex-1">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-semibold text-zinc-300">{title}</h4>
        <span className={`text-[10px] px-2 py-0.5 rounded-full ${badge}`}>{title.split(' ')[0]}</span>
      </div>

      {prediction ? (
        <>
          <motion.div
            key={prediction.label}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="text-center mb-5"
          >
            <div
              className={`text-5xl font-bold mb-1 ${
                prediction.label.length > 1 ? 'text-3xl' : ''
              }`}
              style={{ color }}
            >
              {prediction.label}
            </div>
            <div className="text-xs text-zinc-500">
              {(prediction.confidence * 100).toFixed(1)}% confidence
            </div>
          </motion.div>

          {top3 && (
            <div className="space-y-1.5">
              <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-2">
                Top 3 predictions
              </p>
              {top3.map((p, i) => (
                <ConfidenceBar
                  key={p.label}
                  label={p.label}
                  confidence={p.confidence}
                  highlight={i === 0}
                  color={color === '#3b82f6' ? 'bg-blue-500' : 'bg-emerald-500'}
                />
              ))}
            </div>
          )}
        </>
      ) : noHand ? (
        <div className="text-center py-6">
          <div className="text-2xl text-zinc-600 mb-2">?</div>
          <p className="text-xs text-zinc-500">No hand detected</p>
        </div>
      ) : (
        <div className="text-center py-6">
          <div className="text-2xl text-zinc-700 mb-2">--</div>
          <p className="text-xs text-zinc-600">Waiting for input</p>
        </div>
      )}
    </Card>
  );
}

export default function PredictionDisplay({ result, loading }: PredictionDisplayProps) {
  const match =
    result?.resnet && result?.landmark && result.resnet.label === result.landmark.label;

  return (
    <div className="space-y-3">
      {result && result.resnet && result.landmark && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className={`text-center text-xs py-1.5 rounded-lg ${
            match
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
              : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
          }`}
        >
          {match ? 'Both models agree' : 'Models disagree'}
        </motion.div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ModelResultCard
          title="ResNet50"
          badge="bg-blue-500/20 text-blue-400"
          prediction={result?.resnet}
          top3={result?.resnetTop3}
          color="#3b82f6"
        />
        <ModelResultCard
          title="Landmark NN"
          badge="bg-emerald-500/20 text-emerald-400"
          prediction={result?.landmark}
          top3={result?.landmarkTop3}
          color="#10b981"
          noHand={result !== null && !result.landmark && result.handDetection === null}
        />
      </div>

      {loading && (
        <p className="text-center text-xs text-zinc-500 animate-pulse">Processing...</p>
      )}
    </div>
  );
}
