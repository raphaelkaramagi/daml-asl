'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import Section from './ui/Section';
import Card from './ui/Card';

interface EpochData {
  epoch: number;
  loss: number;
  accuracy: number;
  val_loss: number;
  val_accuracy: number;
}

interface PhaseData {
  label: string;
  description: string;
  optimizer: string;
  epochs: EpochData[];
  annotations?: { epoch: number; text: string }[];
}

interface TrainingData {
  resnet: {
    phase1: PhaseData;
    phase2: PhaseData;
    summary: Record<string, string | number>;
  };
  landmark: {
    training: PhaseData;
    summary: Record<string, string | number>;
  };
}

type Tab = 'resnet-phase1' | 'resnet-phase2' | 'landmark';

export default function TrainingReplay() {
  const [data, setData] = useState<TrainingData | null>(null);
  const [tab, setTab] = useState<Tab>('landmark');
  const [visibleEpochs, setVisibleEpochs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_BASE_PATH ?? ''}/training-data.json`)
      .then((r) => r.json())
      .then(setData);
  }, []);

  const currentPhase =
    tab === 'resnet-phase1'
      ? data?.resnet.phase1
      : tab === 'resnet-phase2'
      ? data?.resnet.phase2
      : data?.landmark.training;

  const maxEpochs = currentPhase?.epochs.length ?? 0;

  useEffect(() => {
    setVisibleEpochs(0);
    setPlaying(false);
  }, [tab]);

  const tick = useCallback(() => {
    setVisibleEpochs((prev) => {
      if (prev >= maxEpochs) {
        setPlaying(false);
        return prev;
      }
      return prev + 1;
    });
  }, [maxEpochs]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(tick, 600 / speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [playing, speed, tick]);

  const displayData = currentPhase?.epochs.slice(0, visibleEpochs) ?? [];
  const activeAnnotations = currentPhase?.annotations?.filter(
    (a) => a.epoch <= visibleEpochs
  );

  if (!data) return null;

  return (
    <Section
      id="training"
      title="Training Replay"
      subtitle="Watch the real training process unfold epoch by epoch with actual metrics from model training."
      dark
    >
      <div className="flex gap-2 p-1 bg-zinc-800/50 rounded-lg mb-6 max-w-lg">
        {([
          { key: 'landmark', label: 'Landmark NN' },
          { key: 'resnet-phase1', label: 'ResNet Phase 1' },
          { key: 'resnet-phase2', label: 'ResNet Phase 2' },
        ] as const).map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex-1 py-2 px-3 rounded-md text-xs font-medium transition-colors ${
              tab === t.key ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-300'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <Card className="mb-4">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-sm font-semibold text-white">{currentPhase?.label}</h3>
            <p className="text-xs text-zinc-500">{currentPhase?.description}</p>
          </div>
          <span className="text-[10px] text-zinc-500 font-mono">
            {currentPhase?.optimizer}
          </span>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => {
              if (visibleEpochs >= maxEpochs) setVisibleEpochs(0);
              setPlaying(!playing);
            }}
            className="w-10 h-10 shrink-0 rounded-full bg-blue-600 hover:bg-blue-500 text-white flex items-center justify-center transition-colors"
          >
            {playing ? (
              <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                <rect x="2" y="1" width="3.5" height="12" rx="1" />
                <rect x="8.5" y="1" width="3.5" height="12" rx="1" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                <path d="M3 1.5v11l9-5.5z" />
              </svg>
            )}
          </button>

          <div className="flex-1">
            <input
              type="range"
              min={0}
              max={maxEpochs}
              value={visibleEpochs}
              onChange={(e) => {
                setVisibleEpochs(Number(e.target.value));
                setPlaying(false);
              }}
              className="w-full accent-blue-500"
            />
          </div>

          <span className="text-xs text-zinc-400 font-mono w-20 text-right">
            Epoch {visibleEpochs}/{maxEpochs}
          </span>

          <div className="flex gap-1">
            {[1, 2, 4].map((s) => (
              <button
                key={s}
                onClick={() => setSpeed(s)}
                className={`px-2 py-1 rounded text-[10px] font-mono ${
                  speed === s
                    ? 'bg-zinc-700 text-white'
                    : 'text-zinc-500 hover:text-zinc-300'
                }`}
              >
                {s}x
              </button>
            ))}
          </div>
        </div>

        {visibleEpochs > 0 && displayData.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            className="mt-3 grid grid-cols-4 gap-3"
          >
            {[
              { label: 'Train Acc', value: displayData[displayData.length - 1].accuracy, fmt: 'pct' },
              { label: 'Val Acc', value: displayData[displayData.length - 1].val_accuracy, fmt: 'pct' },
              { label: 'Train Loss', value: displayData[displayData.length - 1].loss, fmt: 'num' },
              { label: 'Val Loss', value: displayData[displayData.length - 1].val_loss, fmt: 'num' },
            ].map((m) => (
              <div key={m.label} className="text-center">
                <div className="text-lg font-bold text-white font-mono">
                  {m.fmt === 'pct'
                    ? `${(m.value * 100).toFixed(1)}%`
                    : m.value.toFixed(4)}
                </div>
                <div className="text-[10px] text-zinc-500">{m.label}</div>
              </div>
            ))}
          </motion.div>
        )}
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <p className="text-xs text-zinc-500 mb-3 uppercase tracking-wider">
            Accuracy
          </p>
          <div className="h-[240px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={displayData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fill: '#71717a', fontSize: 11 }}
                  stroke="#3f3f46"
                />
                <YAxis
                  tick={{ fill: '#71717a', fontSize: 11 }}
                  stroke="#3f3f46"
                  domain={[0, 1]}
                  tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
                />
                <Tooltip
                  contentStyle={{
                    background: '#18181b',
                    border: '1px solid #3f3f46',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v) => `${(Number(v) * 100).toFixed(2)}%`}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  name="Train Acc"
                />
                <Line
                  type="monotone"
                  dataKey="val_accuracy"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={false}
                  name="Val Acc"
                />
                {activeAnnotations?.map((a) => (
                  <ReferenceLine
                    key={a.epoch}
                    x={a.epoch}
                    stroke="#f59e0b"
                    strokeDasharray="3 3"
                    label={{
                      value: a.text,
                      fill: '#f59e0b',
                      fontSize: 9,
                      position: 'top',
                    }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <p className="text-xs text-zinc-500 mb-3 uppercase tracking-wider">Loss</p>
          <div className="h-[240px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={displayData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis
                  dataKey="epoch"
                  tick={{ fill: '#71717a', fontSize: 11 }}
                  stroke="#3f3f46"
                />
                <YAxis
                  tick={{ fill: '#71717a', fontSize: 11 }}
                  stroke="#3f3f46"
                />
                <Tooltip
                  contentStyle={{
                    background: '#18181b',
                    border: '1px solid #3f3f46',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(v) => Number(v).toFixed(4)}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  name="Train Loss"
                />
                <Line
                  type="monotone"
                  dataKey="val_loss"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                  name="Val Loss"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </Section>
  );
}
