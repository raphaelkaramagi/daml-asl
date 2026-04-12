'use client';

import { useState, useRef, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import Section from './ui/Section';
import Card from './ui/Card';
import { trainMicroModel, type EpochResult } from '@/lib/micro-trainer';

export default function MicroTraining() {
  const [learningRate, setLearningRate] = useState(0.01);
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(16);
  const [hiddenUnits, setHiddenUnits] = useState(64);
  const [history, setHistory] = useState<EpochResult[]>([]);
  const [training, setTraining] = useState(false);
  const [finalAccuracy, setFinalAccuracy] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const handleTrain = useCallback(async () => {
    setHistory([]);
    setFinalAccuracy(null);
    setTraining(true);
    abortRef.current = new AbortController();

    try {
      const { history: h } = await trainMicroModel(
        { learningRate, epochs, batchSize, hiddenUnits },
        (result) => {
          setHistory((prev) => [...prev, result]);
        },
        abortRef.current.signal
      );
      if (h.length > 0) {
        setFinalAccuracy(h[h.length - 1].valAccuracy);
      }
    } catch (err) {
      console.error('Training error:', err);
    } finally {
      setTraining(false);
    }
  }, [learningRate, epochs, batchSize, hiddenUnits]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setTraining(false);
  }, []);

  const handleReset = useCallback(() => {
    abortRef.current?.abort();
    setHistory([]);
    setFinalAccuracy(null);
    setTraining(false);
  }, []);

  const chartData = history.map((h) => ({
    epoch: h.epoch,
    accuracy: h.accuracy,
    valAccuracy: h.valAccuracy,
    loss: h.loss,
    valLoss: h.valLoss,
  }));

  return (
    <Section
      id="micro-training"
      title="Micro Training"
      subtitle="Train a small neural network live in your browser using TensorFlow.js. Adjust hyperparameters and watch the model learn in real-time."
    >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card>
          <h3 className="text-sm font-semibold text-white mb-5">Hyperparameters</h3>

          <div className="space-y-5">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-zinc-400">Learning Rate</span>
                <span className="text-white font-mono">{learningRate}</span>
              </div>
              <input
                type="range"
                min={-4}
                max={-1}
                step={0.5}
                value={Math.log10(learningRate)}
                onChange={(e) => setLearningRate(Math.pow(10, Number(e.target.value)))}
                disabled={training}
                className="w-full accent-blue-500"
              />
              <div className="flex justify-between text-[10px] text-zinc-600">
                <span>0.0001</span>
                <span>0.1</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-zinc-400">Epochs</span>
                <span className="text-white font-mono">{epochs}</span>
              </div>
              <input
                type="range"
                min={5}
                max={100}
                step={5}
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                disabled={training}
                className="w-full accent-blue-500"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-zinc-400">Batch Size</span>
                <span className="text-white font-mono">{batchSize}</span>
              </div>
              <select
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                disabled={training}
                className="w-full bg-zinc-800 text-white text-sm rounded-md p-2 border border-zinc-700"
              >
                {[8, 16, 32, 64].map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-zinc-400">Hidden Units</span>
                <span className="text-white font-mono">{hiddenUnits}</span>
              </div>
              <select
                value={hiddenUnits}
                onChange={(e) => setHiddenUnits(Number(e.target.value))}
                disabled={training}
                className="w-full bg-zinc-800 text-white text-sm rounded-md p-2 border border-zinc-700"
              >
                {[16, 32, 64, 128].map((v) => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="mt-6 space-y-2">
            {!training ? (
              <button
                onClick={handleTrain}
                className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-sm font-medium transition-colors"
              >
                Start Training
              </button>
            ) : (
              <button
                onClick={handleStop}
                className="w-full py-2.5 bg-red-600/20 text-red-400 border border-red-600/30 rounded-lg text-sm font-medium"
              >
                Stop
              </button>
            )}
            <button
              onClick={handleReset}
              disabled={training}
              className="w-full py-2 text-zinc-500 hover:text-zinc-300 text-sm transition-colors disabled:opacity-30"
            >
              Reset
            </button>
          </div>

          {training && (
            <div className="mt-4 text-center">
              <div className="text-xs text-zinc-500">
                Epoch {history.length}/{epochs}
              </div>
              <div className="h-1 bg-zinc-800 rounded-full mt-1.5 overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all duration-300"
                  style={{ width: `${(history.length / epochs) * 100}%` }}
                />
              </div>
            </div>
          )}

          {finalAccuracy !== null && !training && (
            <div className="mt-4 p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-center">
              <div className="text-lg font-bold text-emerald-400">
                {(finalAccuracy * 100).toFixed(1)}%
              </div>
              <div className="text-[10px] text-emerald-400/70">Final Val Accuracy</div>
            </div>
          )}
        </Card>

        <div className="lg:col-span-2 space-y-4">
          <Card>
            <p className="text-xs text-zinc-500 mb-3 uppercase tracking-wider">
              Accuracy
            </p>
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="epoch" tick={{ fill: '#71717a', fontSize: 11 }} stroke="#3f3f46" />
                  <YAxis tick={{ fill: '#71717a', fontSize: 11 }} stroke="#3f3f46" domain={[0, 1]}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  />
                  <Tooltip
                    contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', borderRadius: 8, fontSize: 12 }}
                    formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="accuracy" stroke="#3b82f6" strokeWidth={2} dot={false} name="Train" isAnimationActive={false} />
                  <Line type="monotone" dataKey="valAccuracy" stroke="#10b981" strokeWidth={2} dot={false} name="Validation" isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card>
            <p className="text-xs text-zinc-500 mb-3 uppercase tracking-wider">Loss</p>
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="epoch" tick={{ fill: '#71717a', fontSize: 11 }} stroke="#3f3f46" />
                  <YAxis tick={{ fill: '#71717a', fontSize: 11 }} stroke="#3f3f46" />
                  <Tooltip
                    contentStyle={{ background: '#18181b', border: '1px solid #3f3f46', borderRadius: 8, fontSize: 12 }}
                    formatter={(v: number) => v.toFixed(4)}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} name="Train" isAnimationActive={false} />
                  <Line type="monotone" dataKey="valLoss" stroke="#f59e0b" strokeWidth={2} dot={false} name="Validation" isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card>
            <p className="text-xs text-zinc-500 mb-3">
              This trains a small Dense network in your browser on synthetic landmark-style data
              (10 classes, 63 features). Architecture mirrors the real Landmark NN but on a
              smaller scale for instant feedback.
            </p>
          </Card>
        </div>
      </div>
    </Section>
  );
}
