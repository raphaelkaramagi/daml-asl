'use client';

import { motion } from 'framer-motion';
import Section from './ui/Section';
import Card from './ui/Card';
import EvaluationSummary from './EvaluationSummary';

const MODELS = [
  {
    name: 'ResNet50',
    color: 'blue',
    badge: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    architecture: [
      { label: 'Input', detail: '96x96 RGB image' },
      { label: 'Backbone', detail: 'ResNet50 (ImageNet pretrained)' },
      { label: 'Pooling', detail: 'GlobalAveragePooling2D' },
      { label: 'Dropout', detail: '0.2' },
      { label: 'Dense', detail: '128 units, ReLU' },
      { label: 'Dropout', detail: '0.2' },
      { label: 'Output', detail: '29 units, Softmax' },
    ],
    stats: {
      parameters: '~23.6M',
      modelSize: '208 MB Keras (~91 MB TF.js)',
      trainingTime: '~4 hours (hand-cropped, Metal GPU)',
      bestValAccuracy: 'See docs/RESULTS.md',
      testAccuracy: '96.4% end-to-end (27/28)',
      detectionRate: '67.9% (shared MediaPipe)',
    },
    strengths: [
      'Best end-to-end accuracy (96.4%)',
      'Always produces a prediction',
      'Hand-crop at eval when landmarks found',
      'Recommended for live demo',
    ],
    weaknesses: [
      'Large model (~91 MB in browser)',
      'Slower inference than Landmark NN',
    ],
  },
  {
    name: 'Landmark NN',
    color: 'emerald',
    badge: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    architecture: [
      { label: 'Input', detail: '63 features (21x3 coords)' },
      { label: 'Dense', detail: '128 units, ReLU' },
      { label: 'Dropout', detail: '0.3' },
      { label: 'Dense', detail: '64 units, ReLU' },
      { label: 'Dropout', detail: '0.3' },
      { label: 'Output', detail: '29 units, Softmax' },
    ],
    stats: {
      parameters: '~18K',
      modelSize: '~72 KB TF.js',
      trainingTime: '~2 min',
      bestValAccuracy: '98.97%',
      testAccuracy: '100% given detection',
      detectionRate: '67.9% end-to-end on 28 photos',
    },
    strengths: [
      '100% accuracy when hand is detected',
      'Extremely lightweight (~18K params)',
      'Instant inference (<1ms)',
      'Ideal when MediaPipe finds a hand',
    ],
    weaknesses: [
      'Requires MediaPipe hand detection first',
      'End-to-end capped at 67.9% by detection rate',
      'Cannot classify without landmarks',
    ],
  },
];

export default function ModelComparison() {
  return (
    <Section
      id="comparison"
      title="Model Comparison"
      subtitle="ResNet50 for best end-to-end accuracy; Landmark NN for lightweight geometry-based classification."
      dark
    >
      <EvaluationSummary />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {MODELS.map((model, idx) => (
          <motion.div
            key={model.name}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: idx * 0.15 }}
          >
            <Card className="h-full">
              <div className="flex items-center gap-3 mb-5">
                <span className={`text-xs px-2.5 py-1 rounded-full border ${model.badge}`}>
                  {model.name}
                </span>
              </div>

              <div className="mb-5">
                <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-2">
                  Architecture
                </p>
                <div className="space-y-1">
                  {model.architecture.map((layer, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 text-xs"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-zinc-600" />
                      <span className="text-zinc-400 w-20">{layer.label}</span>
                      <span className="text-zinc-300 font-mono text-[11px]">
                        {layer.detail}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mb-5">
                <p className="text-[10px] uppercase tracking-wider text-zinc-600 mb-2">
                  Metrics
                </p>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(model.stats).map(([key, val]) => (
                    <div key={key} className="bg-zinc-800/50 rounded-md p-2">
                      <div className="text-xs font-mono text-white">{val}</div>
                      <div className="text-[10px] text-zinc-500">
                        {key.replace(/([A-Z])/g, ' $1').toLowerCase()}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-emerald-500/70 mb-1.5">
                    Strengths
                  </p>
                  {model.strengths.map((s) => (
                    <p key={s} className="text-[11px] text-zinc-400 flex gap-1.5 mb-1">
                      <span className="text-emerald-500 mt-0.5">+</span> {s}
                    </p>
                  ))}
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-red-500/70 mb-1.5">
                    Weaknesses
                  </p>
                  {model.weaknesses.map((w) => (
                    <p key={w} className="text-[11px] text-zinc-400 flex gap-1.5 mb-1">
                      <span className="text-red-500 mt-0.5">-</span> {w}
                    </p>
                  ))}
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      <Card>
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path
                d="M8 5v3m0 3h.01M14 8A6 6 0 112 8a6 6 0 0112 0z"
                stroke="#f59e0b"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-white mb-1">Key Finding</h4>
            <p className="text-xs text-zinc-400 leading-relaxed">
              <span className="text-blue-400 font-semibold">ResNet50</span> is the primary model
              for live demo — 96.4% end-to-end on the 28-photo test set (27/28), always predicting
              from resized webcam or upload frames.{' '}
              <span className="text-emerald-400 font-semibold">Landmark NN</span> classifies
              wrist-relative geometry with 100% accuracy when MediaPipe detects a hand, but
              end-to-end accuracy is limited to 67.9% by the shared detection rate.
            </p>
          </div>
        </div>
      </Card>
    </Section>
  );
}
