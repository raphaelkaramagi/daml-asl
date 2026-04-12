'use client';

import { motion } from 'framer-motion';
import Section from './ui/Section';
import Card from './ui/Card';

const MODELS = [
  {
    name: 'ResNet50',
    color: 'blue',
    badge: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    architecture: [
      { label: 'Input', detail: '96x96 RGB image' },
      { label: 'Backbone', detail: 'ResNet50 (ImageNet)' },
      { label: 'Pooling', detail: 'GlobalAveragePooling2D' },
      { label: 'Dropout', detail: '0.2' },
      { label: 'Dense', detail: '128 units, ReLU' },
      { label: 'Dropout', detail: '0.2' },
      { label: 'Output', detail: '29 units, Softmax' },
    ],
    stats: {
      parameters: '~23.6M',
      modelSize: '208 MB (22 MB quantized)',
      trainingTime: '~4 hours',
      valAccuracy: '47.24%',
      testAccuracy: '71.43%',
    },
    strengths: [
      'Works directly on raw images',
      'No preprocessing pipeline required',
      'Leverages ImageNet knowledge',
    ],
    weaknesses: [
      'Large model size',
      'Slower inference',
      'Lower validation accuracy',
    ],
  },
  {
    name: 'Landmark NN',
    color: 'emerald',
    badge: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
    architecture: [
      { label: 'Input', detail: '63 features (21x3)' },
      { label: 'Dense', detail: '128 units, ReLU' },
      { label: 'Dropout', detail: '0.3' },
      { label: 'Dense', detail: '64 units, ReLU' },
      { label: 'Dropout', detail: '0.3' },
      { label: 'Output', detail: '29 units, Softmax' },
    ],
    stats: {
      parameters: '~18K',
      modelSize: '~244 KB',
      trainingTime: '~10 min',
      valAccuracy: '98.88%',
      testAccuracy: '71.43%',
    },
    strengths: [
      '100% accuracy when hand detected',
      'Extremely lightweight',
      'Fast inference',
      'Invariant to lighting/background',
    ],
    weaknesses: [
      'Depends on MediaPipe detection',
      'Fails if no hand detected',
      'Limited to static poses',
    ],
  },
];

export default function ModelComparison() {
  return (
    <Section
      id="comparison"
      title="Model Comparison"
      subtitle="Two fundamentally different approaches to the same problem. Pixels vs. geometry."
      dark
    >
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
              The Landmark NN achieves <span className="text-emerald-400 font-semibold">100% accuracy when MediaPipe successfully detects a hand</span>.
              The real bottleneck is MediaPipe&apos;s hand detection, not the classification model.
              With detection confidence tuned to 0.1, both models achieve equal{' '}
              <span className="text-white font-semibold">71.43% test accuracy</span> on the
              28-image test set (missing the &quot;del&quot; class).
            </p>
          </div>
        </div>
      </Card>
    </Section>
  );
}
