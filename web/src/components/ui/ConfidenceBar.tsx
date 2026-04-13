'use client';

import { motion } from 'framer-motion';

interface ConfidenceBarProps {
  label: string;
  confidence: number;
  highlight?: boolean;
  color?: string;
}

export default function ConfidenceBar({
  label,
  confidence,
  highlight = false,
  color = 'bg-blue-500',
}: ConfidenceBarProps) {
  const pct = (confidence * 100).toFixed(1);

  return (
    <div className="flex items-center gap-3 text-sm">
      <span
        className={`w-16 text-right font-mono ${
          highlight ? 'text-white font-semibold' : 'text-zinc-400'
        }`}
      >
        {label}
      </span>
      <div className="flex-1 h-5 bg-zinc-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${confidence * 100}%` }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
          className={`h-full rounded-full ${highlight ? color : 'bg-zinc-600'}`}
        />
      </div>
      <span
        className={`w-14 text-right font-mono text-xs ${
          highlight ? 'text-white' : 'text-zinc-500'
        }`}
      >
        {pct}%
      </span>
    </div>
  );
}
