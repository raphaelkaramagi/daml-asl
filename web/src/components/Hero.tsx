'use client';

import { motion } from 'framer-motion';
import { useModels } from '@/hooks/useModels';

const ASL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

export default function Hero() {
  const { loading, landmarkReady, resnetReady, landmarkProgress, resnetProgress } = useModels();

  return (
    <section className="relative min-h-[90vh] flex flex-col items-center justify-center px-4 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-blue-950/20 via-transparent to-transparent" />

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative z-10 text-center max-w-3xl"
      >
        <div className="flex justify-center gap-1 mb-8 flex-wrap">
          {ASL_LETTERS.map((letter, i) => (
            <motion.span
              key={letter}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.02 + 0.3, duration: 0.3 }}
              className="w-8 h-8 rounded-md bg-zinc-800/80 border border-zinc-700/50 flex items-center justify-center text-xs font-mono text-zinc-300 hover:bg-blue-600/30 hover:border-blue-500/50 hover:text-white transition-colors cursor-default"
            >
              {letter}
            </motion.span>
          ))}
        </div>

        <h1 className="text-4xl md:text-6xl font-bold text-white tracking-tight mb-4">
          ASL Alphabet
          <span className="block text-blue-400">Recognition</span>
        </h1>

        <p className="text-zinc-400 text-lg md:text-xl mb-4 max-w-xl mx-auto">
          Deep learning models for classifying American Sign Language alphabet
          gestures. Two approaches compared side-by-side.
        </p>

        <div className="flex gap-4 justify-center text-sm text-zinc-500 mb-10">
          <span className="px-3 py-1 rounded-full bg-zinc-800/50 border border-zinc-700/50">
            29 classes
          </span>
          <span className="px-3 py-1 rounded-full bg-zinc-800/50 border border-zinc-700/50">
            87K+ images
          </span>
          <span className="px-3 py-1 rounded-full bg-zinc-800/50 border border-zinc-700/50">
            2 models
          </span>
        </div>

        {loading ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="space-y-3 max-w-sm mx-auto"
          >
            <p className="text-sm text-zinc-400">
              Loading{' '}
              <span className="text-white font-medium">
                {loading === 'preprocessing'
                  ? 'preprocessing data'
                  : loading === 'mediapipe'
                  ? 'MediaPipe hand detection'
                  : loading === 'landmark'
                  ? 'Landmark NN'
                  : 'ResNet50'}
              </span>
              ...
            </p>
            <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-blue-500 rounded-full"
                animate={{
                  width: `${
                    loading === 'resnet'
                      ? resnetProgress * 100
                      : loading === 'landmark'
                      ? landmarkProgress * 100
                      : 50
                  }%`,
                }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <div className="flex gap-4 justify-center text-xs">
              <span className={landmarkReady ? 'text-emerald-400' : 'text-zinc-600'}>
                {landmarkReady ? 'Landmark NN ready' : 'Landmark NN...'}
              </span>
              <span className={resnetReady ? 'text-emerald-400' : 'text-zinc-600'}>
                {resnetReady ? 'ResNet50 ready' : 'ResNet50...'}
              </span>
            </div>
          </motion.div>
        ) : (
          <motion.a
            href="#predictor"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white font-medium rounded-lg transition-colors"
          >
            Try it live
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M8 3v10M8 13l4-4M8 13L4 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </motion.a>
        )}
      </motion.div>
    </section>
  );
}
