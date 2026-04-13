'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppStore } from '@/store/app-store';

export default function SettingsPanel() {
  const [open, setOpen] = useState(false);
  const {
    detectionConfidence,
    enableResnet,
    enableLandmark,
    landmarkLoaded,
    resnetLoaded,
    setDetectionConfidence,
    setEnableResnet,
    setEnableLandmark,
  } = useAppStore();

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-6 right-6 z-50 w-11 h-11 rounded-full bg-zinc-800 border border-zinc-700 text-zinc-400 hover:text-white hover:border-zinc-500 flex items-center justify-center transition-colors shadow-xl"
        aria-label="Settings"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
        </svg>
      </button>

      <AnimatePresence>
        {open && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setOpen(false)}
              className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm"
            />
            <motion.div
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 30, stiffness: 300 }}
              className="fixed right-0 top-0 bottom-0 z-50 w-80 bg-zinc-900 border-l border-zinc-800 p-6 overflow-y-auto"
            >
              <div className="flex items-center justify-between mb-8">
                <h3 className="text-lg font-semibold text-white">Settings</h3>
                <button
                  onClick={() => setOpen(false)}
                  className="text-zinc-500 hover:text-white transition-colors"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>

              <div className="space-y-6">
                <div>
                  <h4 className="text-xs uppercase tracking-wider text-zinc-500 mb-3">
                    Models
                  </h4>
                  <div className="space-y-3">
                    <label className="flex items-center justify-between cursor-pointer">
                      <div>
                        <span className="text-sm text-white">ResNet50</span>
                        <p className={`text-[10px] ${resnetLoaded ? 'text-emerald-400' : 'text-zinc-500'}`}>
                          {resnetLoaded ? 'Loaded (~23 MB)' : 'Loading...'}
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        checked={enableResnet}
                        onChange={(e) => setEnableResnet(e.target.checked)}
                        className="accent-blue-500 w-4 h-4"
                      />
                    </label>
                    <label className="flex items-center justify-between cursor-pointer">
                      <div>
                        <span className="text-sm text-white">Landmark NN</span>
                        <p className={`text-[10px] ${landmarkLoaded ? 'text-emerald-400' : 'text-zinc-500'}`}>
                          {landmarkLoaded ? 'Loaded (~73 KB)' : 'Loading...'}
                        </p>
                      </div>
                      <input
                        type="checkbox"
                        checked={enableLandmark}
                        onChange={(e) => setEnableLandmark(e.target.checked)}
                        className="accent-emerald-500 w-4 h-4"
                      />
                    </label>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs uppercase tracking-wider text-zinc-500 mb-3">
                    Hand Detection
                  </h4>
                  <div>
                    <div className="flex justify-between text-xs mb-1.5">
                      <span className="text-zinc-400">Detection Confidence</span>
                      <span className="text-white font-mono">
                        {detectionConfidence.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0.05}
                      max={1}
                      step={0.05}
                      value={detectionConfidence}
                      onChange={(e) => setDetectionConfidence(Number(e.target.value))}
                      className="w-full accent-blue-500"
                    />
                    <div className="flex justify-between text-[10px] text-zinc-600 mt-0.5">
                      <span>More sensitive</span>
                      <span>More strict</span>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs uppercase tracking-wider text-zinc-500 mb-3">
                    About
                  </h4>
                  <div className="text-xs text-zinc-500 space-y-1">
                    <p>ASL Alphabet Recognition Demo</p>
                    <p>Models run entirely in your browser</p>
                    <p>No data is sent to any server</p>
                    <p className="pt-2">
                      <a
                        href="https://www.kaggle.com/datasets/grassknoted/asl-alphabet"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-400 hover:text-blue-300"
                      >
                        Dataset: Kaggle ASL Alphabet
                      </a>
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
