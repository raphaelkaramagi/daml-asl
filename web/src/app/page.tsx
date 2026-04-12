'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { motion, AnimatePresence } from 'framer-motion';
import Hero from '@/components/Hero';
import SettingsPanel from '@/components/SettingsPanel';

const LivePredictor = dynamic(() => import('@/components/LivePredictor'), { ssr: false });
const TrainingReplay = dynamic(() => import('@/components/TrainingReplay'), { ssr: false });
const MicroTraining = dynamic(() => import('@/components/MicroTraining'), { ssr: false });
const ModelComparison = dynamic(() => import('@/components/ModelComparison'), { ssr: false });
const SampleGallery = dynamic(() => import('@/components/SampleGallery'), { ssr: false });

const NAV_LINKS = [
  { href: '#predictor', label: 'Predict' },
  { href: '#training', label: 'Training' },
  { href: '#micro-training', label: 'Micro Train' },
  { href: '#comparison', label: 'Compare' },
  { href: '#gallery', label: 'Gallery' },
];

export default function Home() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <main className="min-h-screen">
      <nav className="fixed top-0 inset-x-0 z-40 bg-zinc-950/80 backdrop-blur-md border-b border-zinc-800/50">
        <div className="max-w-6xl mx-auto px-4 md:px-8 h-14 flex items-center justify-between">
          <a href="#" className="text-sm font-semibold text-white tracking-tight">
            ASL Recognition
          </a>
          <div className="hidden md:flex items-center gap-6 text-xs text-zinc-400">
            {NAV_LINKS.map((link) => (
              <a key={link.href} href={link.href} className="hover:text-white transition-colors">
                {link.label}
              </a>
            ))}
          </div>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden text-zinc-400 hover:text-white p-1"
            aria-label="Toggle menu"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              {mobileMenuOpen ? (
                <>
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </>
              ) : (
                <>
                  <line x1="4" y1="7" x2="20" y2="7" />
                  <line x1="4" y1="12" x2="20" y2="12" />
                  <line x1="4" y1="17" x2="20" y2="17" />
                </>
              )}
            </svg>
          </button>
        </div>

        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="md:hidden overflow-hidden bg-zinc-950/95 border-b border-zinc-800/50"
            >
              <div className="px-4 py-3 space-y-1">
                {NAV_LINKS.map((link) => (
                  <a
                    key={link.href}
                    href={link.href}
                    onClick={() => setMobileMenuOpen(false)}
                    className="block py-2 text-sm text-zinc-400 hover:text-white transition-colors"
                  >
                    {link.label}
                  </a>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </nav>

      <div className="pt-14">
        <Hero />
        <LivePredictor />
        <TrainingReplay />
        <MicroTraining />
        <ModelComparison />
        <SampleGallery />
      </div>

      <footer className="border-t border-zinc-800/50 py-10 px-4 text-center">
        <p className="text-xs text-zinc-500">
          ASL Alphabet Recognition Demo
        </p>
        <p className="text-[10px] text-zinc-600 mt-1">
          Built with Next.js, TensorFlow.js &amp; MediaPipe &middot; All inference runs in your browser
        </p>
        <div className="flex justify-center gap-4 mt-3">
          <a
            href="https://www.kaggle.com/datasets/grassknoted/asl-alphabet"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[10px] text-zinc-600 hover:text-zinc-400 transition-colors"
          >
            Dataset
          </a>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[10px] text-zinc-600 hover:text-zinc-400 transition-colors"
          >
            Source
          </a>
        </div>
      </footer>

      <SettingsPanel />
    </main>
  );
}
