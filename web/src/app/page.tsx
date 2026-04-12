'use client';

import dynamic from 'next/dynamic';
import Hero from '@/components/Hero';
import SettingsPanel from '@/components/SettingsPanel';

const LivePredictor = dynamic(() => import('@/components/LivePredictor'), { ssr: false });
const TrainingReplay = dynamic(() => import('@/components/TrainingReplay'), { ssr: false });
const MicroTraining = dynamic(() => import('@/components/MicroTraining'), { ssr: false });
const ModelComparison = dynamic(() => import('@/components/ModelComparison'), { ssr: false });
const SampleGallery = dynamic(() => import('@/components/SampleGallery'), { ssr: false });

export default function Home() {
  return (
    <main className="min-h-screen">
      <nav className="fixed top-0 inset-x-0 z-40 bg-zinc-950/80 backdrop-blur-md border-b border-zinc-800/50">
        <div className="max-w-6xl mx-auto px-4 md:px-8 h-14 flex items-center justify-between">
          <span className="text-sm font-semibold text-white tracking-tight">
            ASL Recognition
          </span>
          <div className="hidden md:flex items-center gap-6 text-xs text-zinc-400">
            <a href="#predictor" className="hover:text-white transition-colors">
              Predict
            </a>
            <a href="#training" className="hover:text-white transition-colors">
              Training
            </a>
            <a href="#micro-training" className="hover:text-white transition-colors">
              Micro Train
            </a>
            <a href="#comparison" className="hover:text-white transition-colors">
              Compare
            </a>
            <a href="#gallery" className="hover:text-white transition-colors">
              Gallery
            </a>
          </div>
        </div>
      </nav>

      <div className="pt-14">
        <Hero />
        <LivePredictor />
        <TrainingReplay />
        <MicroTraining />
        <ModelComparison />
        <SampleGallery />
      </div>

      <footer className="border-t border-zinc-800/50 py-8 px-4 text-center">
        <p className="text-xs text-zinc-600">
          ASL Alphabet Recognition Demo &middot; Built with Next.js, TensorFlow.js &amp; MediaPipe
        </p>
        <p className="text-[10px] text-zinc-700 mt-1">
          All inference runs in your browser. No data leaves your device.
        </p>
      </footer>

      <SettingsPanel />
    </main>
  );
}
