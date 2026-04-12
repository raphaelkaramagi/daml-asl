'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Section from './ui/Section';
import Card from './ui/Card';
import { CLASS_NAMES } from '@/lib/constants';
import { usePrediction } from '@/hooks/usePrediction';
import PredictionDisplay from './PredictionDisplay';

export default function SampleGallery() {
  const [manifest, setManifest] = useState<string[]>([]);
  const [selectedClass, setSelectedClass] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const { result, loading, predict, clear } = usePrediction();
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetch('/samples/manifest.json')
      .then((r) => r.json())
      .then(setManifest)
      .catch(() => {});
  }, []);

  const filteredClasses = search
    ? CLASS_NAMES.filter((c) => c.toLowerCase().includes(search.toLowerCase()))
    : [...CLASS_NAMES];

  const classImages = selectedClass
    ? manifest.filter((f) => {
        const name = f.replace('.jpg', '');
        return (
          name === `${selectedClass}_test` ||
          name.startsWith(`${selectedClass}_train_`)
        );
      })
    : [];

  const handleImageClick = useCallback(
    (filename: string) => {
      setSelectedImage(filename);
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => predict(img);
      img.src = `/samples/${filename}`;
    },
    [predict]
  );

  return (
    <Section
      id="gallery"
      title="Sample Gallery"
      subtitle="Browse sample images for each ASL class. Click an image to see predictions from both models."
    >
      <div className="mb-6">
        <input
          type="text"
          placeholder="Search classes..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full max-w-xs bg-zinc-800/50 border border-zinc-700 rounded-lg px-4 py-2 text-sm text-white placeholder:text-zinc-600 focus:outline-none focus:border-blue-500/50"
        />
      </div>

      <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 lg:grid-cols-15 gap-2 mb-8">
        {filteredClasses.map((cls) => (
          <motion.button
            key={cls}
            onClick={() => {
              setSelectedClass(selectedClass === cls ? null : cls);
              setSelectedImage(null);
              clear();
            }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`aspect-square rounded-lg flex items-center justify-center text-sm font-mono transition-colors ${
              selectedClass === cls
                ? 'bg-blue-600 text-white border-2 border-blue-400'
                : 'bg-zinc-800/60 text-zinc-400 border border-zinc-700/50 hover:bg-zinc-700/60 hover:text-white'
            }`}
          >
            {cls}
          </motion.button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {selectedClass && (
          <motion.div
            key={selectedClass}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <h3 className="text-sm font-semibold text-white mb-1">
                  Class: <span className="text-blue-400">{selectedClass}</span>
                </h3>
                <p className="text-xs text-zinc-500 mb-4">
                  {classImages.length} sample image{classImages.length !== 1 ? 's' : ''} available.
                  Click to predict.
                </p>

                <div className="grid grid-cols-3 gap-3">
                  {classImages.map((filename) => (
                    <button
                      key={filename}
                      onClick={() => handleImageClick(filename)}
                      className={`relative rounded-lg overflow-hidden border-2 transition-colors aspect-square ${
                        selectedImage === filename
                          ? 'border-blue-500'
                          : 'border-zinc-700 hover:border-zinc-500'
                      }`}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`/samples/${filename}`}
                        alt={filename}
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute bottom-0 inset-x-0 bg-black/60 py-0.5 text-[9px] text-zinc-300 text-center">
                        {filename.includes('test') ? 'test' : 'train'}
                      </div>
                    </button>
                  ))}
                </div>
              </Card>

              <div>
                {selectedImage ? (
                  <PredictionDisplay result={result} loading={loading} />
                ) : (
                  <Card className="h-full flex items-center justify-center">
                    <p className="text-sm text-zinc-600 text-center">
                      Select an image to see predictions
                    </p>
                  </Card>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </Section>
  );
}
