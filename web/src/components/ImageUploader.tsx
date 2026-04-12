'use client';

import { useRef, useCallback, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ImageUploaderProps {
  onImage: (img: HTMLImageElement) => void;
  previewUrl: string | null;
  setPreviewUrl: (url: string | null) => void;
}

export default function ImageUploader({ onImage, previewUrl, setPreviewUrl }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const processFile = useCallback(
    (file: File) => {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      const img = new Image();
      img.onload = () => onImage(img);
      img.src = url;
    },
    [onImage, setPreviewUrl]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file?.type.startsWith('image/')) processFile(file);
    },
    [processFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) processFile(file);
    },
    [processFile]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`relative cursor-pointer rounded-xl border-2 border-dashed transition-colors min-h-[240px] flex items-center justify-center ${
        dragOver
          ? 'border-blue-500 bg-blue-500/5'
          : 'border-zinc-700 hover:border-zinc-500 bg-zinc-900/30'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleChange}
      />

      <AnimatePresence mode="wait">
        {previewUrl ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="relative w-full h-full min-h-[240px] flex items-center justify-center p-2"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={previewUrl}
              alt="Uploaded"
              className="max-h-[300px] rounded-lg object-contain"
            />
            <button
              onClick={(e) => {
                e.stopPropagation();
                setPreviewUrl(null);
              }}
              className="absolute top-3 right-3 w-7 h-7 rounded-full bg-zinc-800/80 text-zinc-400 hover:text-white flex items-center justify-center text-sm"
            >
              x
            </button>
          </motion.div>
        ) : (
          <motion.div
            key="placeholder"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="text-center p-8"
          >
            <svg
              className="w-10 h-10 mx-auto mb-3 text-zinc-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <p className="text-sm text-zinc-400">
              Drop an image here or <span className="text-blue-400">browse</span>
            </p>
            <p className="text-xs text-zinc-600 mt-1">JPG, PNG supported</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
