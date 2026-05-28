'use client';

import { useRef, useEffect, useCallback, useState } from 'react';
import { motion } from 'framer-motion';
import { useWebcam } from '@/hooks/useWebcam';
import LandmarkVisualizer from './LandmarkVisualizer';
import { detectHandFromVideo, setRunningMode } from '@/lib/landmarks';
import {
  predictWithLandmarkNN,
  predictWithResnet,
  getTopPredictions,
  loadPreprocessing,
} from '@/lib/models';
import { CLASS_NAMES } from '@/lib/constants';
import { useAppStore } from '@/store/app-store';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import type { PredictionResult } from '@/hooks/usePrediction';

interface WebcamFeedProps {
  onPrediction: (result: PredictionResult) => void;
}

function OverlayBadge({
  children,
  variant = 'neutral',
}: {
  children: React.ReactNode;
  variant?: 'live' | 'success' | 'warn' | 'neutral';
}) {
  const styles = {
    live: 'bg-black/75 text-white border-white/20',
    success: 'bg-black/80 text-emerald-300 border-emerald-400/40',
    warn: 'bg-black/80 text-amber-200 border-amber-400/40',
    neutral: 'bg-black/75 text-zinc-100 border-white/20',
  }[variant];

  return (
    <span
      className={`inline-flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-md border backdrop-blur-md shadow-lg ${styles}`}
    >
      {children}
    </span>
  );
}

export default function WebcamFeed({ onPrediction }: WebcamFeedProps) {
  const { videoRef, active, error, start, stop } = useWebcam();
  const containerRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const [landmarks, setLandmarks] = useState<NormalizedLandmark[] | null>(null);
  const [displaySize, setDisplaySize] = useState({ width: 640, height: 480 });
  const [handDetected, setHandDetected] = useState(false);
  const lastPredTime = useRef(0);
  const classesRef = useRef<string[]>([...CLASS_NAMES]);

  useEffect(() => {
    loadPreprocessing().then((p) => {
      classesRef.current = p.classes;
    });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver(() => {
      const video = videoRef.current;
      if (video && video.videoWidth > 0) {
        setDisplaySize({ width: video.clientWidth, height: video.clientHeight });
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [videoRef]);

  const processFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) {
      animRef.current = requestAnimationFrame(processFrame);
      return;
    }

    if (video.clientWidth > 0) {
      setDisplaySize({ width: video.clientWidth, height: video.clientHeight });
    }

    const now = performance.now();
    if (now - lastPredTime.current < 200) {
      animRef.current = requestAnimationFrame(processFrame);
      return;
    }
    lastPredTime.current = now;

    const { enableResnet, enableLandmark, resnetLoaded, landmarkLoaded } =
      useAppStore.getState();

    try {
      const handResult = detectHandFromVideo(video, now);
      setLandmarks(handResult?.landmarks ?? null);
      setHandDetected(handResult !== null);

      if (handResult || (enableResnet && resnetLoaded)) {
        const result: PredictionResult = {
          handDetection: handResult,
        };

        if (enableLandmark && landmarkLoaded && handResult) {
          const pred = predictWithLandmarkNN(handResult.features);
          result.landmark = pred;
          result.landmarkTop3 = getTopPredictions(
            pred.allConfidences,
            classesRef.current
          );
        }

        if (enableResnet && resnetLoaded) {
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext('2d')!;
          ctx.drawImage(video, 0, 0);
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const pred = predictWithResnet(imageData);
          result.resnet = pred;
          result.resnetTop3 = getTopPredictions(pred.allConfidences, [...CLASS_NAMES]);
        }

        onPrediction(result);
      }
    } catch {
      // prediction errors are non-fatal
    }

    animRef.current = requestAnimationFrame(processFrame);
  }, [videoRef, onPrediction]);

  useEffect(() => {
    if (active) {
      setRunningMode('VIDEO');
      animRef.current = requestAnimationFrame(processFrame);
    }
    return () => {
      cancelAnimationFrame(animRef.current);
    };
  }, [active, processFrame]);

  const handleToggle = useCallback(() => {
    if (active) {
      stop();
      setLandmarks(null);
      setHandDetected(false);
      setRunningMode('IMAGE');
    } else {
      start();
    }
  }, [active, start, stop]);

  return (
    <div className="space-y-3">
      <div
        ref={containerRef}
        className="relative rounded-xl overflow-hidden bg-zinc-900 border border-zinc-800 min-h-[240px]"
      >
        <video
          ref={videoRef}
          className="w-full h-auto"
          muted
          playsInline
          style={{ transform: 'scaleX(-1)' }}
        />
        {active && (
          <LandmarkVisualizer
            landmarks={
              landmarks ? landmarks.map((lm) => ({ ...lm, x: 1 - lm.x })) : null
            }
            width={displaySize.width}
            height={displaySize.height}
          />
        )}
        {!active && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/90">
            <div className="text-center">
              <svg
                className="w-12 h-12 mx-auto mb-3 text-zinc-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              <p className="text-sm text-zinc-500">Camera off</p>
            </div>
          </div>
        )}
        {active && (
          <div className="absolute top-3 left-3 flex flex-col items-start gap-2 pointer-events-none">
            <OverlayBadge variant="live">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse shrink-0" />
              Live
            </OverlayBadge>
            {handDetected ? (
              <OverlayBadge variant="success">Hand detected</OverlayBadge>
            ) : (
              <OverlayBadge variant="warn">No hand detected</OverlayBadge>
            )}
          </div>
        )}
      </div>

      <motion.button
        onClick={handleToggle}
        whileTap={{ scale: 0.97 }}
        className={`w-full py-2.5 rounded-lg font-medium text-sm transition-colors ${
          active
            ? 'bg-red-600/20 text-red-400 border border-red-600/30 hover:bg-red-600/30'
            : 'bg-blue-600/20 text-blue-400 border border-blue-600/30 hover:bg-blue-600/30'
        }`}
      >
        {active ? 'Stop Camera' : 'Start Camera'}
      </motion.button>

      {error && <p className="text-xs text-red-400 text-center">{error}</p>}
    </div>
  );
}
