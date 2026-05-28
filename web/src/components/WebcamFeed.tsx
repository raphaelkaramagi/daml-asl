'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useWebcam } from '@/hooks/useWebcam';
import LandmarkVisualizer from './LandmarkVisualizer';
import { detectHandWithHold, resetTemporalHold } from '@/lib/landmarks';
import {
  predictWithLandmarkNN,
  predictWithResnet,
  getTopPredictions,
  loadPreprocessing,
} from '@/lib/models';
import {
  canvasToResnetImageData,
  drawSourceToCanvas,
  getPaddedBBox,
  SMALL_HAND_AREA_THRESHOLD,
} from '@/lib/image-utils';
import { useAppStore } from '@/store/app-store';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';
import type { PredictionResult } from '@/hooks/usePrediction';

interface WebcamFeedProps {
  onPrediction: (result: PredictionResult) => void;
}

export default function WebcamFeed({ onPrediction }: WebcamFeedProps) {
  const { videoRef, active, error, start, stop } = useWebcam();
  const containerRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number>(0);
  const isVisibleRef = useRef(true);
  const [landmarks, setLandmarks] = useState<NormalizedLandmark[] | null>(null);
  const [displaySize, setDisplaySize] = useState({ width: 640, height: 480 });
  const [detectionStatus, setDetectionStatus] = useState<
    'none' | 'detected' | 'held' | 'missing' | 'small'
  >('none');
  const lastPredTime = useRef(0);
  const classesRef = useRef<string[]>([]);

  useEffect(() => {
    loadPreprocessing().then((p) => {
      classesRef.current = p.classes;
    });
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        isVisibleRef.current = entry.isIntersecting && entry.intersectionRatio >= 0.25;
      },
      { threshold: [0, 0.25, 0.5, 1] }
    );
    observer.observe(container);
    return () => observer.disconnect();
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

    if (!isVisibleRef.current) {
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
      const handResult = detectHandWithHold(video, { mirror: true });
      setLandmarks(handResult?.landmarks ?? null);

      let status: typeof detectionStatus = 'missing';
      if (handResult) {
        const bbox = getPaddedBBox(handResult.landmarks);
        if (bbox.area < SMALL_HAND_AREA_THRESHOLD) {
          status = 'small';
        } else if (handResult.held) {
          status = 'held';
        } else {
          status = 'detected';
        }
      }
      setDetectionStatus(status);

      if (handResult || (enableResnet && resnetLoaded)) {
        const result: PredictionResult = {
          handDetection: handResult,
          handTooSmall: status === 'small',
          detectionHeld: handResult?.held,
        };

        const classes = classesRef.current;

        if (enableLandmark && landmarkLoaded && handResult) {
          const pred = predictWithLandmarkNN(handResult.features);
          result.landmark = pred;
          result.landmarkTop3 = getTopPredictions(pred.allConfidences, classes);
        }

        if (enableResnet && resnetLoaded) {
          const canvas = drawSourceToCanvas(video, { mirror: true });
          const imageData = canvasToResnetImageData(
            canvas,
            handResult?.landmarks
          );
          const pred = predictWithResnet(imageData);
          result.resnet = pred;
          result.resnetTop3 = getTopPredictions(pred.allConfidences, classes);
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
      setDetectionStatus('none');
      resetTemporalHold();
    } else {
      start();
    }
  }, [active, start, stop]);

  const statusLabel = {
    none: '',
    detected: 'Hand detected',
    held: 'Hand detected',
    missing: 'No hand — center hand in frame',
    small: 'Move hand closer',
  }[detectionStatus];

  const statusColor = {
    none: '',
    detected: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
    held: 'text-emerald-400/80 bg-emerald-500/10 border-emerald-500/30',
    missing: 'text-amber-400 bg-amber-500/10 border-amber-500/30',
    small: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
  }[detectionStatus];

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
            landmarks={landmarks}
            width={displaySize.width}
            height={displaySize.height}
          />
        )}
        {active && (
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
            <div className="w-[55%] aspect-square border border-dashed border-zinc-600/50 rounded-2xl" />
          </div>
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
          <div className="absolute top-3 left-3 flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              <span className="text-[10px] text-zinc-400 uppercase tracking-wider">
                Live
              </span>
            </div>
            {statusLabel && (
              <span
                className={`text-[10px] px-2 py-0.5 rounded border ${statusColor}`}
              >
                {statusLabel}
              </span>
            )}
          </div>
        )}
        {active && detectionStatus === 'missing' && (
          <p className="absolute bottom-3 left-0 right-0 text-center text-[10px] text-zinc-500 pointer-events-none">
            Center your hand, fill ~40% of frame
          </p>
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
