'use client';

import { useEffect, useRef } from 'react';
import { HAND_CONNECTIONS } from '@/lib/constants';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

interface LandmarkVisualizerProps {
  landmarks: NormalizedLandmark[] | null;
  width: number;
  height: number;
  className?: string;
  showLabels?: boolean;
}

const FINGER_COLORS = [
  '#ef4444', // thumb - red
  '#f59e0b', // index - amber
  '#22c55e', // middle - green
  '#3b82f6', // ring - blue
  '#a855f7', // pinky - purple
];

function getFingerIndex(landmarkIdx: number): number {
  if (landmarkIdx <= 4) return 0;
  if (landmarkIdx <= 8) return 1;
  if (landmarkIdx <= 12) return 2;
  if (landmarkIdx <= 16) return 3;
  return 4;
}

export default function LandmarkVisualizer({
  landmarks,
  width,
  height,
  className = '',
  showLabels = false,
}: LandmarkVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    if (!landmarks || landmarks.length === 0) return;

    // Draw connections
    for (const [i, j] of HAND_CONNECTIONS) {
      const a = landmarks[i];
      const b = landmarks[j];
      const fingerIdx = getFingerIndex(Math.max(i, j));
      ctx.beginPath();
      ctx.moveTo(a.x * width, a.y * height);
      ctx.lineTo(b.x * width, b.y * height);
      ctx.strokeStyle = FINGER_COLORS[fingerIdx];
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.7;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Draw landmarks
    landmarks.forEach((lm, idx) => {
      const x = lm.x * width;
      const y = lm.y * height;
      const fingerIdx = getFingerIndex(idx);
      const radius = idx === 0 ? 5 : 3;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = idx === 0 ? '#ffffff' : FINGER_COLORS[fingerIdx];
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();

      if (showLabels) {
        ctx.font = '9px monospace';
        ctx.fillStyle = '#fff';
        ctx.fillText(String(idx), x + 5, y - 5);
      }
    });
  }, [landmarks, width, height, showLabels]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className={`absolute top-0 left-0 pointer-events-none ${className}`}
      style={{ width, height }}
    />
  );
}
