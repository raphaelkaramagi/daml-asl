'use client';

import { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  padding?: boolean;
}

export default function Card({ children, className = '', padding = true }: CardProps) {
  return (
    <div
      className={`rounded-xl border border-zinc-800 bg-zinc-900/70 backdrop-blur-sm ${
        padding ? 'p-5 md:p-6' : ''
      } ${className}`}
    >
      {children}
    </div>
  );
}
