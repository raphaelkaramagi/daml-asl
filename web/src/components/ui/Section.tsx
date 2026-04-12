'use client';

import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface SectionProps {
  id?: string;
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
  dark?: boolean;
}

export default function Section({ id, title, subtitle, children, className = '', dark }: SectionProps) {
  return (
    <section
      id={id}
      className={`py-16 md:py-24 px-4 md:px-8 ${dark ? 'bg-zinc-900/50' : ''} ${className}`}
    >
      <div className="max-w-6xl mx-auto">
        {title && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-100px' }}
            transition={{ duration: 0.5 }}
            className="mb-10 md:mb-14"
          >
            <h2 className="text-2xl md:text-3xl font-bold text-white tracking-tight">
              {title}
            </h2>
            {subtitle && (
              <p className="mt-3 text-zinc-400 text-base md:text-lg max-w-2xl">
                {subtitle}
              </p>
            )}
          </motion.div>
        )}
        {children}
      </div>
    </section>
  );
}
