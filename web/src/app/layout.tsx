import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.VERCEL_PROJECT_PRODUCTION_URL
      ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
      : process.env.VERCEL_URL
        ? `https://${process.env.VERCEL_URL}`
        : 'http://localhost:3000'
  ),
  title: 'ASL Alphabet Recognition',
  description:
    'Interactive demo of deep learning models for American Sign Language alphabet classification. Two approaches compared: ResNet50 transfer learning and Landmark Neural Network.',
  manifest: '/manifest.json',
  icons: {
    icon: [
      { url: '/favicon.ico', sizes: 'any' },
      { url: '/icons/icon-32x32.png', sizes: '32x32', type: 'image/png' },
      { url: '/icons/icon-16x16.png', sizes: '16x16', type: 'image/png' },
      { url: '/icons/icon-192x192.png', sizes: '192x192', type: 'image/png' },
    ],
    shortcut: '/favicon.ico',
    apple: [
      { url: '/icons/icon-180x180.png', sizes: '180x180', type: 'image/png' },
    ],
  },
  openGraph: {
    title: 'ASL Alphabet Recognition',
    description:
      'Interactive demo comparing ResNet50 and Landmark NN for American Sign Language alphabet classification. Live webcam predictions, training replay, and more.',
    images: [{ url: '/og-image.png', width: 512, height: 512 }],
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'ASL Alphabet Recognition',
    description:
      'Interactive demo comparing deep learning approaches for ASL alphabet classification.',
    images: ['/og-image.png'],
  },
  other: {
    'msapplication-TileColor': '#09090b',
    'msapplication-TileImage': '/icons/icon-144x144.png',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark scroll-smooth">
      <head>
        <meta name="theme-color" content="#09090b" />
      </head>
      <body
        className={`${inter.variable} font-sans bg-zinc-950 text-white antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
