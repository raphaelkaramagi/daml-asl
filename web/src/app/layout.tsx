import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: 'ASL Alphabet Recognition',
  description:
    'Interactive demo of deep learning models for American Sign Language alphabet classification. Two approaches compared: ResNet50 transfer learning and Landmark Neural Network.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark scroll-smooth">
      <body
        className={`${inter.variable} font-sans bg-zinc-950 text-white antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
