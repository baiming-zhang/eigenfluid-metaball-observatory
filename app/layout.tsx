import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Eigenfluid Metaball · K2048 Basis Observatory",
  description: "Interactive selected-mode inference for geometry-conditioned K=2048 eigenfluid bases.",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
