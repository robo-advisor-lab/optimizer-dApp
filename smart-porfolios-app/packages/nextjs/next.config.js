// @ts-check

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  typescript: {
    ignoreBuildErrors: process.env.NEXT_PUBLIC_IGNORE_BUILD_ERROR === "true",
  },
  eslint: {
    ignoreDuringBuilds: process.env.NEXT_PUBLIC_IGNORE_BUILD_ERROR === "true",
  },
  webpack: config => {
    config.resolve.fallback = { fs: false, net: false, tls: false };
    config.externals.push("pino-pretty", "lokijs", "encoding");
    return config;
  },
  rewrites: async () => {
    return [
      {
        source: "/api/:path*",
        destination: process.env.NODE_ENV === "development" ? "http://127.0.0.1:5328/api/:path*" : "/api/",
      },
    ];
  },
};

module.exports = nextConfig;
