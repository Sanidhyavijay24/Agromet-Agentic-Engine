/**
 * @file server.ts
 * @description Production-ready Bun HTTP server with API reverse proxy and fast asset delivery
 * @module frontend/server
 */

import { serve } from "bun";
import { join } from "path";

const PORT = parseInt(process.env.PORT || "3000", 10);
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

async function buildFrontend() {
  const result = await Bun.build({
    entrypoints: [join(import.meta.dir, "src", "index.tsx")],
    outdir: join(import.meta.dir, "public", "dist"),
    target: "browser",
    naming: {
      entry: "[name].[ext]",
      asset: "[name].[ext]",
    },
    minify: process.env.NODE_ENV === "production",
    sourcemap: "inline",
  });

  if (!result.success) {
    console.error("Bun build failed:", result.logs);
  } else {
    console.log(`[Bun] Frontend bundle compiled successfully.`);
  }
}

// Initial bundle build
await buildFrontend();

const server = serve({
  port: PORT,
  async fetch(req) {
    const url = new URL(req.url);

    // 1. Proxy /api/* requests to FastAPI backend
    if (url.pathname.startsWith("/api/")) {
      const targetUrl = `${BACKEND_URL}${url.pathname}${url.search}`;
      try {
        const proxyRes = await fetch(targetUrl, {
          method: req.method,
          headers: req.headers,
          body: req.method !== "GET" && req.method !== "HEAD" ? await req.blob() : undefined,
        });

        const resHeaders = new Headers(proxyRes.headers);
        resHeaders.set("Access-Control-Allow-Origin", "*");
        resHeaders.set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        resHeaders.set("Access-Control-Allow-Headers", "Content-Type, Authorization");

        return new Response(proxyRes.body, {
          status: proxyRes.status,
          statusText: proxyRes.statusText,
          headers: resHeaders,
        });
      } catch (err: any) {
        return new Response(
          JSON.stringify({ error: "GATEWAY_ERROR", message: "FastAPI Backend is unreachable", details: err.message }),
          { status: 502, headers: { "Content-Type": "application/json" } }
        );
      }
    }

    // 2. Handle static dist files (JS, CSS, SourceMaps)
    if (url.pathname.startsWith("/dist/")) {
      const filename = url.pathname.replace("/dist/", "");
      const filePath = join(import.meta.dir, "public", "dist", filename);
      const file = Bun.file(filePath);
      if (await file.exists()) {
        const contentType = filename.endsWith(".css")
          ? "text/css"
          : filename.endsWith(".js")
          ? "application/javascript"
          : "application/octet-stream";
        return new Response(file, { headers: { "Content-Type": contentType } });
      }
    }

    // 3. Fallback direct index.css if requested
    if (url.pathname === "/index.css") {
      const file = Bun.file(join(import.meta.dir, "src", "index.css"));
      if (await file.exists()) {
        return new Response(file, { headers: { "Content-Type": "text/css" } });
      }
    }

    // 4. Default: Serve public/index.html (SPA routing)
    const indexHtml = Bun.file(join(import.meta.dir, "public", "index.html"));
    return new Response(indexHtml, {
      headers: { "Content-Type": "text/html; charset=utf-8" },
    });
  },
});

console.log(`[A²E Frontend] Pure Bun server listening on http://localhost:${PORT}`);
console.log(`[A²E Frontend] Proxying API calls to ${BACKEND_URL}`);
