// Service Worker for To-Do PWA
const CACHE_NAME = "todo-pwa-v1";
const ASSETS_TO_CACHE = [
  "./",
  "./index.html",
  "./manifest.json",
  "./icons/icon-192x192.png",
  "./icons/icon-512x512.png",
  // External CDN resources (cache on first use)
  "https://unpkg.com/react@18/umd/react.production.min.js",
  "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js",
  "https://unpkg.com/@babel/standalone/babel.min.js",
  "https://cdn.tailwindcss.com",
  "https://unpkg.com/framer-motion@10.16.4/dist/framer-motion.js",
  "https://cdn.jsdelivr.net/npm/marked/marked.min.js",
  "https://cdn.jsdelivr.net/npm/dompurify@3.0.6/dist/purify.min.js",
  "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap",
];

// Install event - cache core assets
self.addEventListener("install", (event) => {
  console.log("[SW] Installing Service Worker...");
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => {
        console.log("[SW] Caching app shell...");
        return cache
          .addAll(ASSETS_TO_CACHE.filter((url) => !url.startsWith("https://")))
          .then(() => {
            // Try to cache CDN resources, but don't fail if they can't be cached
            return Promise.allSettled(
              ASSETS_TO_CACHE.filter((url) => url.startsWith("https://")).map(
                (url) =>
                  cache
                    .add(url)
                    .catch((err) => console.log("[SW] Failed to cache:", url))
              )
            );
          });
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener("activate", (event) => {
  console.log("[SW] Activating Service Worker...");
  event.waitUntil(
    caches
      .keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== CACHE_NAME) {
              console.log("[SW] Deleting old cache:", cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => self.clients.claim())
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener("fetch", (event) => {
  // Skip non-GET requests
  if (event.request.method !== "GET") return;

  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        // Return cached version
        return cachedResponse;
      }

      // Fetch from network
      return fetch(event.request)
        .then((networkResponse) => {
          // Don't cache non-successful responses
          if (
            !networkResponse ||
            networkResponse.status !== 200 ||
            (networkResponse.type !== "basic" &&
              networkResponse.type !== "cors")
          ) {
            return networkResponse;
          }

          // Clone the response for caching
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });

          return networkResponse;
        })
        .catch(() => {
          // Offline fallback for navigation requests
          if (event.request.mode === "navigate") {
            return caches.match("./index.html");
          }
        });
    })
  );
});

// Handle messages from the app
self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});
