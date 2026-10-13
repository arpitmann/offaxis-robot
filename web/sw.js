// Keeps a copy of the robot on the phone, so after the first visit it opens instantly (and works offline).
const CACHE = 'my-robot-v1';
// Only the small files are saved up front. The big ones (tracking engine, model, voices) are saved
// the first time the app itself downloads them, so nothing is ever downloaded twice.
const FILES = ['./', './index.html', './app.js', './manifest.webmanifest', './icon-192.png', './icon-512.png',
  './lib/three.module.js', './lib/three.core.js', './lib/vision_bundle.mjs', './sounds.json', './config.json'];
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => Promise.all(FILES.map((f) => c.add(f).catch(() => {})))).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request).then((res) => {
    if (res.ok && new URL(e.request.url).origin === location.origin) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
    return res;
  })));
});

// Sent by the app once the robot is running: save anything not saved yet (normally nothing is missing).
const BIG = ['./wasm/vision_wasm_internal.js', './wasm/vision_wasm_internal.wasm', './wasm/vision_wasm_nosimd_internal.js',
  './wasm/vision_wasm_nosimd_internal.wasm', './models/face_landmarker.task', './models/pose_landmarker_full.task',
  './models/hand_landmarker.task', './voices/hey_beautiful.wav', './voices/i_love_you.wav', './voices/wake_up.wav'];
self.addEventListener('message', (e) => {
  if (!e.data || !e.data.saveAll) return;
  e.waitUntil(caches.open(CACHE).then(async (c) => {
    let saved = 0;
    for (const f of [...new Set(FILES.concat(BIG, e.data.files || []))]) {
      if (await c.match(f)) { saved++; continue; }
      try { const r = await fetch(f); if (r.ok) { await c.put(f, r); saved++; } } catch (err) {}
    }
    const clients = await self.clients.matchAll();
    clients.forEach((cl) => cl.postMessage({ offlineReady: true, saved }));
  }));
});
