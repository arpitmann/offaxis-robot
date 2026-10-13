// =====================================================================
//  My Robot - web / Android version of the off-axis robot
//  Three.js draws the 3D world, MediaPipe (face + body + hand trackers) watches you through
//  the front camera, Web Audio makes the beeps and plays the voice lines.
// =====================================================================
import * as THREE from './lib/three.module.js';
import { FilesetResolver, FaceLandmarker, PoseLandmarker, HandLandmarker } from './lib/vision_bundle.mjs';

// ----------------------------- tweakables -----------------------------
const S = {
  EYE_DIST: 8.0,       // your eye's distance from the screen (world units)
  HEAD_GAIN: 5.0,      // camera motion -> world motion
  SMOOTH: 0.6,         // head smoothing (lower = smoother but laggier)
  LOOK_SMOOTH: 0.3,    // how quickly the robot turns to look at you
  ARM_SMOOTH: 0.75, FINGER_SMOOTH: 0.7, FACE_SMOOTH: 0.7, BODY_SMOOTH: 0.5, LEG_SMOOTH: 0.7,
  WALK_GAIN_X: 2.2, WALK_GAIN_Z: 3.0, JUMP_GAIN: 1.0,
  HOLD_TIME: 0.35,     // seconds a gesture must be held before it counts
  CLAP_CLOSE: 0.55,    // wrists closer than this x shoulder-width = clap
  BLINK_CLOSED: 0.45,  // eye counts as shut below this fraction of its normal openness
  SLEEP_TIME: 2.0,     // seconds with both eyes shut before WAKE UP
  YAWN_JAW: 0.7,       // how wide your jaw must open to count as a yawn (0..1)
  PUCKER: 0.55,        // how strong a pucker counts as a kiss (0..1)
  NOD_THR: 0.03, SHAKE_THR: 0.05, GAZE_GAIN: 6.0,
  SOUND_VOLUME: 0.6,
};
const GROUND = -1.8, ROBOT_DEPTH = 3.8, THIGH = 0.33, SHIN = 0.31, FOOT_H = 0.06;
// Three separate trackers (the all-in-one "holistic" one isn't fully supported in browsers on phones).
const GS = 'https://storage.googleapis.com/mediapipe-models';
const MODELS = {
  face: { name: 'face', local: './models/face_landmarker.task', remote: `${GS}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task` },
  pose: { name: 'body', local: './models/pose_landmarker_full.task', remote: `${GS}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task` },
  hand: { name: 'hands', local: './models/hand_landmarker.task', remote: `${GS}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task` },
};
const VOICE_LINES = [['hey_beautiful', 'HEY BEAUTIFUL!'], ['i_love_you', 'I LOVE YOU!'], ['wake_up', 'WAKE UP!']];

// ----------------------------- small maths -----------------------------
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const n = Math.hypot(a[0], a[1], a[2]); return n < 1e-9 ? [0, -1, 0] : [a[0] / n, a[1] / n, a[2] / n]; };
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const wrap = (a) => ((((a + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const lerpDir = (a, b, k) => norm(add(a, mul(sub(b, a), k)));
// Same order as THREE.Euler 'YXZ': roll about z, then pitch about x, then yaw about y.
function rot(v, yaw, pitch = 0, roll = 0) {
  let [x, y, z] = v;
  if (roll) { const c = Math.cos(roll), s = Math.sin(roll); [x, y] = [x * c - y * s, x * s + y * c]; }
  if (pitch) { const c = Math.cos(pitch), s = Math.sin(pitch); [y, z] = [y * c - z * s, y * s + z * c]; }
  if (yaw) { const c = Math.cos(yaw), s = Math.sin(yaw); [x, z] = [x * c + z * s, -x * s + z * c]; }
  return [x, y, z];
}
function rng(seed) { return () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function noise1d(n, scale, rand, octaves = 5) {
  const out = new Float32Array(n); let amp = 1, freq = 1 / scale, tot = 0;
  for (let o = 0; o < octaves; o++) {
    const pts = Array.from({ length: Math.floor(n * freq) + 3 }, rand);
    for (let i = 0; i < n; i++) { const xi = i * freq, i0 = Math.floor(xi), f = (1 - Math.cos((xi - i0) * Math.PI)) / 2; out[i] += amp * (pts[i0] * (1 - f) + pts[i0 + 1] * f); }
    tot += amp; amp *= 0.5; freq *= 2;
  }
  let lo = Infinity, hi = -Infinity; for (const v of out) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  for (let i = 0; i < n; i++) out[i] = (out[i] - lo) / (hi - lo + 1e-9);
  return out;
}
const lerpColor = (a, b, t) => a.clone().lerp(b, t);

// ============================== the world ==============================
function canvasTex(w, h, draw) {
  const c = document.createElement('canvas'); c.width = w; c.height = h; draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}

function buildWorld(scene) {
  const rand = rng(7);
  const SKY_TOP = new THREE.Color('#3a70ba'), SKY_H = new THREE.Color('#c4d8e8'), HAZE = new THREE.Color('#bacee0');
  scene.fog = new THREE.Fog(HAZE, 60, 1100);

  // sky dome (infinitely far away, so it never shifts with you)
  const skyGeo = new THREE.SphereGeometry(2000, 32, 20);
  const cols = []; const p = skyGeo.attributes.position;
  for (let i = 0; i < p.count; i++) { const e = Math.max(0, p.getY(i) / 2000); const c = lerpColor(SKY_H, SKY_TOP, Math.pow(Math.min(1, e * 3.2), 0.7)); cols.push(c.r, c.g, c.b); }
  skyGeo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  scene.add(new THREE.Mesh(skyGeo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide, fog: false, depthWrite: false })));

  // lights: soft sky light + a warm sun behind you, up and to the left
  scene.add(new THREE.HemisphereLight(0xdbe8ff, 0x50703c, 1.1));
  const sun = new THREE.DirectionalLight(0xfff1dc, 1.7); sun.position.set(-5, 8, 9); scene.add(sun);

  // ground
  const grassTex = canvasTex(256, 256, (g, w, h) => {
    g.fillStyle = '#4f8a36'; g.fillRect(0, 0, w, h);
    for (let i = 0; i < 2600; i++) { const v = rand(); g.fillStyle = `rgba(${30 + v * 70},${80 + v * 80},${25 + v * 30},0.35)`; g.fillRect(rand() * w, rand() * h, 2 + rand() * 4, 2 + rand() * 6); }
  });
  grassTex.wrapS = grassTex.wrapT = THREE.RepeatWrapping; grassTex.repeat.set(500, 500); grassTex.anisotropy = 4;
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), new THREE.MeshLambertMaterial({ map: grassTex }));
  ground.rotation.x = -Math.PI / 2; ground.position.set(0, GROUND, -1900); scene.add(ground);

  // mountain ranges (flat cut-outs at real distances: the far ones barely move, the near ones slide)
  function range(depth, hLo, hHi, top, base, ridged, snow, scale) {
    const width = depth * 5 + 200, n = 600, nz = noise1d(n, scale, rand, 6);
    const pos = [], col = [], idx = [];
    const cTop = new THREE.Color(top), cBase = new THREE.Color(base), cSnow = new THREE.Color('#eef2f8');
    for (let i = 0; i < n; i++) {
      let v = nz[i]; if (ridged) v = Math.pow(1 - Math.abs(2 * v - 1), 1.4);
      const x = -width / 2 + (width * i) / (n - 1), hgt = hLo + (hHi - hLo) * v;
      pos.push(x, GROUND + hgt, -depth, x, GROUND - 0.5, -depth);
      const ct = snow && v > 0.6 ? cTop.clone().lerp(cSnow, 0.85) : cTop;
      col.push(ct.r, ct.g, ct.b, cBase.r, cBase.g, cBase.b);
      if (i > 0) { const a = (i - 1) * 2; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3)); g.setIndex(idx);
    scene.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true })));
  }
  range(520, 45, 125, '#96a5be', '#5f7391', true, true, 70);
  range(180, 10, 26, '#5a7078', '#465f5c', true, false, 60);
  range(72, 1.6, 5.2, '#58804a', '#3e643a', false, false, 90);

  // pine trees: three stacked cones + a trunk, instanced so phones stay fast
  const trees = [];
  for (const [depth, dens, lo, hi] of [[45, 1.0, 3.0, 5.5], [32, 0.8, 3.5, 6.5]]) {
    const half = depth * 1.4 + 20; let x = -half;
    while (x < half) { const h = lo + rand() * (hi - lo); trees.push([x, depth + rand() * 3, h]); x += h * (0.15 + rand() * 0.3) / dens; }
  }
  for (const [x, depth, h] of [[-15, 26, 8], [-8.5, 26, 7], [9, 26, 7.5], [16, 26, 8.5], [-11, 19, 8], [-14.5, 19, 6.5],
    [11.5, 19, 9], [7.2, 19, 6], [7.4, 13, 8.5], [11.5, 13, 7], [-6.3, 9.5, 7.5]]) trees.push([x, depth, h]);
  const cone = new THREE.ConeGeometry(1, 1, 9), trunk = new THREE.CylinderGeometry(1, 1, 1, 6);
  const leaf = new THREE.InstancedMesh(cone, new THREE.MeshLambertMaterial({ color: 0xffffff }), trees.length * 3);
  const wood = new THREE.InstancedMesh(trunk, new THREE.MeshLambertMaterial({ color: 0x46342a }), trees.length);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), sc = new THREE.Vector3(), ps = new THREE.Vector3(), c = new THREE.Color();
  trees.forEach(([x, depth, h], i) => {
    m.compose(ps.set(x, GROUND + h * 0.06, -depth), q, sc.set(h * 0.035, h * 0.12, h * 0.035)); wood.setMatrixAt(i, m);
    for (let k = 0; k < 3; k++) {
      const w = h * (0.3 - 0.07 * k), hh = h * (0.45 - 0.06 * k), y = GROUND + h * (0.1 + 0.27 * k) + hh / 2;
      q.setFromEuler(new THREE.Euler(0, rand() * 6, 0));
      m.compose(ps.set(x, y, -depth), q, sc.set(w, hh, w)); leaf.setMatrixAt(i * 3 + k, m);
      leaf.setColorAt(i * 3 + k, c.setHSL(0.36 + rand() * 0.04, 0.42, 0.15 + 0.05 * k + rand() * 0.04));
    }
    q.identity();
  });
  scene.add(leaf, wood);

  // grass blades that sway (a tiny bit of shader code bends their tips)
  const blade = new THREE.BufferGeometry();
  blade.setAttribute('position', new THREE.Float32BufferAttribute([-0.012, 0, 0, 0.012, 0, 0, 0, 1, 0], 3));
  blade.computeVertexNormals();
  const bladeMat = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide });
  const uTime = { value: 0 };
  bladeMat.onBeforeCompile = (sh) => {
    sh.uniforms.uTime = uTime;
    sh.vertexShader = 'uniform float uTime;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      `#include <begin_vertex>
       vec4 wpos = instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
       transformed.x += position.y * 0.035 * sin(1.6 * uTime + wpos.x * 0.3 + wpos.z * 1.7);`);
  };
  const NB = 2600, grass = new THREE.InstancedMesh(blade, bladeMat, NB);
  for (let i = 0; i < NB; i++) {
    const depth = -1.8 + 20 * Math.pow(rand(), 1.8), half = 1.2 * (S.EYE_DIST + depth) + 2, x = -half + 2 * half * rand();
    q.setFromEuler(new THREE.Euler(0, rand() * Math.PI, (rand() - 0.5) * 0.5));
    m.compose(ps.set(x, GROUND, -depth), q, sc.set(1, 0.12 + rand() * 0.2, 1)); grass.setMatrixAt(i, m);
    grass.setColorAt(i, c.setHSL(0.27 + rand() * 0.06, 0.5, 0.2 + rand() * 0.25));
  }
  scene.add(grass);
  const flowers = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(0.04, 0), new THREE.MeshLambertMaterial({ color: 0xffffff }), 160);
  const fcols = ['#faf9f5', '#fad646', '#ba78dc', '#f06e78'];
  for (let i = 0; i < 160; i++) {
    const depth = -1 + 18 * Math.pow(rand(), 1.6), half = 1.1 * (S.EYE_DIST + depth), x = -half + 2 * half * rand();
    m.compose(ps.set(x, GROUND + 0.12 + rand() * 0.08, -depth), q.identity(), sc.set(1, 1, 1)); flowers.setMatrixAt(i, m);
    flowers.setColorAt(i, c.set(fcols[i % 4]));
  }
  scene.add(flowers);

  // clouds drifting across the sky
  const cloudTex = canvasTex(256, 110, (g, w, h) => {
    for (let k = 0; k < 2; k++) for (let i = 0; i < 14; i++) {
      const x = w * (0.2 + 0.6 * rand()), y = h * (0.45 + 0.25 * rand()), r = w * (0.1 + 0.1 * rand());
      const gr = g.createRadialGradient(x, y, 0, x, y, r);
      gr.addColorStop(0, k ? 'rgba(255,255,255,0.75)' : 'rgba(210,218,232,0.6)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
    }
  });
  const clouds = [];
  for (let i = 0; i < 9; i++) {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, fog: false, depthWrite: false, transparent: true }));
    const w = 140 + rand() * 160; sp.scale.set(w, w * 0.43, 1);
    sp.userData = { x0: -1100 + rand() * 2200, y: 90 + rand() * 170, speed: 3 + rand() * 6 };
    sp.position.z = -1100; scene.add(sp); clouds.push(sp);
  }
  return {
    update(t) {
      uTime.value = t;
      for (const sp of clouds) { const u = sp.userData; sp.position.x = ((u.x0 + u.speed * t + 1300) % 2600) - 1300; sp.position.y = u.y; }
    },
  };
}

// ============================== the robot ==============================
const FINGER_BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20], [5, 9], [9, 13], [13, 17]];
const TIPS = new Set([4, 8, 12, 16, 20]);
const UP = new THREE.Vector3(0, 1, 0);
const BOX = new THREE.BoxGeometry(1, 1, 1);

// The guitar: body (two rounded lobes), sound hole, neck, head and six strings. Its neck points along +x.
function buildGuitar(scene) {
  const g = new THREE.Group();
  const wood = new THREE.MeshStandardMaterial({ color: 0xc8783c, roughness: 0.45, metalness: 0.05 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x5a3a22, roughness: 0.6 });
  const black = new THREE.MeshBasicMaterial({ color: 0x1a1210 }), str = new THREE.MeshBasicMaterial({ color: 0xeeeeee });
  for (const [r, x] of [[0.27, -0.06], [0.21, 0.2]]) { const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.1, 28), wood); m.rotation.x = Math.PI / 2; m.position.set(x, 0, 0); g.add(m); }
  const part = (mat, sx, sy, sz, x, y, z) => { const m = new THREE.Mesh(BOX, mat); m.scale.set(sx, sy, sz); m.position.set(x, y, z); g.add(m); };
  const hole = new THREE.Mesh(new THREE.CircleGeometry(0.07, 20), black); hole.position.set(0.1, 0, 0.051); g.add(hole);
  part(dark, 0.7, 0.07, 0.04, 0.62, 0, 0.03); part(dark, 0.15, 0.1, 0.04, 1.04, 0, 0.03); part(dark, 0.04, 0.16, 0.02, -0.16, 0, 0.058);
  for (let i = 0; i < 6; i++) part(str, 1.12, 0.004, 0.004, 0.44, -0.025 + i * 0.01, 0.062);
  g.visible = false; scene.add(g); return g;
}
const GUITAR_TILT = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.15, 0, 0.42));

// Two-bone arm solver: shoulder S reaches target T with upper/fore lengths a, b; the elbow bends toward "pole".
function ik2(S, T, a, b, pole) {
  const d = sub(T, S), u = norm(d); let L = Math.hypot(...d);
  L = clamp(L, Math.abs(a - b) + 1e-3, a + b - 1e-3);
  const cosA = (a * a + L * L - b * b) / (2 * a * L), sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
  const w = norm(sub(pole, mul(u, dot(pole, u)))), E = add(S, add(mul(u, a * cosA), mul(w, a * sinA)));
  return [norm(sub(E, S)), norm(sub(add(S, mul(u, L)), E))];
}

class Robot {
  constructor(scene) {
    const std = (c, extra = {}) => new THREE.MeshStandardMaterial({ color: c, roughness: 0.5, metalness: 0.25, ...extra });
    this.mat = { grey: std(0xafb6c2), dark: std(0x62687a), finger: std(0xcdd3de), tip: std(0x78c8e6),
      red: std(0xff3c3c, { emissive: 0x551010 }), chest: std(0x000000, { emissive: 0xf08c32, emissiveIntensity: 1.2 }) };
    const mk = (mat) => { const o = new THREE.Mesh(BOX, mat); scene.add(o); return o; };
    // head with a face drawn on a texture every frame
    this.faceCanvas = document.createElement('canvas'); this.faceCanvas.width = 256; this.faceCanvas.height = 215;
    this.glowCanvas = document.createElement('canvas'); this.glowCanvas.width = 256; this.glowCanvas.height = 215;
    this.faceTex = new THREE.CanvasTexture(this.faceCanvas); this.faceTex.colorSpace = THREE.SRGBColorSpace;
    this.glowTex = new THREE.CanvasTexture(this.glowCanvas); this.glowTex.colorSpace = THREE.SRGBColorSpace;
    const faceMat = std(0xffffff, { map: this.faceTex, emissive: 0xffffff, emissiveMap: this.glowTex, emissiveIntensity: 1.0, metalness: 0.1 });
    const g = this.mat.grey;
    this.head = new THREE.Mesh(BOX, [g, g, g, g, faceMat, g]); scene.add(this.head);
    this.p = {
      torso: mk(g), pelvis: mk(this.mat.dark), neck: mk(this.mat.dark), chest: mk(this.mat.chest),
      antenna: mk(this.mat.dark), tip: mk(this.mat.red),
    };
    this.limb = {};
    for (const s of [-1, 1]) {
      this.limb[s] = {
        shoulder: mk(g), upper: mk(this.mat.dark), fore: mk(this.mat.dark), fist: mk(g), palm: mk(g),
        bones: FINGER_BONES.map(([, b]) => mk(TIPS.has(b) ? this.mat.tip : this.mat.finger)),
        thigh: mk(this.mat.dark), knee: mk(g), shin: mk(this.mat.dark), foot: mk(g),
      };
    }
    this.shadow = new THREE.Mesh(new THREE.CircleGeometry(1, 32), new THREE.MeshBasicMaterial({
      map: canvasTex(64, 64, (c, w, h) => { const gr = c.createRadialGradient(32, 32, 0, 32, 32, 32); gr.addColorStop(0, 'rgba(20,40,15,0.55)'); gr.addColorStop(1, 'rgba(20,40,15,0)'); c.fillStyle = gr; c.fillRect(0, 0, w, h); }),
      transparent: true, depthWrite: false }));
    this.shadow.rotation.x = -Math.PI / 2; scene.add(this.shadow);
    this.guitar = buildGuitar(scene);
    const bark = new THREE.MeshStandardMaterial({ color: 0x6b4a2e, roughness: 0.9 }), ring = new THREE.MeshStandardMaterial({ color: 0xc9a26b, roughness: 0.8 });
    this.stump = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.34, 1, 14), [bark, ring, bark]); this.stump.visible = false; scene.add(this.stump);
    this.sit = 0; this.strum = 0; this.chord = 0; this.bob = 0; this.footTap = 0;   // guitar-playing state

    // pose state
    this.fingers = { '-1': null, '1': null }; this.fingerSeen = { '-1': -9, '1': -9 };
    this.arm = {}; this.leg = {};
    for (const s of [-1, 1]) {
      this.arm[s] = [norm([0.12 * s, -1, 0]), norm([0.12 * s, -1, 0])];
      this.leg[s] = [norm([0.06 * s, -1, 0]), [0, -1, 0], norm([0, -0.3, 1])];
    }
    this.torsoYaw = 0; this.hipYaw = 0; this.bend = [0, 0]; this.tracked = false;
    this.rootX = 0; this.rootDepth = ROBOT_DEPTH; this.jump = 0; this.refTorso = null; this.fullSeen = -99; this.feetHist = [];
    this.root = [0, GROUND + 0.7, -ROBOT_DEPTH]; this.headC = [0, 0.8, -ROBOT_DEPTH];
    this.hands = { '-1': [-0.7, -0.8, -ROBOT_DEPTH], '1': [0.7, -0.8, -ROBOT_DEPTH] };
    this.headTop = [0, 1.8, -ROBOT_DEPTH]; this.mouth = [0, 0.6, -ROBOT_DEPTH + 0.5];
  }

  update(body, t) {
    const b = body || {};
    if (b.torsoYaw != null) {
      this.tracked = true;
      const hy = b.hipYaw != null ? b.hipYaw : b.torsoYaw;
      this.torsoYaw += wrap(b.torsoYaw - this.torsoYaw) * S.BODY_SMOOTH;
      this.hipYaw += wrap(hy - this.hipYaw) * S.BODY_SMOOTH;
    } else { this.tracked = false; this.torsoYaw *= 1 - S.BODY_SMOOTH * 0.3; this.hipYaw *= 1 - S.BODY_SMOOTH * 0.3; }
    const [bp, br] = b.bend || [0, 0];
    this.bend = [this.bend[0] + (bp - this.bend[0]) * S.BODY_SMOOTH, this.bend[1] + (br - this.bend[1]) * S.BODY_SMOOTH];

    for (const s of [-1, 1]) {
      let tgt = b.arms && b.arms[s];
      if (!tgt) { const idle = rot(norm([0.12 * s, -1, 0.08 * Math.sin(t * 2 + s)]), this.torsoYaw, this.bend[0], this.bend[1]); tgt = [idle, idle]; }
      this.arm[s] = [lerpDir(this.arm[s][0], tgt[0], S.ARM_SMOOTH), lerpDir(this.arm[s][1], tgt[1], S.ARM_SMOOTH)];
      const hand = b.hands && b.hands[s];
      if (hand) {   // copy your hand's shape, scaled to robot size, around the robot's wrist
        const [x0, y0, z0] = hand[0], k = 0.46 / (Math.hypot(hand[9][0] - x0, hand[9][1] - y0, hand[9][2] - z0) + 1e-6);
        const nw = hand.map(([x, y, z]) => [(x - x0) * k, -(y - y0) * k, -(z - z0) * k]);
        const old = this.fingers[s];
        this.fingers[s] = old ? old.map((o, i) => add(o, mul(sub(nw[i], o), S.FINGER_SMOOTH))) : nw;
        this.fingerSeen[s] = t;
      } else if (t - this.fingerSeen[s] > 0.4) this.fingers[s] = null;

      let lt = b.legs && b.legs[s];
      const footDef = rot(norm([0, -0.3, 1]), this.hipYaw);
      if (!lt) lt = [rot(norm([0.06 * s, -1, 0]), this.hipYaw), [0, -1, 0], footDef];
      else if (!lt[2]) lt = [lt[0], lt[1], footDef];
      this.leg[s] = this.leg[s].map((c, i) => lerpDir(c, lt[i], S.LEG_SMOOTH));
    }
    if (b.full) {   // walking around and jumping, only with your whole body in view
      this.fullSeen = t;
      if (this.refTorso == null) this.refTorso = b.torsoLen;
      const tx = clamp(b.posX * S.WALK_GAIN_X, -2.4, 2.4);
      const td = clamp(ROBOT_DEPTH + (this.refTorso / Math.max(b.torsoLen, 1e-3) - 1) * S.WALK_GAIN_Z, 2.6, 7);
      this.rootX += (tx - this.rootX) * S.BODY_SMOOTH; this.rootDepth += (td - this.rootDepth) * S.BODY_SMOOTH;
      this.feetHist = this.feetHist.filter(([tt]) => t - tt < 0.6); this.feetHist.push([t, b.feetY]);
      const floor = Math.max(...this.feetHist.map(([, v]) => v));
      const rise = (floor - b.feetY) / Math.max(b.legLen, 1e-3);
      const tj = Math.max(0, rise - 0.03) * 0.7 * 1.6 * S.JUMP_GAIN;
      this.jump += (tj - this.jump) * 0.7;
    } else {
      if (t - this.fullSeen > 10) this.refTorso = null;
      this.feetHist = []; this.rootX *= 0.95; this.rootDepth += (ROBOT_DEPTH - this.rootDepth) * 0.05; this.jump *= 0.6;
    }
  }

  static seg(mesh, p0, p1, r) {
    const d = sub(p1, p0), len = Math.hypot(...d) || 1e-4;
    mesh.position.set((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2);
    mesh.scale.set(2 * r, len, 2 * r);
    mesh.quaternion.setFromUnitVectors(UP, new THREE.Vector3(d[0] / len, d[1] / len, d[2] / len));
  }
  static box(mesh, c, half, yaw = 0, pitch = 0, roll = 0) {
    mesh.position.set(...c); mesh.scale.set(2 * half[0], 2 * half[1], 2 * half[2]); mesh.rotation.set(pitch, yaw, roll, 'YXZ');
  }

  build(eye, t, mode, face, together, headOff, speak) {
    const { grey } = this.mat;
    this.mat.chest.emissive.set(mode === 'heart' ? 0xff508c : 0xf08c32);
    const sit = this.sit, legs = {};
    for (const s of [-1, 1]) {   // sitting: thighs forward, shins down (the right foot taps the beat)
      const tap = s === 1 ? this.footTap : 0;
      const sitLeg = [norm([s * 0.12, -0.05, 1]), norm([0, -1, 0.05 + 0.45 * tap]), norm([0, -0.3 + 0.5 * tap, 1])];
      legs[s] = sit > 0.001 ? this.leg[s].map((d, i) => lerpDir(d, sitLeg[i], sit)) : this.leg[s];
    }
    let ext = 0.15;
    for (const s of [-1, 1]) ext = Math.max(ext, -(THIGH * legs[s][0][1] + SHIN * legs[s][1][1]));
    const root = [this.rootX, GROUND + FOOT_H + ext + this.jump, -this.rootDepth];
    const dh = sub(eye, add(root, [0, 1.9, 0]));
    const lookYaw = Math.atan2(dh[0], dh[2]);
    const lookPitch = clamp(-Math.atan2(dh[1], Math.hypot(dh[0], dh[2])), -0.7, 0.7);
    const torsoYaw = (this.tracked ? this.torsoYaw : lookYaw * 0.45) * (1 - sit), hipYaw = (this.tracked ? this.hipYaw : torsoYaw) * (1 - sit);
    const bp = this.bend[0] + (0.15 - this.bend[0]) * sit, br = this.bend[1] * (1 - sit);
    const R = (v) => add(root, rot(v, torsoYaw, bp, br));

    for (const s of [-1, 1]) {   // legs
      const L = this.limb[s], [thigh, shin, foot] = legs[s];
      const hip = add(root, rot([s * 0.26, -0.02, 0], hipYaw)), knee = add(hip, mul(thigh, THIGH)), ankle = add(knee, mul(shin, SHIN));
      Robot.seg(L.thigh, hip, knee, 0.13); Robot.box(L.knee, knee, [0.1, 0.1, 0.1], hipYaw);
      Robot.seg(L.shin, knee, ankle, 0.115); Robot.seg(L.foot, add(ankle, mul(foot, -0.07)), add(ankle, mul(foot, 0.26)), 0.075);
    }
    Robot.box(this.p.pelvis, add(root, [0, 0.02, 0]), [0.4, 0.12, 0.24], hipYaw);
    Robot.box(this.p.torso, R([0, 0.5, 0]), [0.55, 0.55, 0.32], torsoYaw, bp, br);
    Robot.box(this.p.chest, R([0, 0.55, 0.325]), [0.22, 0.15, 0.012], torsoYaw, bp, br);
    Robot.box(this.p.neck, R([0, 1.15, 0]), [0.15, 0.12, 0.15], torsoYaw, bp, br);

    const hands = {}, elbows = {}, shoulders = {}, arms = { '-1': this.arm[-1], '1': this.arm[1] };
    for (const s of [-1, 1]) shoulders[s] = R([s * 0.68, 0.97, 0]);
    if (sit > 0.01) {   // the guitar rests on the lap; right hand strums at the sound hole, left hand holds the neck
      const g = this.guitar; g.visible = true; g.scale.setScalar(Math.max(0.001, sit));
      g.position.set(...R([-0.12, 0.3, 0.46]));
      g.quaternion.setFromEuler(new THREE.Euler(bp, torsoYaw, br, 'YXZ')).multiply(GUITAR_TILT); g.updateMatrixWorld(true);
      const W = (x, y, z) => { const v = g.localToWorld(new THREE.Vector3(x, y, z)); return [v.x, v.y, v.z]; };
      const ikS = ik2(shoulders[-1], W(0.1, 0.09 - 0.2 * this.strum, 0.16), 0.55, 0.55, rot([-0.5, -1, -0.4], torsoYaw));
      const ikF = ik2(shoulders[1], W(0.62 + 0.09 * (this.chord % 3), 0.03, -0.03), 0.55, 0.55, rot([0.5, -1, -0.2], torsoYaw));
      arms[-1] = [lerpDir(this.arm[-1][0], ikS[0], sit), lerpDir(this.arm[-1][1], ikS[1], sit)];
      arms[1] = [lerpDir(this.arm[1][0], ikF[0], sit), lerpDir(this.arm[1][1], ikF[1], sit)];
      const H = Math.max(0.05, root[1] - 0.1 - GROUND);
      this.stump.visible = true; this.stump.scale.set(1, Math.max(0.001, H * sit), 1);
      this.stump.position.set(root[0], GROUND + (H * sit) / 2, root[2] - 0.05);
    } else { this.guitar.visible = false; this.stump.visible = false; }
    for (const s of [-1, 1]) { elbows[s] = add(shoulders[s], mul(arms[s][0], 0.55)); hands[s] = add(elbows[s], mul(arms[s][1], 0.55)); }
    if (together) { const mid = mul(add(hands[-1], hands[1]), 0.5); for (const s of [-1, 1]) hands[s] = add(mid, rot([s * 0.08, 0, 0], torsoYaw)); }
    for (const s of [-1, 1]) {
      const L = this.limb[s];
      Robot.box(L.shoulder, shoulders[s], [0.15, 0.15, 0.15], torsoYaw, bp, br);
      Robot.seg(L.upper, shoulders[s], elbows[s], 0.115); Robot.seg(L.fore, elbows[s], hands[s], 0.1);
      const F = sit > 0.5 ? null : this.fingers[s];
      L.fist.visible = !F; L.palm.visible = !!F; L.bones.forEach((m) => (m.visible = !!F));
      if (!F) { Robot.box(L.fist, hands[s], [0.11, 0.11, 0.11], torsoYaw); continue; }
      const P = F.map((o) => add(hands[s], o));
      FINGER_BONES.forEach(([a, bb], i) => Robot.seg(L.bones[i], P[a], P[bb], a <= 1 ? 0.04 : 0.03));
      const xv = norm(sub(P[17], P[5])), yv = norm(sub(P[9], P[0])), zv = norm(cross(xv, yv)), xo = cross(yv, zv);
      const cen = mul([0, 5, 9, 13, 17].reduce((acc, i) => add(acc, P[i]), [0, 0, 0]), 0.2);
      L.palm.position.set(...cen);
      L.palm.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(new THREE.Vector3(...xo), new THREE.Vector3(...yv), new THREE.Vector3(...zv)));
      L.palm.scale.set(d2(P[5], P[17]) + Math.abs(P[5][2] - P[17][2]) + 0.08, Math.hypot(...sub(P[9], P[0])), 0.07);
    }

    const headC = R([0, 1.9, 0]);   // head follows the body, turns (as far as a neck can) to look at you
    const yaw = torsoYaw + clamp(wrap(lookYaw - torsoYaw), -1.1, 1.1) + headOff[0];
    const pitch = clamp(lookPitch + 0.7 * bp, -0.9, 1.3) + headOff[1] + sit * (0.22 + 0.1 * this.bob);
    const roll = (face ? face.roll : 0) + 0.5 * br;
    Robot.box(this.head, headC, [0.5, 0.42, 0.42], yaw, pitch, roll);
    Robot.seg(this.p.antenna, add(headC, rot([0, 0.42, 0], yaw, pitch, roll)), add(headC, rot([0, 0.74, 0], yaw, pitch, roll)), 0.03);
    Robot.box(this.p.tip, add(headC, rot([0, 0.75, 0], yaw, pitch, roll)), [0.07, 0.07, 0.07], yaw, pitch, roll);
    drawFace(this.faceCanvas, this.glowCanvas, mode, face, t, speak);
    this.faceTex.needsUpdate = true; this.glowTex.needsUpdate = true;

    const k = 1 / (1 + 1.5 * this.jump);
    this.shadow.position.set(root[0], GROUND + 0.01, root[2]); this.shadow.scale.set(k, 0.55 * k, 1);
    this.root = root; this.headC = headC; this.hands = hands;
    this.headTop = add(headC, [0, 1.0, 0]); this.mouth = add(headC, rot([0, -0.2, 0.5], yaw, pitch, roll));
  }
}

// ------------------------- the robot's face -------------------------
const EYE_CYAN = '#50e6ff', HEART_PINK = '#ff508c', MOUTH = '#283037', PUPIL = '#121c2c', BROW = '#343a46', TONGUE = '#dc5a64';
function drawFace(fc, gc, mode, face, t, speak) {
  const f = fc.getContext('2d'), g = gc.getContext('2d'), W = fc.width;
  f.fillStyle = '#afb6c2'; f.fillRect(0, 0, W, fc.height); g.fillStyle = '#000'; g.fillRect(0, 0, W, gc.height);
  const X = (x) => (x + 0.5) * W, Y = (y) => (y + 0.42) * W;
  const shape = (pts, color, glow) => {
    for (const [ctx, col] of glow ? [[f, color], [g, color]] : [[f, color], [g, '#000']]) {
      ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)))); ctx.closePath(); ctx.fillStyle = col; ctx.fill();
    }
  };
  const rect = (cx, cy, w, h) => [[cx - w, cy - h], [cx + w, cy - h], [cx + w, cy + h], [cx - w, cy + h]];
  const ngon = (cx, cy, rx, ry = rx, n = 18) => Array.from({ length: n }, (_, i) => [cx + rx * Math.cos((2 * Math.PI * i) / n), cy + ry * Math.sin((2 * Math.PI * i) / n)]);
  const chevron = (cx, cy, w, h, th) => [[cx - w, cy + h], [cx, cy - h], [cx + w, cy + h], [cx + w - th, cy + h], [cx, cy - h + th * 1.5], [cx - w + th, cy + h]];
  const heart = (cx, cy, s) => Array.from({ length: 24 }, (_, i) => { const a = (2 * Math.PI * i) / 24; return [cx + (16 * Math.sin(a) ** 3 * s) / 16, cy - ((13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a)) * s) / 16]; });
  const arc = (cx, cy, r, th, sq = 0.55) => { const o = [], inn = []; for (let i = 0; i <= 12; i++) { const a = 0.25 + (2.65 * i) / 12; o.push([cx + r * Math.cos(a), cy + r * sq * Math.sin(a)]); inn.unshift([cx + (r - th) * Math.cos(a), cy + (r - th) * sq * Math.sin(a)]); } return o.concat(inn); };
  const rrect = (cx, cy, hw, hh, th) => [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => [cx + x * Math.cos(th) - y * Math.sin(th), cy + x * Math.sin(th) + y * Math.cos(th)]);
  const mouth = (smile, op) => {
    const w = 0.2 + 0.05 * Math.max(smile, 0) - 0.04 * Math.max(-smile, 0), c = 0.07 * smile, oh = 0.03 + 0.17 * op, up = [], lo = [];
    for (let i = 0; i <= 14; i++) { const x = -w + (2 * w * i) / 14, u = x / w, sq = Math.sqrt(Math.max(0, 1 - u * u)), mid = 0.2 + c * (1 - u * u) - c / 2, th = oh * Math.max(sq, 0.18); up.push([x, mid - th / 2]); lo.unshift([x, mid + th / 2]); }
    return up.concat(lo);
  };
  const ff = face || {};
  const autoBlink = Math.floor(t * 10) % 40 === 0;
  for (const [side, ex] of [[-1, -0.22], [1, 0.22]]) {
    const op = face ? ff[side < 0 ? 'openL' : 'openR'] : autoBlink ? 0 : 1;
    if (mode === 'heart') shape(heart(ex, -0.08, 0.13), HEART_PINK, true);
    else if (mode === 'happy' || (mode === 'normal' && op > 0.3 && (ff.smile || 0) > 0.7)) shape(chevron(ex, -0.08, 0.12, 0.07, 0.045), EYE_CYAN, true);
    else if (mode === 'yawn') shape(arc(ex, -0.12, 0.1, 0.03, 0.5), EYE_CYAN, true);
    else if (mode === 'surprised') { shape(ngon(ex, -0.08, 0.14), EYE_CYAN, true); shape(ngon(ex, -0.08, 0.06), PUPIL, false); }
    else if (mode === 'wink' && side > 0) shape(rect(ex, -0.06, 0.11, 0.02), EYE_CYAN, true);
    else {
      const hh = Math.max(0.018, 0.11 * op); shape(rect(ex, -0.08, 0.11, hh), EYE_CYAN, true);
      if (op > 0.45) { const [gx, gy] = ff.gaze || [0, 0], lim = Math.max(0, hh - 0.05); shape(rect(ex + gx * 0.055, -0.08 + clamp(gy * 0.05, -lim, lim), 0.045, 0.045), PUPIL, false); }
    }
  }
  for (const [side, ex] of [[-1, -0.22], [1, 0.22]]) {
    let r = ff[side < 0 ? 'raiseL' : 'raiseR'] || 0; if (mode === 'surprised') r = 1;
    shape(rrect(ex, -0.27 - 0.1 * r, 0.13, 0.025, 0.35 * (ff.furrow || 0) * (side < 0 ? 1 : -1)), BROW, false);
  }
  if (speak != null) { shape(mouth(0.5, speak), MOUTH, false); if (speak > 0.4) shape(ngon(0, 0.2 + 0.06 * speak, 0.07, 0.035 * speak), TONGUE, false); }
  else if (mode === 'yawn') { shape(ngon(0, 0.2, 0.12, 0.16), MOUTH, false); shape(ngon(0, 0.29, 0.07, 0.05), TONGUE, false); }
  else if (mode === 'pucker' || ff.pucker) shape(ngon(0, 0.2, 0.05, 0.05, 10), MOUTH, false);
  else if (mode === 'surprised') shape(ngon(0, 0.2, 0.08, 0.08, 10), MOUTH, false);
  else if (mode === 'happy' || mode === 'heart' || mode === 'wink') shape(arc(0, 0.1, 0.24, 0.06), MOUTH, false);
  else if (face) { const op = ff.mouthOpen || 0; shape(mouth(ff.smile || 0, op), MOUTH, false); if (op > 0.4) shape(ngon(0, 0.2 + 0.06 * op, 0.07, 0.035 * op), TONGUE, false); }
  else shape(rect(0, 0.2, 0.25, 0.035), MOUTH, false);
}

// ============================== effects ==============================
function shapeTex(drawFn) { return canvasTex(64, 64, (g) => { g.fillStyle = '#fff'; g.beginPath(); drawFn(g); g.fill(); }); }
const PTEX = {};
function particleTextures() {
  PTEX.heart = shapeTex((g) => { for (let i = 0; i < 40; i++) { const a = (2 * Math.PI * i) / 40, x = 32 + 1.75 * 16 * Math.sin(a) ** 3, y = 30 - 1.75 * (13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a)); i ? g.lineTo(x, y) : g.moveTo(x, y); } });
  const star = (n, inner) => (g) => { for (let i = 0; i < 2 * n; i++) { const r = i % 2 ? 30 * inner : 30, a = -Math.PI / 2 + (Math.PI * i) / n; i ? g.lineTo(32 + r * Math.cos(a), 32 + r * Math.sin(a)) : g.moveTo(32 + r * Math.cos(a), 32 + r * Math.sin(a)); } };
  PTEX.note = shapeTex((g) => { g.ellipse(22, 46, 13, 10, -0.4, 0, 2 * Math.PI); g.rect(31, 8, 6, 40); g.moveTo(37, 8); g.quadraticCurveTo(58, 18, 50, 34); g.quadraticCurveTo(52, 22, 37, 20); });
  PTEX.star = shapeTex(star(5, 0.45)); PTEX.sparkle = shapeTex(star(4, 0.25)); PTEX.confetti = shapeTex((g) => g.rect(8, 22, 48, 20));
}

class FX {
  constructor(scene) { this.scene = scene; this.parts = []; this.bubble = document.getElementById('bubble'); this.bubbleUntil = 0; particleTextures(); }
  say(text, t, dur = 1.8) {
    const b = this.bubble; b.textContent = text; b.style.display = 'block'; b.classList.remove('pop'); void b.offsetWidth; b.classList.add('pop'); this.bubbleUntil = t + dur;
  }
  burst(kind, pos, n, speed, life, size, colors, up = 0, spin = 0) {
    for (let i = 0; i < n; i++) {
      const d = norm([Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1]);
      this.add(kind, pos, add(mul(d, speed * (0.5 + Math.random() * 0.5)), [0, up, 0]), life * (0.7 + 0.3 * Math.random()),
        size * (0.7 + 0.5 * Math.random()), colors[Math.floor(Math.random() * colors.length)], (Math.random() * 2 - 1) * spin);
    }
  }
  add(kind, pos, vel, life, size, color, spin = 0) {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: PTEX[kind], color, transparent: true, depthWrite: false }));
    sp.position.set(...pos); sp.scale.set(size * 2, size * 2, 1); this.scene.add(sp);
    this.parts.push({ sp, kind, vel, life, max: life, spin });
  }
  sparkles(p) { this.burst('sparkle', p, 16, 2.6, 0.8, 0.09, ['#fff096', '#ffffff', '#ffd25a']); }
  stars(p) { this.burst('star', p, 12, 1.6, 1.5, 0.12, ['#ffdc3c', '#fff58c'], 0.8, 3); }
  hearts(p) { this.burst('heart', p, 10, 0.6, 2.2, 0.14, [HEART_PINK, '#ff78aa', '#e6325a'], 1.0); }
  confetti(p) { this.burst('confetti', p, 80, 4.0, 2.6, 0.07, ['#ff5a5a', '#ffc83c', '#5ac8ff', '#82e678', '#c878ff'], 2.5, 8); }
  kiss(p) { this.add('heart', p, [0, 0.25, 3.2], 1.5, 0.16, HEART_PINK); }
  note(p) { const c = ['#ffffff', '#ffe066', '#ff8fc0', '#9ae6ff'][Math.floor(Math.random() * 4)]; this.add('note', p, [(Math.random() - 0.5) * 0.6, 0.7, 0.2], 1.8, 0.14, c, 0.8); }
  update(dt, t, camera) {
    this.parts = this.parts.filter((q) => {
      q.life -= dt;
      if (q.life <= 0 || q.sp.position.z > S.EYE_DIST - 2) { this.scene.remove(q.sp); q.sp.material.dispose(); return false; }
      const g = { confetti: -4, sparkle: -0.5, star: -0.8, heart: 0.2, note: 0.1 }[q.kind];
      const drag = q.vel[2] > 1 ? 0 : 1.2;
      q.vel = [q.vel[0] * (1 - drag * dt), q.vel[1] + g * dt, q.vel[2] * (1 - drag * dt)];
      q.sp.position.x += q.vel[0] * dt; q.sp.position.y += q.vel[1] * dt; q.sp.position.z += q.vel[2] * dt;
      q.sp.material.rotation += q.spin * dt; q.sp.material.opacity = Math.min(1, q.life / 0.4);
      if (q.kind === 'heart') q.sp.position.x += 0.01 * Math.sin(3 * t + q.max * 7);
      return true;
    });
    if (this.bubble.style.display === 'block' && t > this.bubbleUntil) this.bubble.style.display = 'none';
  }
  placeBubble(worldPos, camera) {
    if (this.bubble.style.display !== 'block') return;
    const v = new THREE.Vector3(...worldPos).project(camera);
    this.bubble.style.left = `${((v.x + 1) / 2) * innerWidth}px`; this.bubble.style.top = `${((1 - v.y) / 2) * innerHeight}px`;
  }
}

// ============================== sound ==============================
// Where each note of the song starts, so the robot strums in time.
function findOnsets(d, sr) {
  const blk = Math.floor(sr * 0.02), env = [];
  for (let i = 0; i + blk <= d.length; i += blk) { let s = 0; for (let j = i; j < i + blk; j++) s += d[j] * d[j]; env.push(Math.sqrt(s / blk)); }
  const mx = Math.max(...env), out = []; let last = -99;
  for (let i = 6; i < env.length; i++) {
    let avg = 0; for (let k = i - 6; k < i; k++) avg += env[k]; avg /= 6;
    if (env[i] > 1.4 * avg && env[i] > 0.15 * mx && i - last > 8) { out.push(i * 0.02); last = i; }
  }
  return out;
}

function silentWavUrl() {   // a tiny silent sound, used to switch iPhones into "media" mode
  const n = 800, b = new ArrayBuffer(44 + n * 2), v = new DataView(b), w = (o, t) => [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8, 'WAVEfmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 8000, true); v.setUint32(28, 16000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, n * 2, true);
  return URL.createObjectURL(new Blob([b], { type: 'audio/wav' }));
}

class Audio {
  constructor() { this.ctx = null; this.voices = {}; this.talk = null; }
  // Must run straight away inside the tap (phones only allow sound to start from a tap).
  unlock() {
    try { if (navigator.audioSession) navigator.audioSession.type = 'playback'; } catch (e) {}   // iPhone: play even on silent
    try {
      const el = new window.Audio(silentWavUrl()); el.setAttribute('playsinline', ''); el.play().catch(() => {});
      if (window.speechSynthesis) speechSynthesis.speak(new SpeechSynthesisUtterance(''));   // lets the phone's voice speak later
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.ctx.resume();
      const src = this.ctx.createBufferSource(); src.buffer = this.ctx.createBuffer(1, 1, 22050); src.connect(this.ctx.destination); src.start(0);
    } catch (e) { this.ctx = null; }
    const wake = () => { if (this.ctx && this.ctx.state !== 'running') this.ctx.resume(); };
    addEventListener('pointerdown', wake); document.addEventListener('visibilitychange', wake);
  }
  async init() {
    if (!this.ctx) return;
    this.wave = this.ctx.createPeriodicWave(new Float32Array([0, 0, 0]), new Float32Array([0, 1, 0.25]));
    let list = null;   // which sound files this build has (written by the Mac build script)
    try { const r = await fetch('sounds.json'); if (r.ok) list = await r.json(); } catch (e) {}
    if (!list) list = Object.fromEntries(VOICE_LINES.map(([k]) => [k, `voices/${k}.wav`]));
    this.files = Object.values(list);
    for (const [key, path] of Object.entries(list)) {
      try {
        const r = await fetch(path); if (!r.ok) continue;
        const buf = await this.ctx.decodeAudioData(await r.arrayBuffer());
        const d = buf.getChannelData(0), blk = Math.max(1, Math.floor(buf.sampleRate * 0.03)), env = [];
        for (let i = 0; i < d.length; i += blk) { let s = 0; const e = Math.min(d.length, i + blk); for (let j = i; j < e; j++) s += d[j] * d[j]; env.push(Math.sqrt(s / (e - i))); }
        const mx = Math.max(...env) + 1e-9; this.voices[key] = { buf, env: env.map((v) => Math.sqrt(v / mx)) };
        if (key === 'song') this.voices.song.onsets = findOnsets(d, buf.sampleRate);
      } catch (e) { console.warn('sound', key, e); }
    }
  }
  playSong() {
    const v = this.voices.song; if (!this.ctx || !v) return false;
    this.stopSong(); if (this.ctx.state !== 'running') this.ctx.resume();
    const src = this.ctx.createBufferSource(); src.buffer = v.buf; src.connect(this.ctx.destination); src.start();
    this.song = { src, t0: this.ctx.currentTime, dur: v.buf.duration, onsets: v.onsets || [] };
    src.onended = () => { if (this.song && this.song.src === src) this.song = null; };
    return true;
  }
  stopSong() { if (this.song) { const s = this.song; this.song = null; try { s.src.stop(); } catch (e) {} } }
  songTime() { return this.song ? Math.min(this.ctx.currentTime - this.song.t0, this.song.dur) : null; }
  speakText(text) {   // the phone's own voice, used when a line wasn't recorded
    try {
      if (!window.speechSynthesis) return false;
      const u = new SpeechSynthesisUtterance(text); u.rate = 0.95; u.pitch = 1.35;
      u.onstart = () => (this.synthTalking = true); u.onend = u.onerror = () => (this.synthTalking = false);
      speechSynthesis.cancel(); speechSynthesis.speak(u); this.synthTalking = true; this.synthUntil = performance.now() + 2500; return true;
    } catch (e) { return false; }
  }
  note(f0, dur, t0, f1 = f0, noise = false) {
    const c = this.ctx; if (c.state !== 'running') c.resume();
    const g = c.createGain(); g.connect(c.destination);
    g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(S.SOUND_VOLUME, t0 + 0.005); g.gain.setValueAtTime(S.SOUND_VOLUME, t0 + Math.max(0.006, dur - 0.03)); g.gain.linearRampToValueAtTime(0, t0 + dur);
    if (noise) {
      const n = Math.floor(c.sampleRate * dur), b = c.createBuffer(1, n, c.sampleRate), d = b.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.exp((-8 * i) / n);
      const s = c.createBufferSource(); s.buffer = b; s.connect(g); s.start(t0); return;
    }
    const o = c.createOscillator(); o.setPeriodicWave(this.wave); o.frequency.setValueAtTime(f0, t0); if (f1 !== f0) o.frequency.linearRampToValueAtTime(f1, t0 + dur);
    o.connect(g); o.start(t0); o.stop(t0 + dur + 0.02);
  }
  beep(name) {
    if (!this.ctx) return;
    const seqs = { clap: [['n', 0.035]], spark: [[1568, 0.05], [2093, 0.05], [2637, 0.08]], yay: [[523, 0.08], [659, 0.08], [784, 0.08], [1047, 0.18]],
      hi: [[880, 0.09], [660, 0.14]], yes: [[660, 0.08], [990, 0.14]], no: [[440, 0.1], [330, 0.18]], love: [[784, 0.1], [988, 0.1], [1175, 0.22]],
      kiss: [['s', 1400, 500, 0.22]], yawn: [['s', 520, 220, 0.7]], alarm: [[880, 0.1], [660, 0.1], [880, 0.1], [660, 0.1]], boing: [['s', 300, 760, 0.16]] };
    let t0 = this.ctx.currentTime + 0.01;
    for (const it of seqs[name] || []) {
      if (it[0] === 'n') { this.note(0, it[1], t0, 0, true); t0 += it[1] + 0.015; }
      else if (it[0] === 's') { this.note(it[1], it[3], t0, it[2]); t0 += it[3] + 0.015; }
      else { this.note(it[0], it[1], t0); t0 += it[1] + 0.015; }
    }
  }
  say(key) {
    const v = this.voices[key]; if (!this.ctx || !v) return false;
    if (this.ctx.state !== 'running') this.ctx.resume();
    if (this.talk) try { this.talk.src.stop(); } catch (e) {}
    const src = this.ctx.createBufferSource(); src.buffer = v.buf; src.connect(this.ctx.destination); src.start();
    this.talk = { src, env: v.env, t0: this.ctx.currentTime }; return true;
  }
  mouth() {
    if (this.synthTalking && performance.now() < this.synthUntil) return 0.35 + 0.35 * Math.sin(performance.now() / 70);
    if (!this.talk) return null;
    const i = Math.floor((this.ctx.currentTime - this.talk.t0) / 0.03);
    if (i >= this.talk.env.length) { this.talk = null; return null; }
    return Math.min(1, this.talk.env[Math.max(0, i)] * 1.1);
  }
}

// ========================= reading your body =========================
// Everything is mirrored, like looking in a mirror: x is flipped, and 3D directions are
// converted to the robot's world (y up, z toward you), which is just (-x, -y, -z).
const SET_A = [11, 13, 15, 23, 25, 27, 29, 31], SET_B = [12, 14, 16, 24, 26, 28, 30, 32];
const EYES = [[33, 133, 159, 145], [362, 263, 386, 374]];

function dir3(a, b, ref) {   // 3D limb direction from its 2D image (shorter than real = pointing at the camera)
  const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy);
  return L >= ref ? norm([dx, -dy, 0]) : norm([dx, -dy, Math.sqrt(ref * ref - L * L)]);
}

function handGesture(P) {
  const d = (a, b) => d2(P[a], P[b]), size = d(0, 9) + 1e-6;
  const fingers = [[8, 6], [12, 10], [16, 14], [20, 18]];
  const ext = fingers.map(([tp, pp]) => d(0, tp) > d(0, pp) * 1.15), fold = fingers.map(([tp, pp]) => d(0, tp) < d(0, pp));
  const thumbUp = P[4][1] < P[3][1] && P[3][1] < P[2][1] && P[2][1] - P[4][1] > 0.45 * size && P[4][1] < Math.min(...P.slice(5).map((p) => p[1]));
  if (fold.every(Boolean) && thumbUp) return 'thumbs_up';
  if (ext[0] && fold[1] && fold[2] && ext[3]) return 'love';
  if (ext[0] && ext[1] && fold[2] && fold[3]) return 'peace';
  if (ext.every(Boolean)) return 'open';
  if (fold.every(Boolean)) return 'fist';
  return null;
}

function faceMetrics(F, w, h, blend) {
  const P = (i) => [(1 - F[i].x) * w, F[i].y * h];
  const eyes = EYES.map(([c1, c2, up, lo]) => { const a = P(c1), b = P(c2), ww = d2(a, b) + 1e-6; return { c: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], w: ww, ear: d2(P(up), P(lo)) / ww }; }).sort((a, b) => a.c[0] - b.c[0]);
  const [L, R] = eyes, fw = d2(P(234), P(454)) + 1e-6, fh = d2(P(10), P(152)) + 1e-6;
  const m = { earL: L.ear, earR: R.ear, roll: -Math.atan2(R.c[1] - L.c[1], R.c[0] - L.c[0]),
    nod: (P(1)[1] - (L.c[1] + R.c[1]) / 2) / fh, turn: (P(1)[0] - (P(234)[0] + P(454)[0]) / 2) / fw, gaze: null, bs: blend || {} };
  if (F.length >= 478) {
    const ir = [P(468), P(473)], g = [L, R].map((e) => { const q = d2(ir[0], e.c) < d2(ir[1], e.c) ? ir[0] : ir[1]; return [(q[0] - e.c[0]) / e.w, (q[1] - e.c[1]) / e.w]; });
    m.gaze = [(g[0][0] + g[1][0]) / 2, (g[0][1] + g[1][1]) / 2];
  }
  return m;
}

function analyze(res, w, h, state) {
  const body = { head: null, arms: {}, wristRatio: null, bothUp: false, oneUp: false, face: null, gestures: new Set(), hands: {},
    legs: {}, torsoYaw: null, hipYaw: null, bend: null, full: false };
  const px = (l) => [(1 - l.x) * w, l.y * h];
  let midX = w / 2; const wrists = {};
  const F = res.faceLandmarks && res.faceLandmarks[0];
  if (F && F.length > 454) {
    body.head = [(1 - F[168].x - 0.5) * 2, (F[168].y - 0.5) * 2];
    const blend = {}; const cat = res.faceBlendshapes && res.faceBlendshapes[0];
    if (cat && cat.categories) for (const c of cat.categories) blend[c.categoryName] = c.score;
    body.face = faceMetrics(F, w, h, blend);
  }
  const L = res.poseLandmarks && res.poseLandmarks[0], WL = res.poseWorldLandmarks && res.poseWorldLandmarks[0];
  if (L) {
    // Some versions report visibility 0 for every point; then judge by "is it inside the camera picture".
    const noVis = L.every((l) => !l.visibility);
    const vis = (i) => (noVis ? L[i].x > 0.01 && L[i].x < 0.99 && L[i].y > 0.01 && L[i].y < 0.99 : L[i].visibility > 0.5);
    const W3 = (i) => [-WL[i].x, -WL[i].y, -WL[i].z];
    if (!body.head && vis(0)) body.head = [(1 - L[0].x - 0.5) * 2, (L[0].y - 0.5) * 2];
    if (vis(11) && vis(12)) {
      const s1 = px(L[11]), s2 = px(L[12]), sw = d2(s1, s2) + 1e-6, noseY = px(L[0])[1]; midX = (s1[0] + s2[0]) / 2;
      let facing = true;
      if (WL) { const v3 = sub(W3(11), W3(12)); facing = Math.abs(v3[0]) > 0.8 * Math.hypot(...v3); }
      // which side of your body is the robot's right? learned while you face the camera
      if (state.sideA == null || (body.face && facing)) state.sideA = s1[0] > s2[0] ? 1 : -1;
      const sides = { [state.sideA]: SET_A, [-state.sideA]: SET_B };
      for (const side of [-1, 1]) {
        const [si, ei, wi] = sides[side]; if (!vis(ei)) continue;
        const Sp = px(L[si]), Ep = px(L[ei]);
        const upper = WL ? norm(sub(W3(ei), W3(si))) : dir3(Sp, Ep, 0.85 * sw); let fore = upper;
        if (vis(wi)) { const Wp = px(L[wi]); fore = WL ? norm(sub(W3(wi), W3(ei))) : dir3(Ep, Wp, 0.75 * sw); wrists[side] = [Wp, Sp]; }
        body.arms[side] = [upper, fore];
      }
      if (wrists[1] && wrists[-1]) { body.wristRatio = d2(wrists[1][0], wrists[-1][0]) / sw; body.bothUp = wrists[1][0][1] < noseY && wrists[-1][0][1] < noseY; }
      body.oneUp = Object.values(wrists).filter(([Wp, Sp]) => Wp[1] < Sp[1] - 0.35 * sw).length === 1;
      if (WL) {
        const sv = sub(W3(sides[1][0]), W3(sides[-1][0])); body.torsoYaw = Math.atan2(-sv[2], sv[0]);
        if (vis(23) && vis(24)) {
          const hv = sub(W3(sides[1][3]), W3(sides[-1][3])); body.hipYaw = Math.atan2(-hv[2], hv[0]);
          const up = norm(sub(mul(add(W3(11), W3(12)), 0.5), mul(add(W3(23), W3(24)), 0.5))), ul = rot(up, -body.torsoYaw);
          body.bend = [clamp(Math.atan2(ul[2], ul[1]), -0.5, 1.4), clamp(Math.atan2(-ul[0], ul[1]), -0.6, 0.6)];
          for (const side of [-1, 1]) {
            const [, , , hi, ki, ai, he, ti] = sides[side];
            if (vis(ki) && vis(ai)) body.legs[side] = [norm(sub(W3(ki), W3(hi))), norm(sub(W3(ai), W3(ki))), vis(ti) && vis(he) ? norm(sub(W3(ti), W3(he))) : null];
          }
          if (body.legs[1] && body.legs[-1]) {   // whole body in view
            const H1 = px(L[23]), H2 = px(L[24]), A1 = px(L[27]), A2 = px(L[28]), hc = [(H1[0] + H2[0]) / 2, (H1[1] + H2[1]) / 2], sc = [(s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2];
            Object.assign(body, { full: true, posX: (hc[0] / w - 0.5) * 2, feetY: Math.max(A1[1], A2[1]) / h,
              legLen: (d2(H1, A1) + d2(H2, A2)) / 2 / h, torsoLen: d2(hc, sc) / h });
          }
        }
      }
    }
  }
  for (const hand of [res.leftHandLandmarks && res.leftHandLandmarks[0], res.rightHandLandmarks && res.rightHandLandmarks[0]]) {
    if (!hand || hand.length < 21) continue;
    const g = handGesture(hand.map(px)); if (g) body.gestures.add(g);
    const pts = hand.map((l) => [(1 - l.x) * w, l.y * h, l.z * w]);
    const ws = Object.keys(wrists).map(Number);
    const side = ws.length ? ws.reduce((best, s) => (d2(wrists[s][0], pts[0]) < d2(wrists[best][0], pts[0]) ? s : best), ws[0]) : pts[0][0] > midX ? 1 : -1;
    body.hands[side] = pts;
  }
  return body;
}

function countSwings(vals, thr) {
  if (!vals.length) return 0; let n = 0, dir = 0, ext = vals[0];
  for (const v of vals.slice(1)) {
    if (dir === 0) { if (Math.abs(v - ext) > thr) { dir = v > ext ? 1 : -1; ext = v; n++; } }
    else if (dir === 1) { if (v > ext) ext = v; else if (ext - v > thr) { dir = -1; ext = v; n++; } }
    else { if (v < ext) ext = v; else if (v - ext > thr) { dir = 1; ext = v; n++; } }
  }
  return n;
}

// ============================== reactions ==============================
class Reactor {
  constructor(fx, audio) {
    this.fx = fx; this.audio = audio; this.hold = {}; this.nextOk = {}; this.apart = true; this.claps = [];
    this.timed = null; this.timedUntil = 0; this.cheer = false; this.together = false; this.base = {}; this.frames = 0; this.face = null;
    this.closedSince = null; this.nextWake = 0; this.lastFace = -99; this.nextGreet = 0; this.nods = []; this.turns = []; this.nextHead = 0;
    this.anim = null; this.animT0 = 0; this.active = []; this.airborne = false;
    this.COOLDOWN = { yay: 2.5, thumbs: 3, peace: 3, yawn: 6, love: 6, kiss: 6 }; this.HOLDS = { yawn: 0.8, love: 0.3, kiss: 0.45 };
  }
  held(name, cond, t) {
    if (!cond) { delete this.hold[name]; return false; }
    if (this.hold[name] == null) this.hold[name] = t;
    if (t - this.hold[name] >= (this.HOLDS[name] || S.HOLD_TIME) && t >= (this.nextOk[name] || 0)) { this.hold[name] = Infinity; this.nextOk[name] = t + this.COOLDOWN[name]; return true; }
    return false;
  }
  timedExpr(name, t, dur) { this.timed = name; this.timedUntil = t + dur; }
  line(key, text, t, beep) { if (!this.audio.say(key)) this.audio.beep(beep); this.fx.say(text, t); }
  greet(t, robot) { if (t < this.nextGreet) return; this.nextGreet = t + 12; this.line('hey_beautiful', 'HEY BEAUTIFUL!', t, 'hi'); this.fx.sparkles(add(robot.headC, [0, 0.6, 0.5])); this.timedExpr('happy', t, 2); }
  love(t, robot, blown = false) {
    this.nextOk.love = this.nextOk.kiss = t + this.COOLDOWN.love; this.line('i_love_you', 'I LOVE YOU!', t, 'love');
    if (blown) { this.audio.beep('kiss'); this.fx.kiss(robot.mouth); }
    this.fx.hearts(add(robot.headC, [0, 0.3, 0.6])); this.timedExpr('heart', t, 3);
  }
  faceState(m) {
    this.frames++; const r = this.frames < 45 ? 0.12 : 0.004, B = this.base;
    const base = (k, v, upd = true) => { if (B[k] == null) B[k] = v; else if (upd) B[k] += (v - B[k]) * r; return B[k]; };
    const opens = ['earL', 'earR'].map((k) => { const v = m[k], b = base(k, v, v > 0.7 * (B[k] == null ? v : B[k])); return clamp((v / b - S.BLINK_CLOSED) / (0.9 - S.BLINK_CLOSED), 0, 1); });
    const bs = (n) => m.bs[n] || 0, avg = (a, b) => (bs(a) + bs(b)) / 2;
    const up = 0.5 * bs('browInnerUp') + 0.5 * avg('browOuterUpLeft', 'browOuterUpRight'), down = avg('browDownLeft', 'browDownRight');
    const bUp = base('up', up), bDown = base('down', down), bSm = base('sm', avg('mouthSmileLeft', 'mouthSmileRight')), bFr = base('fr', avg('mouthFrownLeft', 'mouthFrownRight'));
    const raise = clamp((up - bUp) * 3 - Math.max(0, down - bDown) * 1.5, -1, 1);
    let gaze = [0, 0];
    if (m.gaze) gaze = [clamp((m.gaze[0] - base('gx', m.gaze[0])) * S.GAZE_GAIN, -1, 1), clamp((m.gaze[1] - base('gy', m.gaze[1])) * S.GAZE_GAIN * 1.5, -1, 1)];
    return { openL: opens[0], openR: opens[1], raiseL: raise, raiseR: raise, furrow: clamp((down - bDown) * 2.5, 0, 1),
      smile: clamp((avg('mouthSmileLeft', 'mouthSmileRight') - bSm) * 2.2 - (avg('mouthFrownLeft', 'mouthFrownRight') - bFr) * 2.5, -1, 1),
      mouthOpen: clamp((bs('jawOpen') - 0.05) / 0.5, 0, 1), roll: clamp(m.roll, -0.5, 0.5), gaze,
      pucker: bs('mouthPucker') > S.PUCKER && bs('jawOpen') < 0.25 };
  }
  smoothFace(nw) {
    if (!this.face) { this.face = nw; return; }
    const o = this.face, out = {};
    for (const k in nw) {
      if (k === 'pucker') out[k] = nw[k];
      else if (k === 'gaze') out[k] = [o.gaze[0] + (nw.gaze[0] - o.gaze[0]) * S.FACE_SMOOTH, o.gaze[1] + (nw.gaze[1] - o.gaze[1]) * S.FACE_SMOOTH];
      else out[k] = o[k] + (nw[k] - o[k]) * (k.startsWith('open') ? 0.9 : S.FACE_SMOOTH);
    }
    this.face = out;
  }
  update(body, t, robot) {
    this.active = [];
    if (!body) { this.cheer = this.together = false; return; }
    const m = body.face, handsMid = mul(add(robot.hands[-1], robot.hands[1]), 0.5);
    let shut = false;
    if (m) {
      if (t - this.lastFace > 4) this.greet(t, robot);
      this.lastFace = t; this.smoothFace(this.faceState(m)); const f = this.face;
      shut = f.openL < 0.25 && f.openR < 0.25;
      if (shut) {
        this.closedSince = this.closedSince || t;
        if (t - this.closedSince > S.SLEEP_TIME && t >= this.nextWake) { this.nextWake = t + 5; this.line('wake_up', 'WAKE UP!', t, 'alarm'); this.timedExpr('surprised', t, 1.5); this.anim = 'shake'; this.animT0 = t; }
      } else this.closedSince = null;
      if (this.held('yawn', (m.bs.jawOpen || 0) > S.YAWN_JAW, t)) { this.audio.beep('yawn'); this.fx.say('*YAAAWN*', t); this.timedExpr('yawn', t, 2.2); }
      if (this.held('kiss', f.pucker, t)) this.love(t, robot, true);
      this.nods = this.nods.filter(([tt]) => t - tt < 1.5); this.nods.push([t, m.nod]);
      this.turns = this.turns.filter(([tt]) => t - tt < 1.5); this.turns.push([t, m.turn]);
      const nN = countSwings(this.nods.map((x) => x[1]), S.NOD_THR), nT = countSwings(this.turns.map((x) => x[1]), S.SHAKE_THR);
      if (t >= this.nextHead && !shut) {
        if (nN >= 3 && nT <= 1) { this.audio.beep('yes'); this.fx.say('YES!', t); this.anim = 'nod'; this.animT0 = t; this.nextHead = t + 2; this.nods = []; }
        else if (nT >= 3 && nN <= 1) { this.audio.beep('no'); this.fx.say('NOPE!', t); this.anim = 'shake'; this.animT0 = t; this.nextHead = t + 2; this.turns = []; }
      }
    }
    const wr = body.wristRatio; this.together = wr != null && wr < S.CLAP_CLOSE;
    if (wr != null) {
      if (this.apart && wr < S.CLAP_CLOSE) {
        this.apart = false; this.claps = this.claps.filter((c) => t - c < 2.5); this.claps.push(t); this.audio.beep('clap'); this.fx.sparkles(handsMid);
        if (this.claps.length >= 3) { this.audio.beep('yay'); this.fx.say('BRAVO!', t); this.fx.confetti(add(robot.headC, [0, 1, 0])); this.timedExpr('happy', t, 2); this.claps = []; }
        else this.fx.say('CLAP!', t);
      } else if (wr > S.CLAP_CLOSE * 1.7) this.apart = true;
    }
    this.cheer = body.bothUp; const g = body.gestures;
    if (this.held('yay', body.bothUp, t)) { this.audio.beep('yay'); this.fx.confetti(add(robot.headC, [0, 1.2, 0])); this.fx.say('YAY!', t); }
    if (body.oneUp && !body.bothUp && !g.has('love')) this.greet(t, robot);
    if (this.held('love', g.has('love'), t)) this.love(t, robot);
    if (this.held('thumbs', g.has('thumbs_up'), t)) { this.audio.beep('yes'); this.fx.hearts(add(robot.headC, [0, 0.3, 0.6])); this.fx.say('AWW, THANKS!', t); this.timedExpr('heart', t, 2.5); }
    if (this.held('peace', g.has('peace'), t)) { this.audio.beep('spark'); this.fx.stars(add(robot.headC, [0, 0.2, 0.6])); this.fx.say('PEACE!', t); this.timedExpr('wink', t, 2); }
    if (robot.jump > 0.12 && !this.airborne) { this.airborne = true; this.audio.beep('boing'); } else if (robot.jump < 0.04) this.airborne = false;
    this.active = [...g];
    if (m && this.face) { const f = this.face; [['blink L', f.openL < 0.3], ['blink R', f.openR < 0.3], ['smile', f.smile > 0.4], ['frown', f.smile < -0.4], ['brows up', f.raiseL > 0.4], ['furrow', f.furrow > 0.4], ['mouth open', f.mouthOpen > 0.3], ['pucker', f.pucker]].forEach(([n, on]) => on && this.active.push(n)); }
    [['hands up', body.bothUp], ['hand up', body.oneUp], ['hands together', this.together], ['full body', body.full], ['jump!', this.airborne]].forEach(([n, on]) => on && this.active.push(n));
  }
  mode(t) { return t < this.timedUntil ? this.timed : this.cheer ? 'happy' : 'normal'; }
  headOffsets(t) {
    if (this.timed === 'yawn' && t < this.timedUntil) return [0, -0.25];
    if (this.anim) { const age = t - this.animT0; if (age > 1.2) this.anim = null; else { const w = 0.3 * Math.sin(age * 2 * Math.PI * 2.2) * (1 - age / 1.2); return this.anim === 'shake' ? [w * 1.2, 0] : [0, w]; } }
    return [0, 0];
  }
}

// ============================== main ==============================
const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera();
const world = buildWorld(scene), robot = new Robot(scene), fx = new FX(scene), audio = new Audio(), reactor = new Reactor(fx, audio);
let SW = 4.67, SH = 4.67;
function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  const a = innerWidth / innerHeight;
  if (a >= 1) { SH = 4.67; SW = SH * a; } else { SW = 3.4; SH = SW / a; }   // portrait phones: keep the robot fully in view
}
addEventListener('resize', resize); resize();

// Off-axis projection: the screen is a window, your eye is at (ex, ey, EYE_DIST) in front of it.
function setEye(ex, ey) {
  const n = 0.1, f = 4000, e = S.EYE_DIST;
  camera.position.set(ex, ey, e); camera.quaternion.identity(); camera.updateMatrixWorld(true);
  camera.projectionMatrix.makePerspective((-SW / 2 - ex) * n / e, (SW / 2 - ex) * n / e, (SH / 2 - ey) * n / e, (-SH / 2 - ey) * n / e, n, f);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}

// The camera video must be ON the page (tiny and invisible) - Android Chrome won't deliver frames otherwise.
const video = document.createElement('video');
video.setAttribute('playsinline', ''); video.setAttribute('muted', ''); video.setAttribute('autoplay', '');
video.playsInline = true; video.muted = true; video.autoplay = true;
Object.assign(video.style, { position: 'fixed', left: '0', bottom: '0', width: '2px', height: '2px', opacity: '0.01', pointerEvents: 'none', zIndex: '-1' });
document.body.appendChild(video);
const diag = { results: 0, errors: 0, lastError: '', lastDetect: 0 };
let trackers = null, lastVideoTime = -1, injected = null, sx = 0, sy = 0, lx = 0, ly = 0, lastT = performance.now() / 1000;
const trackState = {};
window.__inject = (res) => { injected = res; };   // used by automated tests

function toast(msg, ms = 5000) { const el = document.getElementById('toast'); el.textContent = msg; el.style.display = 'block'; clearTimeout(toast.h); if (ms) toast.h = setTimeout(() => (el.style.display = 'none'), ms); }

async function fetchWithProgress(url, status, label) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const total = Number(r.headers.get('content-length')) || 0;
  if (!r.body || !r.body.getReader) return new Uint8Array(await r.arrayBuffer());
  const reader = r.body.getReader(), parts = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    parts.push(value); got += value.length;
    status(`Downloading the robot's brain (${label})… ${(got / 1e6).toFixed(1)}${total ? ` / ${(total / 1e6).toFixed(1)}` : ''} MB`);
  }
  const out = new Uint8Array(got); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

// One tracker (face, body or hands). Uses the phone's graphics chip first; if that fails, even
// later while running, it quietly switches to the processor instead of stopping.
class Tracker {
  constructor(Kind, opts, buf, fileset, label) { Object.assign(this, { Kind, opts, buf, fileset, label, lm: null, delegate: null, busy: false, errors: 0, lastError: '' }); }
  async make(delegate) {
    this.busy = true;
    try {
      if (this.lm) { try { this.lm.close(); } catch (e) {} this.lm = null; }
      this.lm = await this.Kind.createFromOptions(this.fileset, { ...this.opts, baseOptions: { modelAssetBuffer: this.buf, delegate }, runningMode: 'VIDEO' });
      this.delegate = delegate; return true;
    } catch (e) { this.lastError = String(e && e.message || e).slice(0, 140); console.warn(this.label, delegate, e); return false; }
    finally { this.busy = false; }
  }
  detect(video, ts) {
    if (!this.lm || this.busy) return null;
    try { const r = this.lm.detectForVideo(video, ts); this.errors = 0; return r; }
    catch (e) {
      this.errors++; this.lastError = String(e && e.message || e).slice(0, 140); console.warn(this.label, e);
      if (this.delegate === 'GPU') { this.lm = null; this.make('CPU'); }       // fall back to the processor
      return null;
    }
  }
}

async function makeTrackers(status) {
  status('Loading the tracking engine…');
  const fileset = await FilesetResolver.forVisionTasks('./wasm');
  const out = {}, kinds = { face: [FaceLandmarker, { numFaces: 1, outputFaceBlendshapes: true }],
    pose: [PoseLandmarker, { numPoses: 1 }], hand: [HandLandmarker, { numHands: 2 }] };
  let i = 0;
  for (const key of ['face', 'pose', 'hand']) {
    i++; const m = MODELS[key]; let buf = null;
    for (const url of [m.local, m.remote]) {
      try { buf = await fetchWithProgress(url, status, `${i}/3 ${m.name}`); break; } catch (e) { console.warn('model', url, e); }
    }
    if (!buf) throw new Error(`Could not download the ${m.name} model`);
    const tr = new Tracker(kinds[key][0], kinds[key][1], buf, fileset, m.name);
    status(`Waking the robot up (${m.name})…`);
    if (!(await tr.make('GPU')) && !(await tr.make('CPU'))) throw new Error(`The ${m.name} tracker would not start: ${tr.lastError}`);
    out[key] = tr;
  }
  return out;
}

// Ask all three trackers about one camera frame, and combine the answers.
let handFrame = 0, lastHands = { landmarks: [] };
function detectAll(trackers, video, ts) {
  const f = trackers.face.detect(video, ts), p = trackers.pose.detect(video, ts + 0.001);
  if (handFrame++ % 2 === 0) { const h = trackers.hand.detect(video, ts + 0.002); if (h) lastHands = h; }   // hands every other frame (saves time)
  if (!f && !p) return null;
  const hands = lastHands.landmarks || [];
  return { faceLandmarks: f ? f.faceLandmarks : [], faceBlendshapes: f ? f.faceBlendshapes : [],
    poseLandmarks: p ? p.landmarks : [], poseWorldLandmarks: p ? p.worldLandmarks : [],
    leftHandLandmarks: hands[0] ? [hands[0]] : [], rightHandLandmarks: hands[1] ? [hands[1]] : [] };
}

async function start() {
  const st = document.getElementById('status'), status = (s) => (st.textContent = s);
  document.getElementById('go').disabled = true;
  audio.unlock();                 // first thing, while the tap still "counts"
  try { document.documentElement.requestFullscreen?.().catch(() => {}); } catch (e) {}
  await Promise.all([audio.init(), loadName()]);
  status('Asking for the camera…');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    video.srcObject = stream; await video.play();
  } catch (e) { toast('No camera: allow camera access for this site, then reload. The robot will just stand around until then.', 0); }
  try { trackers = await makeTrackers(status); } catch (e) { toast(String(e.message || e) + ' - check the internet connection and reload.', 0); }
  document.getElementById('start').style.display = 'none';
  document.getElementById('dbgbtn').style.display = 'block';
  document.getElementById('guitarbtn').style.display = 'block';
  saveForOffline();
  welcome();
}

// Said once every time the app opens. The name comes from config.json (set when building) or the link: ...?name=Alex
let NAME = '';
async function loadName() {
  try { const r = await fetch('config.json'); if (r.ok) NAME = (await r.json()).name || ''; } catch (e) {}
  const q = new URLSearchParams(location.search).get('name'); if (q) NAME = q.slice(0, 24);
}
function welcome() {
  const t = performance.now() / 1000;
  reactor.nextGreet = t + 10; reactor.lastFace = t;        // don't say HEY BEAUTIFUL on top of it
  if (!audio.say('welcome') && !audio.speakText(NAME ? `Welcome, ${NAME}` : 'Welcome!')) audio.beep('love');
  fx.say(NAME ? `WELCOME ${NAME.toUpperCase()}! ♥` : 'WELCOME! ♥', t, 3);
  fx.hearts(add(robot.headC, [0, 0.3, 0.6])); reactor.timedExpr('happy', t, 3);
}

// ---------------- the guitar button: sit down, play the song, stand up again ----------------
const guitar = { on: false };
function playGuitar() {
  const t = performance.now() / 1000;
  audio.stopSong(); if (audio.talk) { try { audio.talk.src.stop(); } catch (e) {} audio.talk = null; }
  Object.assign(guitar, { on: true, pressT: t, started: false, idx: 0, count: 0, lastStrum: -9, times: [], dur: 0, synth: false });
  fx.say('♪ A SONG FOR YOU ♪', t, 2);
}
const CHORDS = [[196, 247, 294], [262, 330, 392], [294, 370, 440], [247, 294, 370]];   // G C D Bm, for the phone without the song
function guitarFrame(t, dt) {
  if (guitar.on) {
    const since = t - guitar.pressT;
    if (!guitar.started && since > 0.7) {             // start once the robot has sat down
      guitar.started = true; guitar.songT0 = t;
      guitar.synth = !audio.playSong();
      if (guitar.synth) guitar.dur = 8; else guitar.dur = audio.song.dur;
      const on = guitar.synth ? [] : audio.song.onsets;
      guitar.times = on.length >= 4 ? on.slice() : Array.from({ length: Math.floor(guitar.dur / 0.45) }, (_, i) => 0.1 + i * 0.45);
    }
    if (guitar.started) {
      const st = guitar.synth ? t - guitar.songT0 : audio.songTime() ?? guitar.dur;
      while (guitar.idx < guitar.times.length && guitar.times[guitar.idx] <= st) {   // one strum per note of the song
        guitar.idx++; guitar.count++; guitar.lastStrum = t;
        if (guitar.count % 4 === 0) robot.chord++;
        if (guitar.count % 2 === 0) { const g = robot.guitar.position; fx.note([g.x + (guitar.count % 4 ? 0.85 : -0.85), g.y + 0.3, g.z + 0.3]); }
        if (guitar.synth && audio.ctx) CHORDS[robot.chord % 4].forEach((f, i) => audio.note(f, 0.5, audio.ctx.currentTime + i * 0.025));
      }
      if (st >= guitar.dur - 0.02) {                  // song finished: stand up and say thanks
        guitar.on = false; audio.stopSong();
        fx.say('THANK YOU ♥', t, 2); fx.hearts(add(robot.headC, [0, 0.3, 0.6])); reactor.timedExpr('heart', t, 2.5);
      }
    }
  }
  robot.sit += ((guitar.on ? 1 : 0) - robot.sit) * Math.min(1, dt * 4);
  if (robot.sit < 0.002) robot.sit = 0;
  const ago = t - (guitar.lastStrum ?? -9);
  robot.strum = ago < 0.08 ? ago / 0.08 : ago < 0.33 ? 1 - (ago - 0.08) / 0.25 : 0;   // quick down-stroke, slower return
  robot.bob = Math.exp(-ago * 5); robot.footTap = Math.max(0, 1 - ago * 4);
}
document.getElementById('guitarbtn').addEventListener('click', (e) => {
  e.stopPropagation(); if (audio.ctx && audio.ctx.state !== 'running') audio.ctx.resume(); playGuitar();
});

// Ask the service worker to keep a copy of everything, so the robot also opens without internet.
function saveForOffline() {
  if (!('serviceWorker' in navigator) || !navigator.serviceWorker.controller && !navigator.serviceWorker.getRegistration) return;
  navigator.serviceWorker.ready.then((reg) => {
    navigator.serviceWorker.addEventListener('message', (e) => {
      if (e.data && e.data.offlineReady) { diag.offline = e.data.saved; toast('Saved on this phone - the robot now works offline too ✓', 4000); }
    });
    (reg.active || reg.waiting || reg.installing).postMessage({ saveAll: true, files: ['./sounds.json', ...(audio.files || []).map((f) => './' + f.replace(/^\.\//, ''))] });
  }).catch(() => {});
}
document.getElementById('go').addEventListener('click', start);
let showDebug = false;
document.getElementById('dbgbtn').addEventListener('click', () => { showDebug = !showDebug; document.getElementById('debug').style.display = showDebug ? 'block' : 'none'; });

function frame() {
  requestAnimationFrame(frame);
  const now = performance.now(), t = now / 1000, dt = Math.min(t - lastT, 0.1); lastT = t;
  let res = null;
  if (injected) { res = injected; injected = null; }
  else if (trackers && video.readyState >= 2 && video.videoWidth > 0 &&
           (video.currentTime !== lastVideoTime || now - diag.lastDetect > 120)) {
    lastVideoTime = video.currentTime; diag.lastDetect = now;
    res = detectAll(trackers, video, now);
    if (res) diag.results++;
    const bad = Object.values(trackers).find((tr) => tr.errors >= 30 && tr.delegate === 'CPU');
    if (bad && !diag.lastError) { diag.lastError = `${bad.label}: ${bad.lastError}`; toast('Tracking problem: ' + diag.lastError, 0); }
  }
  if (video.paused && video.srcObject) video.play().catch(() => {});
  let body = null;
  if (res) {
    const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
    body = analyze(res, vw, vh, trackState);
    if (body.head) {
      sx += (body.head[0] * S.HEAD_GAIN - sx) * S.SMOOTH; sy += (-body.head[1] * S.HEAD_GAIN - sy) * S.SMOOTH;
      sx = clamp(sx, -5.2, 5.2); sy = clamp(sy, -1.35, 3.7);
    }
    lx += (sx - lx) * S.LOOK_SMOOTH; ly += (sy - ly) * S.LOOK_SMOOTH;
    robot.update(body, t); if (!guitar.on) reactor.update(body, t, robot);
  }
  if (showDebug) {   // what the camera sees + what the robot detects + numbers to diagnose problems
    const c = document.getElementById('dbgcam').getContext('2d');
    c.fillStyle = '#000'; c.fillRect(0, 0, 160, 120);
    c.save(); c.scale(-1, 1); try { c.drawImage(video, -160, 0, 160, 120); } catch (e) {} c.restore();
    document.getElementById('dbgtxt').textContent =
      (reactor.active.join(', ') || '(nothing detected)') +
      `\ncamera: ${video.srcObject ? 'on' : 'OFF'}, ${video.videoWidth}x${video.videoHeight}, state ${video.readyState}${video.paused ? ', paused' : ''}` +
      `\nsound: ${audio.ctx ? audio.ctx.state : 'off'}, voices ${Object.keys(audio.voices).length}/3, offline: ${diag.offline ? 'saved' : 'not yet'}` +
      `\nbrain: ${trackers ? Object.values(trackers).map((tr) => `${tr.label} ${tr.lm ? tr.delegate : 'off'}`).join(', ') : 'NOT loaded'}, frames read ${diag.results}` +
      (trackers ? Object.values(trackers).filter((tr) => tr.lastError).map((tr) => `\n${tr.label} error: ${tr.lastError}`).join('') : '') +
      (diag.lastError ? `\nerror: ${diag.lastError}` : '');
  }
  guitarFrame(t, dt);
  fx.update(dt, t, camera); world.update(t);
  const face = body && body.face ? reactor.face : res ? null : robot._lastFace || null;
  if (res) robot._lastFace = face;
  setEye(sx, sy);
  const playing = robot.sit > 0.4, mode = playing && reactor.mode(t) === 'normal' ? 'happy' : reactor.mode(t);
  robot.build([lx, ly, S.EYE_DIST], t, mode, face, playing ? false : reactor.together, playing ? [0, 0, 0] : reactor.headOffsets(t), audio.mouth());
  fx.placeBubble(robot.headTop, camera);
  renderer.render(scene, camera);
}
setEye(0, 0); requestAnimationFrame(frame);
window.__robot = { robot, reactor, S, detectAll, Tracker, analyze, audio, guitar, playGuitar, fx };   // used by automated tests
