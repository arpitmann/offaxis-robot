import base64
import math
import os
import random
import sys
import wave

import numpy as np
import pygame
import pygame.gfxdraw
import cv2
import mediapipe as mp

# ======================= tweakables =======================
WIN_WIDTH, WIN_HEIGHT = 1000, 700
SCALE = 150            # pixels per world unit on the screen plane
EYE_DIST = 8.0         # your eye's distance from the screen (world units)
HEAD_GAIN = 5.0        # webcam motion -> world motion
SMOOTH = 0.6           # head smoothing (lower = smoother but laggier)
LOOK_SMOOTH = 0.3      # how quickly the robot turns to look at you (1.0 = instantly)
ARM_SMOOTH = 0.75      # how quickly the robot copies your arms
FINGER_SMOOTH = 0.7    # how quickly the robot copies your fingers
FACE_SMOOTH = 0.7      # how quickly the robot copies your face (eyes always react fast)
BODY_SMOOTH = 0.5      # how quickly it copies turning, bending and leaning
LEG_SMOOTH = 0.7       # how quickly it copies your legs
WALK_GAIN_X = 2.2      # how far it walks sideways when you step sideways (full body in view)
WALK_GAIN_Z = 3.0      # how far it walks forward/back when you step toward/away from the camera
JUMP_GAIN = 1.0        # how high it jumps compared with you
HOLD_TIME = 0.35       # seconds a gesture must be held before it counts
CLAP_CLOSE = 0.55      # wrists closer than this x shoulder-width = clap
BLINK_CLOSED = 0.45    # eye counts as shut below this fraction of its normal openness
SLEEP_TIME = 2.0       # seconds with both eyes shut before the robot shouts WAKE UP
YAWN_GAP = 0.28        # how wide your mouth must open (vs face height) to count as a yawn
PUCKER_DROP = 0.06     # how much narrower than normal your lips must get for a kiss
NOD_THR = 0.03         # how big each up/down head movement must be to count as a nod
SHAKE_THR = 0.05       # how big each left/right head movement must be to count as a shake
GAZE_GAIN = 6.0        # how far the robot's pupils move when your eyes move
SEED = 7               # change for a different landscape
SOUND_VOLUME = 0.35    # beeps volume (0..1); your recorded voice always plays at full volume
VOICE_LINES = [("hey_beautiful", "HEY BEAUTIFUL"), ("i_love_you", "I LOVE YOU"), ("wake_up", "WAKE UP")]
HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(HERE, "voices")
# ==========================================================

GROUND_Y = 1.8         # ground is this far below eye level (y points DOWN)
ROBOT_Z = 3.8          # robot's distance behind the screen
MX, MY = 820, 620      # scenery margins so the edges never show when you move
CX, CY = WIN_WIDTH / 2, WIN_HEIGHT / 2
HX_LIM = (MX - 40) / SCALE
HY_LIM = (-(MY - 60) / SCALE, GROUND_Y * 0.75)   # keep your eye above the ground

SKY_TOP = (58, 112, 186)
SKY_HORIZON = (196, 216, 232)
HAZE = (186, 206, 224)


# ======================= small maths =======================
def add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def mul(a, k): return (a[0] * k, a[1] * k, a[2] * k)
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    n = math.sqrt(dot(a, a))
    return (0.0, 1.0, 0.0) if n < 1e-9 else (a[0] / n, a[1] / n, a[2] / n)


def mix(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def fz(z):
    """Screen-plane scale of something at depth z."""
    return EYE_DIST / (EYE_DIST + z)


def haze_amt(z):
    return 1.0 - math.exp(-max(z, 0.0) / 600.0)


TO_LIGHT = norm((-0.45, -0.75, -0.55))   # sun is behind you, up and to the left


def project(p, hx, hy):
    """Off-axis projection onto the screen plane z=0, eye at (hx, hy, -EYE_DIST)."""
    f = EYE_DIST / (EYE_DIST + p[2])
    return (CX + (hx + (p[0] - hx) * f) * SCALE, CY + (hy + (p[1] - hy) * f) * SCALE)


def rot(v, yaw, pitch=0.0, roll=0.0):
    """Roll about z, then pitch about x, then yaw about y. A part's local front is -z."""
    x, y, z = v
    if roll:
        cr, sr = math.cos(roll), math.sin(roll)
        x, y = x * cr - y * sr, x * sr + y * cr
    cp, sp = math.cos(pitch), math.sin(pitch)
    y, z = y * cp - z * sp, y * sp + z * cp
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, z = x * cy + z * sy, -x * sy + z * cy
    return (x, y, z)


# ======================= scenery =======================
class Layer:
    """A flat slice of scenery at one depth. Because it's parallel to the screen,
    the off-axis projection of the whole slice is just a shift: (1 - f) * head."""

    def __init__(self, z, surf):
        rect = surf.get_bounding_rect()
        self.surf = surf.subsurface(rect).copy()
        self.ox, self.oy = rect.x, rect.y
        self.k = (1.0 - fz(z)) * SCALE

    def draw(self, screen, hx, hy):
        screen.blit(self.surf, (self.ox - MX + hx * self.k, self.oy - MY + hy * self.k))


def layer_px(x, y, f):
    return (MX + CX + x * f * SCALE, MY + CY + y * f * SCALE)


def noise1d(n, scale, rng, octaves=5, persistence=0.5):
    x = np.arange(n, dtype=np.float32)
    total = np.zeros(n, np.float32)
    amp, freq, tot = 1.0, 1.0 / scale, 0.0
    for _ in range(octaves):
        pts = rng.random(int(n * freq) + 3).astype(np.float32)
        xi = x * freq
        i0 = np.floor(xi).astype(int)
        fr = (1 - np.cos((xi - i0) * np.pi)) / 2
        total += amp * (pts[i0] * (1 - fr) + pts[i0 + 1] * fr)
        tot += amp
        amp *= persistence
        freq *= 2
    v = total / tot
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


def noise2d(h, w, cell, rng):
    small = rng.random((h // cell + 2, w // cell + 2)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def mountain_layer(z, rng, h_lo, h_hi, col_top, col_base, scale_px, octaves, ridged, snow):
    f = fz(z)
    W2, H2 = WIN_WIDTH + 2 * MX, WIN_HEIGHT + 2 * MY
    base = MY + CY + GROUND_Y * f * SCALE
    n = noise1d(W2, scale_px, rng, octaves)
    if ridged:
        n = 1 - np.abs(2 * n - 1)
        n = (n - n.min()) / (n.max() - n.min() + 1e-9)
        n = n ** 1.4
    height = h_lo + (h_hi - h_lo) * n
    ridge = base - height
    rows = np.arange(H2, dtype=np.float32)[:, None]
    below = rows - ridge[None, :]
    alpha = np.clip(below + 0.5, 0, 1)
    alpha[rows[:, 0] > base + 2, :] = 0
    t = np.clip(below / np.maximum(height[None, :], 1), 0, 1)[..., None]
    col = np.array(col_top, np.float32) * (1 - t) + np.array(col_base, np.float32) * t
    tex = noise2d(H2, W2, 18, rng)[..., None]
    col *= 0.88 + 0.24 * tex
    if snow:
        line = h_lo + (h_hi - h_lo) * 0.55
        depth = np.clip(height - line, 0, None)[None, :] * 0.7
        tex2 = noise2d(H2, W2, 7, rng)
        is_snow = (below < depth * (0.6 + 0.8 * tex2)).astype(np.float32)[..., None]
        col = col * (1 - 0.85 * is_snow) + np.array((238, 242, 248), np.float32) * 0.85 * is_snow
    hz = haze_amt(z)
    col = col * (1 - hz) + np.array(HAZE, np.float32) * hz
    rgba = np.dstack([np.clip(col, 0, 255), alpha[..., None] * 255]).astype(np.uint8)
    surf = pygame.image.frombuffer(rgba.tobytes(), (W2, H2), "RGBA").convert_alpha()
    return Layer(z, surf)


def draw_pine(surf, x, base_y, h, hz, rng):
    trunk_h, trunk_w = h * 0.12, max(1.0, h * 0.04)
    pygame.draw.rect(surf, mix((70, 52, 38), HAZE, hz),
                     (x - trunk_w / 2, base_y - trunk_h, trunk_w, trunk_h + 1))
    tiers = int(rng.integers(6, 10))
    dark = mix((24, 58, 34), HAZE, hz)
    lit = mix((52, 98, 52), HAZE, hz)
    for i in range(tiers):
        frac = i / tiers
        yb = base_y - trunk_h * 0.5 - frac * h * 0.82
        yt = base_y - h if i == tiers - 1 else yb - h * 0.26
        hw = h * 0.30 * (1 - frac) ** 0.9 + h * 0.03
        jag = []
        steps = 6
        for j in range(steps + 1):
            jx = x + hw - 2 * hw * j / steps
            jy = yb - (h * 0.035 * rng.random() if 0 < j < steps else 0)
            jag.append((jx, jy))
        shade = 0.85 + 0.15 * frac
        pygame.draw.polygon(surf, mix(dark, (0, 0, 0), 1 - shade), [(x, yt)] + jag)
        pygame.draw.polygon(surf, mix(lit, dark, 0.25 * (1 - frac)),
                            [(x, yt), (x, yb - h * 0.02)] + [p for p in jag if p[0] <= x])


def tree_layer(z, trees, rng):
    f = fz(z)
    surf = pygame.Surface((WIN_WIDTH + 2 * MX, WIN_HEIGHT + 2 * MY), pygame.SRCALPHA)
    hz = haze_amt(z)
    for x, h in trees:
        px, base = layer_px(x, GROUND_Y, f)
        draw_pine(surf, px, base, h * f * SCALE, hz, rng)
    return Layer(z, surf)


def forest_layer(z, rng, density, h_lo, h_hi):
    f = fz(z)
    half_world = (WIN_WIDTH / 2 + MX) / (f * SCALE)
    trees, x = [], -half_world
    while x < half_world:
        h = rng.uniform(h_lo, h_hi)
        trees.append((x, h))
        x += h * rng.uniform(0.15, 0.45) / density
    return tree_layer(z, trees, rng)


def make_cloud(rng, w):
    h = int(w * 0.42)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    puffs = [(rng.uniform(0.2, 0.8) * w, rng.uniform(0.45, 0.7) * h, rng.uniform(0.12, 0.22) * w)
             for _ in range(14)]
    for cx, cy, r in puffs:
        pygame.draw.circle(s, (205, 213, 228, 150), (cx, cy + r * 0.25), r)
    for cx, cy, r in puffs:
        pygame.draw.circle(s, (255, 255, 255, 170), (cx, cy - r * 0.1), r * 0.85)
    small = pygame.transform.smoothscale(s, (max(1, w // 6), max(1, h // 6)))
    return pygame.transform.smoothscale(small, (w, h))


class Scene:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        self.t = 0.0
        # sky: a vertical gradient, infinitely far away
        sh = WIN_HEIGHT + 2 * MY
        col = pygame.Surface((1, sh))
        horizon = MY + CY
        for r in range(sh):
            t = min(1.0, max(0.0, (horizon - r) / (WIN_HEIGHT * 0.9)))
            col.set_at((0, r), mix(SKY_HORIZON, SKY_TOP, t ** 0.7))
        self.sky = pygame.transform.scale(col, (WIN_WIDTH, sh)).convert()
        # clouds
        self.clouds = []
        for _ in range(9):
            w = int(rng.uniform(220, 460))
            self.clouds.append((make_cloud(rng, w), rng.uniform(-1400, 1400),
                                rng.uniform(-430, -170), rng.uniform(5, 14)))
        # far -> near scenery slices
        self.layers = [
            mountain_layer(500, rng, 120, 300, (150, 165, 190), (95, 115, 145), 420, 6, True, True),
            mountain_layer(170, rng, 70, 170, (90, 112, 120), (70, 95, 92), 300, 5, True, False),
            mountain_layer(70, rng, 25, 80, (88, 126, 72), (62, 100, 58), 380, 3, False, False),
            forest_layer(45, rng, 1.0, 3.0, 5.5),
            forest_layer(32, rng, 0.8, 3.5, 6.5),
            tree_layer(26, [(-15, 8), (-8.5, 7), (9.0, 7.5), (16, 8.5)], rng),
            tree_layer(19, [(-11, 8), (-14.5, 6.5), (11.5, 9), (7.2, 6)], rng),
            tree_layer(13, [(7.4, 8.5), (11.5, 7)], rng),
            tree_layer(9.5, [(-6.3, 7.5)], rng),
        ]
        # ground: strips from just in front of the glass out to the horizon
        zs = list(np.linspace(-5.5, 4, 12)) + list(np.geomspace(4, 5000, 70)[1:])
        self.strips = []
        for z0, z1 in zip(zs[:-1], zs[1:]):
            near = min(1.0, max(0.0, (z0 + 5.5) / 30))
            c = mix((58, 104, 40), (100, 142, 66), near)
            c = mix(c, (0, 0, 0), 0.04 * rng.random())
            self.strips.append((z0, z1, mix(c, HAZE, haze_amt((z0 + z1) / 2))))
        # grass tufts and flowers
        self.tufts = []
        for _ in range(380):
            z = -1.8 + 20 * rng.random() ** 1.8
            half = 1.2 * (EYE_DIST + z) + 2
            x = rng.uniform(-half, half)
            hz = haze_amt(z)
            blades = []
            for _ in range(int(rng.integers(3, 6))):
                g = rng.uniform(0.55, 1.0)
                blades.append((rng.uniform(-0.06, 0.06), rng.uniform(0.12, 0.32), rng.uniform(-0.08, 0.08),
                               rng.uniform(0, 6.28), mix(mix((40, 80, 30), (120, 170, 70), g), HAZE, hz)))
            flower = None
            if rng.random() < 0.12:
                flower = [(250, 250, 245), (250, 214, 70), (186, 120, 220), (240, 110, 120)][int(rng.integers(0, 4))]
            self.tufts.append((z, x, blades, flower))
        self.tufts.sort(key=lambda tf: -tf[0])

    def draw_back(self, screen, hx, hy, t):
        screen.blit(self.sky, (0, -MY + hy * SCALE))
        k = (1 - fz(600)) * SCALE
        for img, x0, y, speed in self.clouds:
            x = (x0 + speed * t + 1600) % 3200 - 1600
            screen.blit(img, (CX + x + hx * k - img.get_width() / 2, CY + y + hy * k - img.get_height() / 2))
        for z0, z1, c in self.strips:
            X0, X1 = 1.5 * (EYE_DIST + z0) + 6, 1.5 * (EYE_DIST + z1) + 6
            pygame.draw.polygon(screen, c, [project((-X0, GROUND_Y, z0), hx, hy), project((X0, GROUND_Y, z0), hx, hy),
                                            project((X1, GROUND_Y, z1), hx, hy), project((-X1, GROUND_Y, z1), hx, hy)])
        for layer in self.layers:
            layer.draw(screen, hx, hy)

    def draw_grass(self, screen, hx, hy, t, behind, split_z=ROBOT_Z):
        for z, x, blades, flower in self.tufts:
            if (z > split_z) != behind:
                continue
            w = 2 if z < 2 else 1
            for dx, h, lean, ph, col in blades:
                sway = 0.035 * math.sin(1.6 * t + ph + x * 0.3)
                pygame.draw.line(screen, col, project((x + dx, GROUND_Y, z), hx, hy),
                                 project((x + dx + lean + sway, GROUND_Y - h, z), hx, hy), w)
            if flower:
                p = project((x, GROUND_Y - 0.2, z), hx, hy)
                pygame.draw.circle(screen, flower, p, max(1, int(0.05 * fz(z) * SCALE)))


# ======================= robot =======================
def face_tuple(pts, n, color, glow=False, anchor=None):
    return (pts, n, color, glow, anchor)


def box_faces(center, half, color, yaw=0.0, pitch=0.0, pivot=None, roll=0.0):
    hx_, hy_, hz_ = half
    pivot = pivot or center
    off = sub(center, pivot)
    local = (
        ((0, 0, -1), [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)]),
        ((0, 0, 1), [(1, -1, 1), (-1, -1, 1), (-1, 1, 1), (1, 1, 1)]),
        ((-1, 0, 0), [(-1, -1, 1), (-1, -1, -1), (-1, 1, -1), (-1, 1, 1)]),
        ((1, 0, 0), [(1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)]),
        ((0, -1, 0), [(-1, -1, 1), (1, -1, 1), (1, -1, -1), (-1, -1, -1)]),
        ((0, 1, 0), [(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)]),
    )
    out = []
    for n, corners in local:
        pts = [add(pivot, rot((c[0] * hx_ + off[0], c[1] * hy_ + off[1], c[2] * hz_ + off[2]), yaw, pitch, roll))
               for c in corners]
        out.append(face_tuple(pts, rot(n, yaw, pitch, roll), color))
    return out


def seg_faces(p0, p1, r, color):
    """A square-section beam from p0 to p1 (used for arm segments)."""
    u = norm(sub(p1, p0))
    helper = (0, 0, 1) if abs(u[2]) < 0.9 else (1, 0, 0)
    v = norm(cross(u, helper))
    w = cross(u, v)
    loop = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    ring = lambda p: [add(p, add(mul(v, a * r), mul(w, b * r))) for a, b in loop]
    A, B = ring(p0), ring(p1)
    normals = [v, mul(w, -1), mul(v, -1), w]
    out = [face_tuple([A[i], A[(i + 1) % 4], B[(i + 1) % 4], B[i]], normals[i], color) for i in range(4)]
    out.append(face_tuple(A, mul(u, -1), color))
    out.append(face_tuple(B, u, color))
    return out


def rect_pts(cx, cy, w, h): return [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h)]


def chevron_pts(cx, cy, w, h, th):
    return [(cx - w, cy + h), (cx, cy - h), (cx + w, cy + h), (cx + w - th, cy + h), (cx, cy - h + th * 1.5),
            (cx - w + th, cy + h)]


def heart_pts(cx, cy, s):
    pts = []
    for i in range(24):
        a = 2 * math.pi * i / 24
        x = 16 * math.sin(a) ** 3
        y = 13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a)
        pts.append((cx + x * s / 16, cy - y * s / 16))
    return pts


def ngon_pts(cx, cy, r, n=10, rot0=0.0):
    return [(cx + r * math.cos(rot0 + 2 * math.pi * i / n), cy + r * math.sin(rot0 + 2 * math.pi * i / n))
            for i in range(n)]


def arc_pts(cx, cy, r, th, a0=0.25, a1=2.9, sq=0.55):
    outer = [(cx + r * math.cos(a), cy + r * sq * math.sin(a)) for a in np.linspace(a0, a1, 12)]
    inner = [(cx + (r - th) * math.cos(a), cy + (r - th) * sq * math.sin(a)) for a in np.linspace(a1, a0, 12)]
    return outer + inner


def star_pts(cx, cy, r, points=5, inner=0.45, rot0=-math.pi / 2):
    pts = []
    for i in range(points * 2):
        rr = r if i % 2 == 0 else r * inner
        a = rot0 + math.pi * i / points
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def ellipse_pts(cx, cy, rx, ry, n=18):
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


def rot_rect_pts(cx, cy, hw, hh, th):
    c, s_ = math.cos(th), math.sin(th)
    return [(cx + x * c - y * s_, cy + x * s_ + y * c) for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]


def mouth_pts(smile, op):
    """Lens-shaped mouth: smile > 0 curls the corners up, op opens it."""
    w = 0.2 + 0.05 * max(smile, 0) - 0.04 * max(-smile, 0)
    c = 0.07 * smile
    oh = 0.03 + 0.17 * op
    up, lo = [], []
    for x in np.linspace(-w, w, 15):
        u = x / w
        sq = math.sqrt(max(0.0, 1 - u * u))
        mid = 0.2 + c * (1 - u * u) - c / 2
        th = oh * max(sq, 0.18)
        up.append((x, mid - th / 2))
        lo.append((x, mid + th / 2))
    return up + lo[::-1]


EYE_CYAN, HEART_PINK, MOUTH_DARK, HEAD_DARK = (80, 230, 255), (255, 80, 140), (40, 45, 55), (60, 66, 78)
PUPIL, BROW, TONGUE = (18, 28, 44), (52, 58, 70), (220, 90, 100)


def expression_shapes(mode, face, t, speak=None):
    """Eyes, pupils, brows and mouth as (2D points on the face, colour, glows), head-local units.
    `face` is your mirrored face (or None if no face is seen); `mode` is a special reaction."""
    shapes = []
    f = face or {}
    auto_blink = (int(t * 10) % 40) == 0
    for side, ex in ((-1, -0.22), (1, 0.22)):
        op = f.get("open_l" if side < 0 else "open_r", 1.0) if face else (0.0 if auto_blink else 1.0)
        if mode == "heart":
            shapes.append((heart_pts(ex, -0.08, 0.13), HEART_PINK, True))
        elif mode == "happy":
            shapes.append((chevron_pts(ex, -0.08, 0.12, 0.07, 0.045), EYE_CYAN, True))
        elif mode in ("sleep", "yawn"):
            shapes.append((arc_pts(ex, -0.12, 0.1, 0.03, sq=0.5), EYE_CYAN, True))
        elif mode == "surprised":
            shapes.append((ngon_pts(ex, -0.08, 0.14), EYE_CYAN, True))
            shapes.append((ngon_pts(ex, -0.08, 0.06), PUPIL, False))
        elif mode == "wink" and side > 0:
            shapes.append((rect_pts(ex, -0.06, 0.11, 0.02), EYE_CYAN, True))
        elif op > 0.3 and f.get("smile", 0) > 0.7:
            shapes.append((chevron_pts(ex, -0.08, 0.12, 0.07, 0.045), EYE_CYAN, True))
        else:
            hh = max(0.018, 0.11 * op)
            shapes.append((rect_pts(ex, -0.08, 0.11, hh), EYE_CYAN, True))
            if op > 0.45:
                gx, gy = f.get("gaze", (0.0, 0.0))
                lim = max(0.0, hh - 0.05)
                shapes.append((rect_pts(ex + gx * 0.055, -0.08 + max(-lim, min(lim, gy * 0.05)), 0.045, 0.045),
                               PUPIL, False))
    furrow = f.get("furrow", 0.0)
    for side, ex in ((-1, -0.22), (1, 0.22)):
        r = f.get("raise_l" if side < 0 else "raise_r", 0.0)
        if mode == "surprised":
            r = 1.0
        elif mode == "sleep":
            r = -0.3
        tilt = 0.35 * furrow * (1 if side < 0 else -1)
        shapes.append((rot_rect_pts(ex, -0.27 - 0.1 * r, 0.13, 0.025, tilt), BROW, False))
    if speak is not None:          # talking with your recorded voice
        shapes.append((mouth_pts(0.5, speak), MOUTH_DARK, False))
        if speak > 0.4:
            shapes.append((ellipse_pts(0, 0.2 + 0.06 * speak, 0.07, 0.035 * speak), TONGUE, False))
    elif mode == "yawn":
        shapes.append((ellipse_pts(0, 0.2, 0.12, 0.16), MOUTH_DARK, False))
        shapes.append((ellipse_pts(0, 0.29, 0.07, 0.05), TONGUE, False))
    elif mode == "pucker" or f.get("pucker"):
        shapes.append((ngon_pts(0, 0.2, 0.05), MOUTH_DARK, False))
    elif mode == "surprised":
        shapes.append((ngon_pts(0, 0.2, 0.08), MOUTH_DARK, False))
    elif mode in ("happy", "heart", "wink"):
        shapes.append((arc_pts(0, 0.1, 0.24, 0.06), MOUTH_DARK, False))
    elif mode == "sleep":
        shapes.append((rect_pts(0, 0.22, 0.1, 0.02), MOUTH_DARK, False))
    elif face:
        op = f.get("mouth_open", 0.0)
        shapes.append((mouth_pts(f.get("smile", 0.0), op), MOUTH_DARK, False))
        if op > 0.4:
            shapes.append((ellipse_pts(0, 0.2 + 0.06 * op, 0.07, 0.035 * op), TONGUE, False))
    else:
        shapes.append((rect_pts(0, 0.2, 0.25, 0.035), MOUTH_DARK, False))
    return shapes


FINGER, FINGER_TIP = (205, 211, 222), (120, 200, 230)
FINGERS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
PALM = (0, 1, 5, 9, 13, 17)


def hand_faces(P, color, dark):
    """Palm + 5 jointed fingers from 21 points (same layout as MediaPipe's hand)."""
    out = []
    pal = [P[i] for i in PALM]
    n = norm(cross(sub(P[5], P[0]), sub(P[17], P[0])))
    out.append(face_tuple(pal, n, color))
    out.append(face_tuple(pal[::-1], mul(n, -1), color))
    for chain in FINGERS:
        for a, b in zip(chain[:-1], chain[1:]):
            r = 0.04 if a in (0, 1) else 0.03
            if a == 0 and b != 1:
                continue            # palm already covers wrist -> knuckle
            out += seg_faces(P[a], P[b], r, FINGER_TIP if b in (4, 8, 12, 16, 20) else FINGER)
    return out


THIGH, SHIN, FOOT_H = 0.33, 0.31, 0.06     # robot leg sizes (hip to floor = 0.7)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Robot:
    def __init__(self):
        self.fingers = {-1: None, 1: None}
        self.finger_seen = {-1: -9.0, 1: -9.0}
        self.arm = {s: (norm((0.12 * s, 1, 0)), norm((0.12 * s, 1, 0))) for s in (-1, 1)}
        self.leg = {s: (norm((0.06 * s, 1, 0)), (0.0, 1.0, 0.0), norm((0, 0.3, -1))) for s in (-1, 1)}
        self.torso_yaw = self.hip_yaw = 0.0
        self.bend = (0.0, 0.0)               # (forward bend, side lean)
        self.tracked = False                 # are we copying your body's turning?
        self.root_x, self.root_z, self.jump = 0.0, ROBOT_Z, 0.0
        self.ref_torso, self.full_seen = None, -99.0
        self.feet_hist = []
        self.root = (0.0, GROUND_Y - 0.7, ROBOT_Z)
        self.head_c = (0.0, -0.8, ROBOT_Z)
        self.hands = {s: (0.7 * s, 0.8, ROBOT_Z) for s in (-1, 1)}
        self.head_top = (0.0, -1.8, ROBOT_Z)
        self.mouth = (0.0, -0.6, ROBOT_Z - 0.5)

    def update(self, body, t):
        b = body or {}
        # ----- turning and bending -----
        ty = b.get("torso_yaw")
        self.tracked = ty is not None
        if self.tracked:
            hy = b.get("hip_yaw")
            hy = ty if hy is None else hy
            self.torso_yaw += wrap(ty - self.torso_yaw) * BODY_SMOOTH
            self.hip_yaw += wrap(hy - self.hip_yaw) * BODY_SMOOTH
        else:
            self.torso_yaw *= 1 - BODY_SMOOTH * 0.3
            self.hip_yaw *= 1 - BODY_SMOOTH * 0.3
        bp, br = b.get("bend") or (0.0, 0.0)
        self.bend = (self.bend[0] + (bp - self.bend[0]) * BODY_SMOOTH, self.bend[1] + (br - self.bend[1]) * BODY_SMOOTH)

        # ----- arms (absolute directions; idle arms hang from the shoulders) -----
        for s in (-1, 1):
            tgt = b.get("arms", {}).get(s)
            if tgt is None:
                idle = rot(norm((0.12 * s, 1, 0.08 * math.sin(t * 2.0 + s))), self.torso_yaw, self.bend[0], self.bend[1])
                tgt = (idle, idle)
            self.arm[s] = tuple(norm(add(c, mul(sub(g, c), ARM_SMOOTH))) for c, g in zip(self.arm[s], tgt))
            # fingers: copy your hand's shape, scaled to robot size, around the robot's wrist
            hand = b.get("hands", {}).get(s)
            if hand:
                x0, y0, z0 = hand[0]
                size = math.dist(hand[0], hand[9]) + 1e-6
                k = 0.46 / size
                new = [((x - x0) * k, (y - y0) * k, (z - z0) * k) for x, y, z in hand]
                old = self.fingers[s]
                self.fingers[s] = new if old is None else [
                    add(o, mul(sub(n_, o), FINGER_SMOOTH)) for o, n_ in zip(old, new)]
                self.finger_seen[s] = t
            elif t - self.finger_seen[s] > 0.4:
                self.fingers[s] = None

        # ----- legs -----
        for s in (-1, 1):
            tgt = b.get("legs", {}).get(s)
            if tgt is None:                 # can't see that leg: stand straight
                tgt = (rot(norm((0.06 * s, 1, 0)), self.hip_yaw, 0), (0.0, 1.0, 0.0), rot(norm((0, 0.3, -1)), self.hip_yaw, 0))
            elif tgt[2] is None:
                tgt = (tgt[0], tgt[1], rot(norm((0, 0.3, -1)), self.hip_yaw, 0))
            self.leg[s] = tuple(norm(add(c, mul(sub(g, c), LEG_SMOOTH))) for c, g in zip(self.leg[s], tgt))

        # ----- walking around and jumping (only when your whole body is in view) -----
        if b.get("full"):
            self.full_seen = t
            if self.ref_torso is None:
                self.ref_torso = b["torso_len"]
            tx = max(-2.4, min(2.4, b["pos_x"] * WALK_GAIN_X))
            tz = ROBOT_Z + (self.ref_torso / max(b["torso_len"], 1e-3) - 1) * WALK_GAIN_Z
            tz = max(2.6, min(7.0, tz))
            self.root_x += (tx - self.root_x) * BODY_SMOOTH
            self.root_z += (tz - self.root_z) * BODY_SMOOTH
            self.feet_hist = [(tt, v) for tt, v in self.feet_hist if t - tt < 0.6] + [(t, b["feet_y"])]
            floor = max(v for _, v in self.feet_hist)            # where your feet were standing
            rise = (floor - b["feet_y"]) / max(b["leg_len"], 1e-3)
            target_jump = max(0.0, rise - 0.03) * 0.7 * 1.6 * JUMP_GAIN
            self.jump += (target_jump - self.jump) * 0.7
        else:
            if t - self.full_seen > 10:
                self.ref_torso = None
            self.feet_hist = []
            self.root_x += (0.0 - self.root_x) * 0.05
            self.root_z += (ROBOT_Z - self.root_z) * 0.05
            self.jump *= 0.6

    update_arms = update        # old name

    def build(self, look_at, t, mode, face, together, head_off=(0.0, 0.0), speak=None):
        faces = []
        grey, dark = (175, 182, 194), (98, 104, 118)
        accent = HEART_PINK if mode == "heart" else (240, 140, 50)

        # the legs decide how high the hips are: feet stay on the ground unless you jump
        ext = max(THIGH * self.leg[s][0][1] + SHIN * self.leg[s][1][1] for s in (-1, 1))
        root = (self.root_x, GROUND_Y - FOOT_H - max(ext, 0.15) - self.jump, self.root_z)

        head_guess = add(root, (0, -1.9, 0))
        d = sub(look_at, head_guess)
        look_yaw = math.atan2(-d[0], -d[2])
        look_pitch = max(-0.7, min(0.7, math.atan2(d[1], math.hypot(d[0], d[2]))))
        torso_yaw = self.torso_yaw if self.tracked else look_yaw * 0.45
        hip_yaw = self.hip_yaw if self.tracked else torso_yaw
        bp, br = self.bend
        R = lambda v: add(root, rot(v, torso_yaw, bp, br))

        # legs and feet
        for s in (-1, 1):
            thigh, shin, foot = self.leg[s]
            hip = add(root, rot((s * 0.26, 0.02, 0), hip_yaw))
            knee = add(hip, mul(thigh, THIGH))
            ankle = add(knee, mul(shin, SHIN))
            faces += seg_faces(hip, knee, 0.13, dark)
            faces += box_faces(knee, (0.1, 0.1, 0.1), grey, hip_yaw, 0)
            faces += seg_faces(knee, ankle, 0.115, dark)
            faces += seg_faces(add(ankle, mul(foot, -0.07)), add(ankle, mul(foot, 0.26)), 0.075, grey)
        faces += box_faces(add(root, (0, -0.02, 0)), (0.4, 0.12, 0.24), dark, hip_yaw, 0)

        # torso, chest light, neck
        faces += box_faces(R((0, -0.5, 0)), (0.55, 0.55, 0.32), grey, torso_yaw, bp, roll=br)
        chest = [R((x, y, -0.33)) for x, y in rect_pts(0, -0.55, 0.22, 0.15)]
        faces.append(face_tuple(chest, rot((0, 0, -1), torso_yaw, bp, br), accent, True, R((0, -0.5, -0.34))))
        faces += box_faces(R((0, -1.15, 0)), (0.15, 0.12, 0.15), dark, torso_yaw, bp, roll=br)

        # arms copy yours
        shoulders, elbows, hands = {}, {}, {}
        for s in (-1, 1):
            sh = R((s * 0.68, -0.97, 0))
            up, fo = self.arm[s]
            shoulders[s] = sh
            elbows[s] = add(sh, mul(up, 0.55))
            hands[s] = add(elbows[s], mul(fo, 0.55))
        if together:   # pull the hands together so a clap actually meets
            mid = mul(add(hands[-1], hands[1]), 0.5)
            for s in (-1, 1):
                hands[s] = add(mid, rot((s * 0.08, 0, 0), torso_yaw, 0))
        for s in (-1, 1):
            faces += box_faces(shoulders[s], (0.15, 0.15, 0.15), grey, torso_yaw, bp, roll=br)
            faces += seg_faces(shoulders[s], elbows[s], 0.115, dark)
            faces += seg_faces(elbows[s], hands[s], 0.1, dark)
            if self.fingers[s]:
                faces += hand_faces([add(hands[s], o) for o in self.fingers[s]], grey, dark)
            else:
                faces += box_faces(hands[s], (0.11, 0.11, 0.11), grey, torso_yaw, 0)

        # head: follows the body, turns (as far as a neck can) to look at you, copies your face
        head_c = R((0, -1.9, 0))
        yaw = torso_yaw + max(-1.1, min(1.1, wrap(look_yaw - torso_yaw))) + head_off[0]
        pitch = max(-0.9, min(1.3, look_pitch + 0.7 * bp)) + head_off[1]
        roll = (face["roll"] if face else 0.0) + 0.5 * br
        faces += box_faces(head_c, (0.5, 0.42, 0.42), grey, yaw, pitch, roll=roll)
        anchor = add(head_c, rot((0, 0, -0.44), yaw, pitch, roll))
        normal = rot((0, 0, -1), yaw, pitch, roll)
        for pts2d, color, glow in expression_shapes(mode, face, t, speak):
            pts = [add(head_c, rot((x, y, -0.43), yaw, pitch, roll)) for x, y in pts2d]
            faces.append(face_tuple(pts, normal, color, glow, anchor))
        tip = add(head_c, rot((0, -0.75, 0), yaw, pitch, roll))
        faces += box_faces(add(head_c, rot((0, -0.58, 0), yaw, pitch, roll)), (0.03, 0.16, 0.03), dark,
                           yaw, pitch, roll=roll)
        faces += box_faces(tip, (0.07, 0.07, 0.07), (255, 60, 60), yaw, pitch, roll=roll)

        self.root, self.head_c, self.hands = root, head_c, hands
        self.head_top = add(head_c, (0, -1.0, 0))
        self.mouth = add(head_c, rot((0, 0.2, -0.5), yaw, pitch, roll))
        return faces


def shade(color, n, glow):
    if glow:
        return color
    k = 0.42 + 0.58 * max(0.0, dot(n, TO_LIGHT))
    c = tuple(min(255, int(ch * k)) for ch in color)
    sky = max(0.0, -n[1]) * 0.25   # upward faces pick up a little blue sky
    return mix(c, SKY_TOP, sky)


def draw_faces(screen, faces, eye, hx, hy):
    visible = []
    for pts, n, color, glow, anchor in faces:
        k = len(pts)
        c = (sum(p[0] for p in pts) / k, sum(p[1] for p in pts) / k, sum(p[2] for p in pts) / k)
        to_eye = sub(eye, c)
        if dot(n, to_eye) <= 0:
            continue
        if anchor:   # face decal: sort right after the face it sits on
            a = sub(eye, anchor)
            dist = dot(a, a) - 1e-4
        else:
            dist = dot(to_eye, to_eye)
        visible.append((dist, pts, n, color, glow))
    visible.sort(key=lambda v: -v[0])
    for _, pts, n, color, glow in visible:
        poly = [project(p, hx, hy) for p in pts]
        c = shade(color, n, glow)
        pygame.draw.polygon(screen, c, poly)
        pygame.draw.aalines(screen, c, True, poly)


def draw_shadow(screen, hx, hy, robot):
    x0, z0 = robot.root[0], robot.root[2]
    k = 1.0 / (1.0 + 1.5 * robot.jump)
    for r, a in ((1.0, 40), (0.75, 45), (0.5, 50)):
        pts = [project((x0 + k * r * math.cos(q), GROUND_Y, z0 + k * 0.55 * r * math.sin(q)), hx, hy)
               for q in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
        pygame.gfxdraw.filled_polygon(screen, pts, (20, 40, 15, a))


# ======================= effects =======================
class Particle:
    def __init__(self, kind, pos, vel, life, size, color, spin=0.0):
        self.kind, self.pos, self.vel = kind, pos, vel
        self.life = self.max_life = life
        self.size, self.color, self.spin, self.angle = size, color, spin, random.uniform(0, 6.28)


class Bubble:
    def __init__(self, text, font, t, dur=1.8):
        txt = font.render(text, True, (35, 40, 55))
        pad = 18
        w, h = txt.get_width() + 2 * pad, txt.get_height() + pad
        self.surf = pygame.Surface((w, h + 16), pygame.SRCALPHA)
        pygame.draw.rect(self.surf, (255, 255, 255, 235), (0, 0, w, h), border_radius=h // 2)
        pygame.draw.polygon(self.surf, (255, 255, 255, 235), [(w / 2 - 12, h - 2), (w / 2 + 12, h - 2), (w / 2, h + 14)])
        self.surf.blit(txt, (pad, pad // 2))
        self.t0, self.dur = t, dur

    def draw(self, screen, pos, t):
        age = t - self.t0
        if age > self.dur:
            return False
        pop = min(1.0, age / 0.15)
        s = 0.6 + 0.4 * pop + 0.08 * math.sin(min(1.0, age / 0.3) * math.pi)
        img = pygame.transform.smoothscale(self.surf, (max(1, int(self.surf.get_width() * s)),
                                                       max(1, int(self.surf.get_height() * s))))
        if age > self.dur - 0.3:
            img.set_alpha(int(255 * (self.dur - age) / 0.3))
        screen.blit(img, (pos[0] - img.get_width() / 2, pos[1] - img.get_height()))
        return True


class FX:
    def __init__(self, font):
        self.parts, self.bubble, self.font = [], None, font
        self.zsurf = font.render("Z", True, (235, 240, 255))

    def say(self, text, t):
        self.bubble = Bubble(text, self.font, t)

    def _burst(self, kind, pos, n, speed, life, size, colors, up=0.0, spin=0.0):
        for _ in range(n):
            d = norm((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)))
            v = add(mul(d, speed * random.uniform(0.5, 1.0)), (0, -up, 0))
            self.parts.append(Particle(kind, pos, v, life * random.uniform(0.7, 1.0), size * random.uniform(0.7, 1.2),
                                       random.choice(colors), random.uniform(-spin, spin)))

    def sparkles(self, pos):
        self._burst("sparkle", pos, 16, 2.6, 0.8, 0.09, [(255, 240, 150), (255, 255, 255), (255, 210, 90)])

    def stars(self, pos):
        self._burst("star", pos, 12, 1.6, 1.5, 0.12, [(255, 220, 60), (255, 245, 140)], up=0.8, spin=3)

    def hearts(self, pos):
        self._burst("heart", pos, 10, 0.6, 2.2, 0.14, [HEART_PINK, (255, 120, 170), (230, 50, 90)], up=1.0)

    def confetti(self, pos):
        cols = [(255, 90, 90), (255, 200, 60), (90, 200, 255), (130, 230, 120), (200, 120, 255)]
        self._burst("confetti", pos, 80, 4.0, 2.6, 0.07, cols, up=2.5, spin=8)

    def snooze(self, pos):
        self.parts.append(Particle("z", pos, (0.35, -0.55, 0.0), 2.4, 0.22, (255, 255, 255)))

    def kiss(self, pos):
        self.parts.append(Particle("heart", pos, (0.0, -0.25, -3.2), 1.5, 0.16, HEART_PINK))

    def update(self, dt):
        alive = []
        for p in self.parts:
            p.life -= dt
            if p.life <= 0:
                continue
            g = {"confetti": 4.0, "sparkle": 0.5, "star": 0.8, "heart": -0.2, "z": 0.0}[p.kind]
            drag = 0.0 if p.kind == "z" or p.vel[2] < -1 else 1.2
            p.vel = (p.vel[0] * (1 - drag * dt), p.vel[1] + g * dt, p.vel[2] * (1 - drag * dt))
            p.pos = add(p.pos, mul(p.vel, dt))
            p.angle += p.spin * dt
            alive.append(p)
        self.parts = alive

    def draw(self, screen, hx, hy, t, head_top):
        for p in sorted(self.parts, key=lambda q: -q.pos[2]):
            if p.pos[2] <= -EYE_DIST + 2.0:
                continue
            x, y = project(p.pos, hx, hy)
            s = p.size * fz(p.pos[2]) * SCALE
            a = int(255 * min(1.0, p.life / 0.4))
            if p.kind == "z":
                grow = 1.0 + (p.max_life - p.life) * 0.5
                hgt = max(4, int(s * 2.2 * grow))
                wid = max(3, int(hgt * self.zsurf.get_width() / self.zsurf.get_height()))
                img = pygame.transform.smoothscale(self.zsurf, (wid, hgt))
                img.set_alpha(a)
                screen.blit(img, (x - wid / 2, y - hgt / 2))
                continue
            if p.kind == "heart":
                pts = heart_pts(x + 6 * math.sin(3 * t + p.angle), y, s)
            elif p.kind == "star":
                pts = star_pts(x, y, s, rot0=p.angle)
            elif p.kind == "sparkle":
                pts = star_pts(x, y, s * (0.6 + 0.4 * math.sin(20 * p.life)), points=4, inner=0.25, rot0=p.angle)
            else:
                c, sn = math.cos(p.angle), math.sin(p.angle)
                w, h = s, s * 0.5 * abs(math.sin(p.angle * 1.7)) + 1
                pts = [(x + c * dx - sn * dy, y + sn * dx + c * dy) for dx, dy in ((-w, -h), (w, -h), (w, h), (-w, h))]
            if len(pts) >= 3 and s > 0.5:
                pygame.gfxdraw.filled_polygon(screen, pts, (*p.color, a))
                pygame.gfxdraw.aapolygon(screen, pts, (*p.color, a))
        if self.bubble and not self.bubble.draw(screen, project(head_top, hx, hy), t):
            self.bubble = None


# ======================= reading your body =======================
def dir3(a, b, ref):
    """3D direction of a limb from its 2D image. If it looks shorter than its real
    length (ref), the missing length is pointing toward the camera."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L >= ref:
        return norm((dx, dy, 0.0))
    return norm((dx, dy, -math.sqrt(ref * ref - L * L)))


def hand_gesture(P):
    d = lambda a, b: math.dist(P[a], P[b])
    size = d(0, 9) + 1e-6
    fingers = ((8, 6), (12, 10), (16, 14), (20, 18))
    ext = [d(0, tip) > d(0, pip) * 1.15 for tip, pip in fingers]
    fold = [d(0, tip) < d(0, pip) for tip, pip in fingers]
    thumb_up = (P[4][1] < P[3][1] < P[2][1] and P[2][1] - P[4][1] > 0.45 * size
                and P[4][1] < min(P[i][1] for i in range(5, 21)))
    if all(fold) and thumb_up:
        return "thumbs_up"
    if ext[0] and fold[1] and fold[2] and ext[3]:
        return "love"          # the I-love-you sign (or rock-on, close enough!)
    if ext[0] and ext[1] and fold[2] and fold[3]:
        return "peace"
    if all(ext):
        return "open"
    if all(fold):
        return "fist"
    return None


EYES = ((33, 133, 159, 145, 105, 55), (362, 263, 386, 374, 334, 285))  # corners, lids, mid brow, inner brow


def face_metrics(F, px):
    P = lambda i: px(F[i])
    eyes = []
    for c1, c2, up, lo, brow, inner in EYES:
        a, b = P(c1), P(c2)
        w = math.dist(a, b) + 1e-6
        center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        eyes.append({"c": center, "w": w, "ear": math.dist(P(up), P(lo)) / w,
                     "brow": center[1] - P(brow)[1], "inner": P(inner)})
    L, R = sorted(eyes, key=lambda e: e["c"][0])          # screen-left eye first
    fw = math.dist(P(234), P(454)) + 1e-6
    fh = math.dist(P(10), P(152)) + 1e-6
    m = {"ear_l": L["ear"], "ear_r": R["ear"], "brow_l": L["brow"] / fh, "brow_r": R["brow"] / fh,
         "inner": math.dist(L["inner"], R["inner"]) / fw,
         "width": math.dist(P(61), P(291)) / fw,
         "lift": (P(13)[1] - (P(61)[1] + P(291)[1]) / 2) / fh,   # corners vs upper lip
         "gap": math.dist(P(13), P(14)) / fh,
         "roll": math.atan2(R["c"][1] - L["c"][1], R["c"][0] - L["c"][0]),
         "nod": (P(1)[1] - (L["c"][1] + R["c"][1]) / 2) / fh,
         "turn": (P(1)[0] - (P(234)[0] + P(454)[0]) / 2) / fw,
         "gaze": None}
    if len(F) >= 478:   # iris points exist
        irises = (P(468), P(473))
        g = []
        for e in (L, R):
            ir = min(irises, key=lambda q: math.dist(q, e["c"]))
            g.append(((ir[0] - e["c"][0]) / e["w"], (ir[1] - e["c"][1]) / e["w"]))
        m["gaze"] = ((g[0][0] + g[1][0]) / 2, (g[0][1] + g[1][1]) / 2)
    return m


SET_A = (11, 13, 15, 23, 25, 27, 29, 31)   # shoulder, elbow, wrist, hip, knee, ankle, heel, toe
SET_B = (12, 14, 16, 24, 26, 28, 30, 32)   # the same on the other side of the body


def analyze(res, w, h, state=None):
    state = {} if state is None else state
    body = {"head": None, "arms": {}, "wrist_ratio": None, "both_up": False, "one_up": False,
            "face": None, "gestures": set(), "hands": {}, "legs": {}, "torso_yaw": None, "hip_yaw": None,
            "bend": None, "full": False}
    mid_x = w / 2
    wrists = {}
    px = lambda l: (l.x * w, l.y * h)
    if res.face_landmarks:
        F = res.face_landmarks.landmark
        body["head"] = ((F[168].x - 0.5) * 2, (F[168].y - 0.5) * 2)
        body["face"] = face_metrics(F, px)
    if res.pose_landmarks:
        L = res.pose_landmarks.landmark
        if body["head"] is None and L[0].visibility > 0.5:
            body["head"] = ((L[0].x - 0.5) * 2, (L[0].y - 0.5) * 2)
        WL = res.pose_world_landmarks.landmark if getattr(res, "pose_world_landmarks", None) else None
        vis = lambda i: L[i].visibility > 0.5
        W3 = lambda i: (WL[i].x, WL[i].y, WL[i].z)      # metres, centred between your hips
        if vis(11) and vis(12):
            s1, s2 = px(L[11]), px(L[12])
            sw = math.dist(s1, s2) + 1e-6
            mid_x = (s1[0] + s2[0]) / 2
            nose_y = px(L[0])[1]
            # Which side of your body is the robot's right? Learned while your face is visible
            # (you're facing the camera), then kept, so turning all the way around still works.
            facing = True
            if WL:   # only trust left/right from the image when you're roughly facing the camera
                v3 = sub(W3(11), W3(12))
                facing = abs(v3[0]) > 0.8 * math.sqrt(dot(v3, v3))
            if "sideA" not in state or (body["face"] is not None and facing):
                state["sideA"] = 1 if s1[0] > s2[0] else -1
            sides = {state["sideA"]: SET_A, -state["sideA"]: SET_B}
            for side, (si, ei, wi, hi, ki, ai, he, ti) in sides.items():
                if not vis(ei):
                    continue
                S, E = px(L[si]), px(L[ei])
                upper = norm(sub(W3(ei), W3(si))) if WL else dir3(S, E, 0.85 * sw)
                fore = upper
                if vis(wi):
                    Wp = px(L[wi])
                    fore = norm(sub(W3(wi), W3(ei))) if WL else dir3(E, Wp, 0.75 * sw)
                    wrists[side] = (Wp, S)
                body["arms"][side] = (upper, fore)
            if WL:
                sv = sub(W3(sides[1][0]), W3(sides[-1][0]))
                body["torso_yaw"] = math.atan2(-sv[2], sv[0])
                if vis(23) and vis(24):
                    hv = sub(W3(sides[1][3]), W3(sides[-1][3]))
                    body["hip_yaw"] = math.atan2(-hv[2], hv[0])
                    up = norm(sub(mul(add(W3(11), W3(12)), 0.5), mul(add(W3(23), W3(24)), 0.5)))
                    ul = rot(up, -body["torso_yaw"], 0)          # undo the turn, then read bend and lean
                    body["bend"] = (max(-0.5, min(1.4, math.atan2(-ul[2], -ul[1]))),
                                    max(-0.6, min(0.6, math.atan2(ul[0], -ul[1]))))
                    for side, (si, ei, wi, hi, ki, ai, he, ti) in sides.items():
                        if vis(ki) and vis(ai):
                            foot = norm(sub(W3(ti), W3(he))) if vis(ti) and vis(he) else None
                            body["legs"][side] = (norm(sub(W3(ki), W3(hi))), norm(sub(W3(ai), W3(ki))), foot)
                    if len(body["legs"]) == 2:                   # whole body in view
                        H1, H2, A1, A2 = px(L[23]), px(L[24]), px(L[27]), px(L[28])
                        hip_c = ((H1[0] + H2[0]) / 2, (H1[1] + H2[1]) / 2)
                        sh_c = ((s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2)
                        body["full"] = True
                        body["pos_x"] = (hip_c[0] / w - 0.5) * 2
                        body["feet_y"] = max(A1[1], A2[1]) / h
                        body["leg_len"] = (math.dist(H1, A1) + math.dist(H2, A2)) / 2 / h
                        body["torso_len"] = math.dist(hip_c, sh_c) / h
            if len(wrists) == 2:
                body["wrist_ratio"] = math.dist(wrists[1][0], wrists[-1][0]) / sw
                body["both_up"] = wrists[1][0][1] < nose_y and wrists[-1][0][1] < nose_y
            ups = [s for s, (W, S) in wrists.items() if W[1] < S[1] - 0.35 * sw]
            body["one_up"] = len(ups) == 1
    for hand in (res.left_hand_landmarks, res.right_hand_landmarks):
        if hand:
            g = hand_gesture([px(l) for l in hand.landmark])
            if g:
                body["gestures"].add(g)
            pts = [(l.x * w, l.y * h, l.z * w) for l in hand.landmark]
            if wrists:   # belongs to whichever of your wrists it's closest to
                side = min(wrists, key=lambda sd: math.dist(wrists[sd][0], pts[0][:2]))
            else:
                side = 1 if pts[0][0] > mid_x else -1
            body["hands"][side] = pts
    return body


def count_swings(vals, thr):
    """How many back-and-forth movements bigger than thr are in vals."""
    if not vals:
        return 0
    n, direction, ext = 0, 0, vals[0]
    for v in vals[1:]:
        if direction == 0:
            if abs(v - ext) > thr:
                direction, ext, n = (1 if v > ext else -1), v, n + 1
        elif direction == 1:
            if v > ext:
                ext = v
            elif ext - v > thr:
                direction, ext, n = -1, v, n + 1
        else:
            if v < ext:
                ext = v
            elif v - ext > thr:
                direction, ext, n = 1, v, n + 1
    return n


# ======================= sound =======================
def load_wav(path):
    with wave.open(path, "rb") as wf:
        sr, ch, sw = wf.getframerate(), wf.getnchannels(), wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype={1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    if sw == 1:
        data = (data - 128) / 128
    else:
        data /= float(2 ** (8 * sw - 1))
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def save_wav(path, data, sr):
    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def resample(data, sr_from, sr_to):
    if sr_from == sr_to:
        return data
    n = int(len(data) * sr_to / sr_from)
    return np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)


class Audio:
    """Simple robot beeps (made from maths, no files) + your recorded voice lines."""

    def __init__(self):
        self.ok, self.sfx, self.voices = False, {}, {}
        self.talk_env, self.talk_t0 = None, 0.0
        try:
            pygame.mixer.init(22050, -16, 1, 512)
            self.sr, _, self.ch = pygame.mixer.get_init()
            pygame.mixer.set_num_channels(8)
            pygame.mixer.set_reserved(1)
            self.voice_ch = pygame.mixer.Channel(0)
            self.ok = True
        except Exception as e:
            print("(No sound:", e, ")")
            return
        notes = {
            "clap":  [("noise", 0.035)],
            "spark": [(1568, 0.05), (2093, 0.05), (2637, 0.08)],
            "yay":   [(523, 0.08), (659, 0.08), (784, 0.08), (1047, 0.18)],
            "hi":    [(880, 0.09), (660, 0.14)],
            "yes":   [(660, 0.08), (990, 0.14)],
            "no":    [(440, 0.1), (330, 0.18)],
            "love":  [(784, 0.1), (988, 0.1), (1175, 0.22)],
            "kiss":  [("sweep", 1400, 500, 0.22)],
            "yawn":  [("sweep", 520, 220, 0.7)],
            "boing": [("sweep", 300, 760, 0.16)],
            "alarm": [(880, 0.1), (660, 0.1), (880, 0.1), (660, 0.1)],
        }
        for name, seq in notes.items():
            self.sfx[name] = self._make(self._synth(seq), SOUND_VOLUME)
        for key, _ in VOICE_LINES:
            path = os.path.join(VOICE_DIR, key + ".wav")
            if os.path.exists(path):
                try:
                    data, sr = load_wav(path)
                    data = resample(data, sr, self.sr)
                    blk = max(1, int(self.sr * 0.03))
                    env = np.array([np.sqrt(np.mean(data[i:i + blk] ** 2)) for i in range(0, len(data), blk)])
                    env = np.sqrt(env / (env.max() + 1e-9))
                    self.voices[key] = (self._make(data, 1.0), env)
                except Exception as e:
                    print("(Couldn't load", path, ":", e, ")")

    def _synth(self, seq):
        out = []
        for item in seq:
            if item[0] == "noise":
                n = int(self.sr * item[1])
                w = np.random.uniform(-1, 1, n) * np.exp(-np.linspace(0, 8, n))
            elif item[0] == "sweep":
                _, f0, f1, dur = item
                n = int(self.sr * dur)
                f = np.linspace(f0, f1, n) * (1 + 0.03 * np.sin(np.linspace(0, 40, n)))
                w = np.sin(2 * np.pi * np.cumsum(f) / self.sr)
            else:
                f, dur = item
                tt = np.arange(int(self.sr * dur)) / self.sr
                w = np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 2 * f * tt)   # slightly robotic
            n = len(w)
            env = np.minimum(1, np.minimum(np.arange(n) / (0.005 * self.sr + 1), (n - np.arange(n)) / (0.03 * self.sr + 1)))
            out.append(w * env)
            out.append(np.zeros(int(self.sr * 0.015)))
        return np.concatenate(out).astype(np.float32)

    def _make(self, data, vol):
        pcm = (np.clip(data * vol, -1, 1) * 32767).astype(np.int16)
        if self.ch == 2:
            pcm = np.repeat(pcm[:, None], 2, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(pcm))

    def beep(self, name):
        if self.ok and name in self.sfx:
            self.sfx[name].play()

    def say(self, key, t):
        """Play your recorded line. Returns False if it hasn't been recorded."""
        if not self.ok or key not in self.voices:
            return False
        snd, env = self.voices[key]
        self.voice_ch.play(snd)
        self.talk_env, self.talk_t0 = env, t
        return True

    def mouth(self, t):
        """How open the robot's mouth should be right now while talking (None = not talking)."""
        if self.talk_env is None:
            return None
        i = int((t - self.talk_t0) / 0.03)
        if i >= len(self.talk_env) or not self.voice_ch.get_busy():
            self.talk_env = None
            return None
        return float(min(1.0, self.talk_env[i] * 1.1))


# ======================= recording your voice =======================
def trim_and_normalize(x, sr):
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak < 0.02:
        return None
    loud = np.where(np.abs(x) > 0.08 * peak)[0]
    a, b = max(0, loud[0] - int(0.08 * sr)), min(len(x), loud[-1] + int(0.15 * sr))
    clip = x[a:b] / peak * 0.9
    fade = min(len(clip) // 4, int(0.01 * sr))
    if fade > 0:
        clip[:fade] *= np.linspace(0, 1, fade)
        clip[-fade:] *= np.linspace(1, 0, fade)
    return clip.astype(np.float32)


def record_voices():
    import sounddevice as sd
    os.makedirs(VOICE_DIR, exist_ok=True)
    sr = 22050
    print("\n  Record the robot's voice lines.")
    print("  The first time, macOS may ask to let Terminal use the microphone - click OK/Allow.\n")
    for key, text in VOICE_LINES:
        path = os.path.join(VOICE_DIR, key + ".wav")
        while True:
            have = " (already recorded - Enter to re-record, s to keep)" if os.path.exists(path) else ""
            ans = input(f'  Next: "{text}"{have}\n  Press Enter, then say it (3 seconds), or type s to skip: ')
            if ans.strip().lower() == "s":
                break
            print("  Recording... speak now!")
            data = sd.rec(int(3 * sr), samplerate=sr, channels=1, dtype="float32")
            sd.wait()
            clip = trim_and_normalize(data[:, 0], sr)
            if clip is None:
                print("  I didn't hear anything. If macOS asked about the microphone, allow it, then try again.\n")
                continue
            print("  Playing it back...")
            sd.play(clip, sr)
            sd.wait()
            if input("  Keep it? [Y/n] ").strip().lower() in ("", "y", "yes"):
                save_wav(path, clip, sr)
                print("  Saved.\n")
                break
    print("  Done. Run  robot  and the robot will use your voice.\n")


# ======================= installer builder =======================
INSTALLER_HEAD = r"""#!/bin/bash
# ------------------------------------------------------------------
#  Off-Axis Robot - one-time installer for macOS
#  Run it with:   bash install_mac.sh
#  Afterwards, open a new Terminal window and type:   robot
# ------------------------------------------------------------------
set -e
APP="$HOME/.offaxis-robot"
UV="$HOME/.local/bin/uv"

echo ""
echo "  Installing the Off-Axis Robot..."
echo ""

if [ "$(uname)" != "Darwin" ]; then
  echo "This installer is for macOS only."; exit 1
fi

# 1) uv: a small tool that downloads its own Python, so no Homebrew is needed
if [ ! -x "$UV" ] && ! command -v uv >/dev/null 2>&1; then
  echo "-> Installing uv (Python manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -x "$UV" ] || UV="$(command -v uv)"

# 2) A private Python 3.11 with the exact package versions the robot needs
echo "-> Setting up Python 3.11 and packages (about 250 MB to download, a few minutes)..."
mkdir -p "$APP/voices"
rm -rf "$APP/.venv"
"$UV" venv --python 3.11 "$APP/.venv"
"$UV" pip install --python "$APP/.venv/bin/python" \
  "mediapipe==0.10.21" "opencv-contrib-python<4.11" "numpy<2" pygame sounddevice

# 3) The robot itself
echo "-> Writing the robot..."
cat > "$APP/robot.py" <<'@MARK@'
"""

INSTALLER_VOICES = r"""
# 3b) Recorded voice lines
"$APP/.venv/bin/python" - <<'@VMARK@'
import base64, os
d = os.path.expanduser("~/.offaxis-robot/voices")
os.makedirs(d, exist_ok=True)
FILES = {
@FILES@}
for name, b64 in FILES.items():
    with open(os.path.join(d, name), "wb") as f:
        f.write(base64.b64decode(b64))
print("-> Added", len(FILES), "voice line(s).")
@VMARK@
"""

INSTALLER_TAIL = r"""
# 4) The "robot" command, plus a double-click icon on the Desktop
cat > "$APP/run.sh" <<'RUN_END'
#!/bin/bash
cd "$HOME/.offaxis-robot" && exec ./.venv/bin/python robot.py "$@"
RUN_END
chmod +x "$APP/run.sh"

touch "$HOME/.zshrc"
if ! grep -q "offaxis-robot/run.sh" "$HOME/.zshrc"; then
  printf '\n# Off-Axis Robot\nalias robot="$HOME/.offaxis-robot/run.sh"\n' >> "$HOME/.zshrc"
fi

mkdir -p "$HOME/Desktop"
cat > "$HOME/Desktop/Robot.command" <<'CMD_END'
#!/bin/bash
"$HOME/.offaxis-robot/run.sh"
CMD_END
chmod +x "$HOME/Desktop/Robot.command"

# 5) Quick self-check
"$APP/.venv/bin/python" -c "import mediapipe, cv2, pygame, numpy" >/dev/null 2>&1 \
  && echo "-> Self-check passed." \
  || { echo "!! Something went wrong installing the packages. Run this installer again."; exit 1; }

echo ""
echo "  All done!"
echo "  Open a NEW Terminal window and type:   robot"
echo "  (or double-click 'Robot' on the Desktop)"
echo "  The first time, macOS will ask for camera access - click Allow."
echo "  Press Esc to close the robot."
echo ""
"""


def make_installer(out_path=None):
    """Write a one-file installer containing this robot and any recorded voice lines."""
    out_path = out_path or os.path.join(os.path.expanduser("~/Desktop"), "install_robot.sh")
    mark, vmark = "ROBOT_PY_" + "END", "VOICES_" + "END"
    src = open(os.path.abspath(__file__)).read()
    parts = [INSTALLER_HEAD.replace("@MARK@", mark), src.rstrip("\n") + "\n", mark + "\n"]
    files = ""
    if os.path.isdir(VOICE_DIR):
        for key, _ in VOICE_LINES:
            p = os.path.join(VOICE_DIR, key + ".wav")
            if os.path.exists(p):
                files += f'    "{key}.wav": "{base64.b64encode(open(p, "rb").read()).decode()}",\n'
    if files:
        parts.append(INSTALLER_VOICES.replace("@VMARK@", vmark).replace("@FILES@", files))
    parts.append(INSTALLER_TAIL)
    with open(out_path, "w") as f:
        f.write("".join(parts))
    n = files.count(".wav")
    print(f"\n  Made {out_path}  (with {n} voice line{'s' if n != 1 else ''}).")
    print(f"  Anyone with a Mac can now run:   bash {os.path.basename(out_path)}\n")


# ======================= reactions =======================
class Reactor:
    """Turns what you're doing into robot reactions."""
    COOLDOWN = {"yay": 2.5, "hi": 12.0, "thumbs": 3.0, "peace": 3.0, "yawn": 6.0, "love": 6.0, "kiss": 6.0}
    HOLD = {"yawn": 0.8, "love": 0.3, "kiss": 0.45}

    def __init__(self, fx, audio):
        self.fx, self.audio = fx, audio
        self.hold, self.next_ok = {}, {}
        self.apart, self.claps = True, []
        self.timed, self.timed_until = None, 0.0
        self.cheer = self.together = False
        self.base, self.face_frames = {}, 0
        self.face = None                    # smoothed, mirrored face state for the robot
        self.closed_since, self.next_wake = None, 0.0
        self.last_face_t, self.next_greet = -99.0, 0.0
        self.nods, self.turns, self.next_head_ok = [], [], 0.0
        self.anim, self.anim_t0 = None, 0.0
        self.active = []
        self.airborne = False

    # ---------- helpers ----------
    def _held(self, name, cond, t):
        if not cond:
            self.hold.pop(name, None)
            return False
        start = self.hold.setdefault(name, t)
        if t - start >= self.HOLD.get(name, HOLD_TIME) and t >= self.next_ok.get(name, 0.0):
            self.hold[name] = float("inf")          # fire once per hold
            self.next_ok[name] = t + self.COOLDOWN[name]
            return True
        return False

    def _timed_expr(self, name, t, dur):
        self.timed, self.timed_until = name, t + dur

    def _line(self, key, text, t, beep):
        """Say a voice line in your voice if recorded, otherwise beep. Always show the bubble."""
        if not self.audio.say(key, t):
            self.audio.beep(beep)
        self.fx.say(text, t)

    def greet(self, t, robot):
        if t < self.next_greet:
            return
        self.next_greet = t + 12.0
        self._line("hey_beautiful", "HEY BEAUTIFUL!", t, "hi")
        self.fx.sparkles(add(robot.head_c, (0, -0.6, -0.5)))
        self._timed_expr("happy", t, 2.0)

    def love(self, t, robot, blown=False):
        self.next_ok["love"] = self.next_ok["kiss"] = t + self.COOLDOWN["love"]
        self._line("i_love_you", "I LOVE YOU!", t, "love")
        if blown:
            self.audio.beep("kiss")
            self.fx.kiss(robot.mouth)
        self.fx.hearts(add(robot.head_c, (0, -0.3, -0.6)))
        self._timed_expr("heart", t, 3.0)

    def _face_state(self, m):
        """Compare your face with your own neutral face (learned in the first ~1.5 s, then slowly)."""
        self.face_frames += 1
        r = 0.12 if self.face_frames < 45 else 0.004
        B = self.base
        clamp = lambda v, a, b: max(a, min(b, v))

        def base(k, v, update=True):
            if k not in B:
                B[k] = v
            elif update:
                B[k] += (v - B[k]) * r
            return B[k]

        opens = []
        for k in ("ear_l", "ear_r"):
            v = m[k]
            b = base(k, v, v > 0.7 * B.get(k, v))   # learn "open" only from open eyes
            opens.append(clamp((v / b - BLINK_CLOSED) / (0.9 - BLINK_CLOSED), 0.0, 1.0))
        raises = []
        for k in ("brow_l", "brow_r"):
            b = base(k, m[k])
            raises.append(clamp((m[k] - b) / (0.25 * b), -1.0, 1.0))
        bi = base("inner", m["inner"])
        bw = base("width", m["width"])
        bl = base("lift", m["lift"])
        gaze = (0.0, 0.0)
        if m["gaze"]:
            gx = m["gaze"][0] - base("gx", m["gaze"][0])
            gy = m["gaze"][1] - base("gy", m["gaze"][1])
            gaze = (clamp(gx * GAZE_GAIN, -1, 1), clamp(gy * GAZE_GAIN * 1.5, -1, 1))
        return {"open_l": opens[0], "open_r": opens[1], "raise_l": raises[0], "raise_r": raises[1],
                "furrow": clamp((bi - m["inner"]) / (0.08 * bi), 0.0, 1.0),
                "smile": clamp(0.5 * max(0.0, m["width"] - bw) / 0.05 + 0.5 * (m["lift"] - bl) / 0.025, -1.0, 1.0),
                "mouth_open": clamp((m["gap"] - 0.02) / 0.22, 0.0, 1.0),
                "roll": clamp(m["roll"], -0.5, 0.5), "gaze": gaze,
                "pucker": m["width"] < bw - PUCKER_DROP and m["gap"] < 0.06}

    def _smooth_face(self, new):
        if self.face is None:
            self.face = new
            return
        out = {}
        for k, v in new.items():
            old = self.face[k]
            if k == "pucker":
                out[k] = v
            elif k == "gaze":
                out[k] = (old[0] + (v[0] - old[0]) * FACE_SMOOTH, old[1] + (v[1] - old[1]) * FACE_SMOOTH)
            else:
                a = 0.9 if k.startswith("open") else FACE_SMOOTH   # eyes react fastest so blinks show
                out[k] = old + (v - old) * a
        self.face = out

    # ---------- main update ----------
    def update(self, body, t, robot):
        self.active = []
        if body is None:
            self.cheer = self.together = False
            return
        hands_mid = mul(add(robot.hands[-1], robot.hands[1]), 0.5)
        m = body["face"]

        # --- face mirroring ---
        if m:
            if t - self.last_face_t > 4.0:          # you just appeared (or came back)
                self.greet(t, robot)
            self.last_face_t = t
            self._smooth_face(self._face_state(m))
            f = self.face
            # eyes shut for a while -> WAKE UP! (repeats until you open them)
            shut = f["open_l"] < 0.25 and f["open_r"] < 0.25
            if shut:
                self.closed_since = self.closed_since or t
                if t - self.closed_since > SLEEP_TIME and t >= self.next_wake:
                    self.next_wake = t + 5.0
                    self._line("wake_up", "WAKE UP!", t, "alarm")
                    self._timed_expr("surprised", t, 1.5)
                    self.anim, self.anim_t0 = "shake", t
            else:
                self.closed_since = None
            # yawn and blown kiss
            if self._held("yawn", m["gap"] > YAWN_GAP, t):
                self.audio.beep("yawn")
                self.fx.say("*YAAAWN*", t)
                self._timed_expr("yawn", t, 2.2)
            if self._held("kiss", f["pucker"], t):
                self.love(t, robot, blown=True)
            # nod / shake: count quick back-and-forth head movements over 1.5 s
            self.nods = [(tt, v) for tt, v in self.nods if t - tt < 1.5] + [(t, m["nod"])]
            self.turns = [(tt, v) for tt, v in self.turns if t - tt < 1.5] + [(t, m["turn"])]
            n_nod = count_swings([v for _, v in self.nods], NOD_THR)
            n_turn = count_swings([v for _, v in self.turns], SHAKE_THR)
            if t >= self.next_head_ok and not shut:
                if n_nod >= 3 and n_turn <= 1:
                    self.audio.beep("yes")
                    self.fx.say("YES!", t)
                    self.anim, self.anim_t0 = "nod", t
                    self.next_head_ok, self.nods = t + 2.0, []
                elif n_turn >= 3 and n_nod <= 1:
                    self.audio.beep("no")
                    self.fx.say("NOPE!", t)
                    self.anim, self.anim_t0 = "shake", t
                    self.next_head_ok, self.turns = t + 2.0, []

        # --- clapping ---
        wr = body["wrist_ratio"]
        self.together = wr is not None and wr < CLAP_CLOSE
        if wr is not None:
            if self.apart and wr < CLAP_CLOSE:
                self.apart = False
                self.claps = [c for c in self.claps if t - c < 2.5] + [t]
                self.audio.beep("clap")
                self.fx.sparkles(hands_mid)
                if len(self.claps) >= 3:
                    self.audio.beep("yay")
                    self.fx.say("BRAVO!", t)
                    self.fx.confetti(add(robot.head_c, (0, -1.0, 0)))
                    self._timed_expr("happy", t, 2.0)
                    self.claps = []
                else:
                    self.fx.say("CLAP!", t)
            elif wr > CLAP_CLOSE * 1.7:
                self.apart = True

        # --- arm and hand gestures ---
        self.cheer = body["both_up"]
        g = body["gestures"]
        if self._held("yay", body["both_up"], t):
            self.audio.beep("yay")
            self.fx.confetti(add(robot.head_c, (0, -1.2, 0)))
            self.fx.say("YAY!", t)
        if body["one_up"] and not body["both_up"] and "love" not in g:
            self.greet(t, robot)                       # a wave
        if self._held("love", "love" in g, t):
            self.love(t, robot)
        if self._held("thumbs", "thumbs_up" in g, t):
            self.audio.beep("yes")
            self.fx.hearts(add(robot.head_c, (0, -0.3, -0.6)))
            self.fx.say("AWW, THANKS!", t)
            self._timed_expr("heart", t, 2.5)
        if self._held("peace", "peace" in g, t):
            self.audio.beep("spark")
            self.fx.stars(add(robot.head_c, (0, -0.2, -0.6)))
            self.fx.say("PEACE!", t)
            self._timed_expr("wink", t, 2.0)

        # --- jumping ---
        if robot.jump > 0.12 and not self.airborne:
            self.airborne = True
            self.audio.beep("boing")
        elif robot.jump < 0.04:
            self.airborne = False

        # --- what's detected (for the C debug view) ---
        self.active = sorted(g)
        if body.get("full"):
            self.active.append("full body")
        if self.airborne:
            self.active.append("jump!")
        if m and self.face:
            f = self.face
            self.active += [n for n, on in (
                ("blink L", f["open_l"] < 0.3), ("blink R", f["open_r"] < 0.3),
                ("smile", f["smile"] > 0.4), ("frown", f["smile"] < -0.4),
                ("brows up", (f["raise_l"] + f["raise_r"]) / 2 > 0.4), ("furrow", f["furrow"] > 0.4),
                ("mouth open", f["mouth_open"] > 0.3), ("pucker", f["pucker"])) if on]
        self.active += [n for n, on in (("hands up", body["both_up"]), ("hand up", body["one_up"]),
                                        ("hands together", self.together)) if on]

    def mode(self, t):
        if t < self.timed_until:
            return self.timed
        if self.cheer:
            return "happy"
        return "normal"

    def head_offsets(self, t):
        """Extra (yaw, pitch) on top of looking at you: nodding, shaking, yawning."""
        if self.timed == "yawn" and t < self.timed_until:
            return 0.0, -0.25
        if self.anim:
            age = t - self.anim_t0
            if age > 1.2:
                self.anim = None
            else:
                w = 0.3 * math.sin(age * 2 * math.pi * 2.2) * (1 - age / 1.2)
                return (w * 1.2, 0.0) if self.anim == "shake" else (0.0, w)
        return 0.0, 0.0

    def numbers(self):
        f = self.face
        if not f:
            return ""
        return (f"eyes L {f['open_l']:.2f} R {f['open_r']:.2f}  smile {f['smile']:+.2f}  "
                f"brows {(f['raise_l'] + f['raise_r']) / 2:+.2f}  furrow {f['furrow']:.2f}  "
                f"mouth {f['mouth_open']:.2f}  gaze {f['gaze'][0]:+.2f},{f['gaze'][1]:+.2f}")


# ======================= main =======================
def draw_debug(screen, frame, res, reactor, small_font):
    img = frame.copy()
    du, H = mp.solutions.drawing_utils, mp.solutions.holistic
    if res is not None:
        du.draw_landmarks(img, res.pose_landmarks, H.POSE_CONNECTIONS)
        du.draw_landmarks(img, res.left_hand_landmarks, H.HAND_CONNECTIONS)
        du.draw_landmarks(img, res.right_hand_landmarks, H.HAND_CONNECTIONS)
        if res.face_landmarks:
            for i in (33, 133, 159, 145, 362, 263, 386, 374, 61, 291, 13, 14, 105, 334, 468, 473):
                if i < len(res.face_landmarks.landmark):
                    l = res.face_landmarks.landmark[i]
                    cv2.circle(img, (int(l.x * img.shape[1]), int(l.y * img.shape[0])), 2, (0, 255, 255), -1)
    img = cv2.cvtColor(cv2.resize(img, (256, 192)), cv2.COLOR_BGR2RGB)
    screen.blit(pygame.image.frombuffer(img.tobytes(), (256, 192), "RGB"), (10, 10))
    pygame.draw.rect(screen, (255, 255, 255), (10, 10, 256, 192), 1)
    for i, line in enumerate((", ".join(reactor.active) or "(nothing detected)", reactor.numbers())):
        if line:
            screen.blit(small_font.render(line, True, (255, 255, 255), (0, 0, 0)), (10, 208 + 20 * i))


def render(screen, scene, robot, fx, sx, sy, lx, ly, t, mode, face, together, head_off, speak=None):
    scene.draw_back(screen, sx, sy, t)
    scene.draw_grass(screen, sx, sy, t, behind=True, split_z=robot.root[2])
    faces = robot.build((lx, ly, -EYE_DIST), t, mode, face, together, head_off, speak)
    draw_shadow(screen, sx, sy, robot)
    draw_faces(screen, faces, (sx, sy, -EYE_DIST), sx, sy)
    scene.draw_grass(screen, sx, sy, t, behind=False, split_z=robot.root[2])
    fx.draw(screen, sx, sy, t, robot.head_top)


def main():
    if "--record" in sys.argv:
        record_voices()
        return
    if "--make-installer" in sys.argv:
        make_installer()
        return

    pygame.mixer.pre_init(22050, -16, 1, 512)
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Off-Axis Robot  -  Esc: quit   C: camera view")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arialroundedmtbold,avenirnext,helvetica,arial", 34, bold=True)
    small_font = pygame.font.SysFont("helvetica,arial", 16)

    def open_camera():
        c = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        return c if c.isOpened() else None

    # First run: macOS asks for camera permission. Wait (and keep retrying) instead of failing.
    cap, waited = open_camera(), 0
    while cap is None:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
        screen.fill((15, 20, 32))
        if waited < 20:
            lines = ["Waiting for the camera...", "If a popup asks, click  Allow  so the robot can see you."]
        else:
            lines = ["The camera still isn't available.",
                     "System Settings > Privacy & Security > Camera: switch Terminal ON,",
                     "then quit Terminal (Cmd+Q), reopen it and type  robot  again."]
        for i, line in enumerate(lines):
            img = (font if i == 0 else small_font).render(line, True, (230, 236, 245))
            screen.blit(img, (CX - img.get_width() / 2, CY - 40 + i * 36))
        pygame.display.flip()
        pygame.time.wait(1000)
        waited += 1
        cap = open_camera()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)            # always use the newest frame (less lag)
    holistic = mp.solutions.holistic.Holistic(model_complexity=1, smooth_landmarks=True,
                                              refine_face_landmarks=True)

    scene, robot = Scene(), Robot()
    fx = FX(font)
    audio = Audio()
    reactor = Reactor(fx, audio)
    sx = sy = lx = ly = 0.0
    track_state = {}
    show_cam = False
    running = True
    while running:
        dt = min(clock.tick(60) / 1000.0, 0.1)
        t = pygame.time.get_ticks() / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_c:
                show_cam = not show_cam

        ok, frame = cap.read()
        body, res = None, None
        if ok:
            frame = cv2.flip(frame, 1)
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            body = analyze(res, frame.shape[1], frame.shape[0], track_state)
            if body["head"]:
                hx, hy = body["head"]
                sx += (hx * HEAD_GAIN - sx) * SMOOTH
                sy += (hy * HEAD_GAIN - sy) * SMOOTH
                sx = max(-HX_LIM, min(HX_LIM, sx))
                sy = max(HY_LIM[0], min(HY_LIM[1], sy))
        lx += (sx - lx) * LOOK_SMOOTH
        ly += (sy - ly) * LOOK_SMOOTH

        robot.update(body, t)
        reactor.update(body, t, robot)
        fx.update(dt)
        face = reactor.face if (body and body["face"]) else None
        render(screen, scene, robot, fx, sx, sy, lx, ly, t, reactor.mode(t), face, reactor.together,
               reactor.head_offsets(t), audio.mouth(t))
        if show_cam and ok:
            draw_debug(screen, frame, res, reactor, small_font)
        pygame.display.flip()

    cap.release()
    holistic.close()
    pygame.quit()


if __name__ == "__main__":
    main()
