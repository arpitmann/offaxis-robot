#!/usr/bin/env python3
"""
Builds the phone / web version of the Off-Axis Robot.

  python3 make_robot_app.py                  -> builds ~/Desktop/RobotApp  (+ RobotApp.zip)
  python3 make_robot_app.py --test           -> builds it, then gives you a temporary https link to try on your phone
  python3 make_robot_app.py --deploy         -> builds it, then publishes it to Cloudflare Pages
  python3 make_robot_app.py --name Alex      -> the robot says "Welcome, Alex" when it opens
  python3 make_robot_app.py --out some/dir   -> build into another folder (used by the GitHub Pages workflow)

Your own sounds are optional. Put any of these in the my_sounds folder (wav, m4a, mp3, ...):
  welcome     said when the app opens            hey_beautiful   when it first sees you
  i_love_you  on a heart sign or a kiss          wake_up         when your eyes stay shut
  song        played when the guitar button is pressed
Missing ones are replaced by the phone's voice, beeps, or (for the song) strummed chords.
Only Python's standard library and curl are used.
"""
import base64, os, re, shutil, signal, socket, subprocess, sys, threading, time, zipfile
import http.server, functools

THREE = "0.186.1"
VISION = "1.1.0"
CDN = os.environ.get("ROBOT_APP_MIRROR", "https://cdn.jsdelivr.net")          # mirror only used for testing
MODELS = os.environ.get("ROBOT_APP_MODELS", "https://storage.googleapis.com/mediapipe-models")   # override only for testing
DOWNLOADS = {
    "lib/three.module.js": f"{CDN}/npm/three@{THREE}/build/three.module.js",
    "lib/three.core.js": f"{CDN}/npm/three@{THREE}/build/three.core.js",
    "lib/vision_bundle.mjs": f"{CDN}/npm/@mediapipe/tasks-vision@{VISION}/vision_bundle.mjs",
    "wasm/vision_wasm_internal.js": f"{CDN}/npm/@mediapipe/tasks-vision@{VISION}/wasm/vision_wasm_internal.js",
    "wasm/vision_wasm_internal.wasm": f"{CDN}/npm/@mediapipe/tasks-vision@{VISION}/wasm/vision_wasm_internal.wasm",
    "wasm/vision_wasm_nosimd_internal.js": f"{CDN}/npm/@mediapipe/tasks-vision@{VISION}/wasm/vision_wasm_nosimd_internal.js",
    "wasm/vision_wasm_nosimd_internal.wasm": f"{CDN}/npm/@mediapipe/tasks-vision@{VISION}/wasm/vision_wasm_nosimd_internal.wasm",
    "models/face_landmarker.task": f"{MODELS}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "models/pose_landmarker_full.task": f"{MODELS}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "models/hand_landmarker.task": f"{MODELS}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}
VOICE_KEYS = ("hey_beautiful", "i_love_you", "wake_up")
EXTRA_SOUNDS = ("welcome", "song")
SOUND_EXTS = (".wav", ".m4a", ".mp3", ".aac", ".caf", ".aiff", ".ogg")
HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv and sys.argv.index(flag) + 1 < len(sys.argv) else default


OUT = os.path.abspath(arg("--out", os.path.join(HOME, "Desktop", "RobotApp")))
NAME = arg("--name", "")
CACHE = os.path.join(HOME, ".offaxis-robot", "app-cache")       # downloads are kept, so rebuilding is instant
SOUND_DIRS = [os.path.join(HERE, "my_sounds"), os.environ.get("ROBOT_VOICES", os.path.join(HOME, ".offaxis-robot", "voices"))]

# The app itself (index.html, app.js, ...) lives in the web folder next to this script.
WEB = os.path.join(HERE, "web")
FILES = {}
for _root, _, _fs in os.walk(WEB):
    for _f in _fs:
        if not _f.startswith("."):
            _p = os.path.join(_root, _f)
            FILES[os.path.relpath(_p, WEB)] = base64.b64encode(open(_p, "rb").read()).decode()


def say(msg=""):
    print(msg, flush=True)


def download_all():
    os.makedirs(CACHE, exist_ok=True)
    for rel, url in DOWNLOADS.items():
        dest = os.path.join(CACHE, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        say(f"  downloading {rel} ...")
        tmp = dest + ".part"
        r = subprocess.run(["curl", "-fL", "--retry", "3", "--silent", "--show-error", "-o", tmp, url])
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) < 1000:
            raise SystemExit(f"\n!! Couldn't download {url}\n   Check the internet connection and run this again.")
        os.replace(tmp, dest)


def build():
    say("\n  Building the robot app...")
    download_all()
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    stamp = time.strftime("%Y%m%d%H%M%S")
    for rel, b64 in FILES.items():
        data = base64.b64decode(b64)
        if rel == "sw.js":     # new name each build, so phones pick up the new version
            data = data.replace(b"my-robot-v1", f"my-robot-{stamp}".encode())
        path = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    for rel in DOWNLOADS:
        os.makedirs(os.path.dirname(os.path.join(OUT, rel)), exist_ok=True)
        shutil.copy2(os.path.join(CACHE, rel), os.path.join(OUT, rel))
    found = add_sounds()
    zpath = OUT + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.join("RobotApp", os.path.relpath(full, OUT)))
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(OUT) for f in fs) / 1e6
    say(f"  Made {OUT}  ({size:.0f} MB) and {os.path.basename(OUT)}.zip")
    say(f"  Sounds included: {', '.join(found) or 'none'}")
    say(f"  Welcome message: {'Welcome, ' + NAME if NAME else 'Welcome!'}   (change with --name)")
    return found


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def to_wav(src, dst):
    """Phones play WAV everywhere, so recordings (m4a etc.) are converted with the Mac's own afconvert."""
    if src.lower().endswith(".wav"):
        shutil.copy2(src, dst)
        return dst
    for cmd in (["afconvert", "-f", "WAVE", "-d", "LEI16@32000", "-c", "1", src, dst],
                ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "32000", dst]):
        if shutil.which(cmd[0]) and subprocess.run(cmd).returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return dst
    raw = os.path.splitext(dst)[0] + os.path.splitext(src)[1].lower()     # last resort: the original file
    shutil.copy2(src, raw)
    return raw


def find_sound(key):
    for folder in SOUND_DIRS:
        for ext in SOUND_EXTS:
            for name in (key + ext, key.capitalize() + ext, key.upper() + ext):
                p = os.path.join(folder, name)
                if os.path.isfile(p) and os.path.getsize(p) > 1000:
                    return p
    return None


def add_sounds():
    """Copies your sounds into the app and writes sounds.json (the list the app reads) and config.json."""
    import json
    table, found = {}, []
    for key in VOICE_KEYS + EXTRA_SOUNDS:
        src = find_sound(key)
        if src:
            os.makedirs(os.path.join(OUT, "sounds"), exist_ok=True)
            out = to_wav(src, os.path.join(OUT, "sounds", key + ".wav"))
            table[key] = os.path.relpath(out, OUT)
            found.append(f"{key} ({os.path.basename(src)})")
    with open(os.path.join(OUT, "sounds.json"), "w") as f:
        json.dump(table, f, indent=1)
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump({"name": NAME}, f)
    return found


def get_cloudflared():
    """Finds Cloudflare's tunnel tool, or downloads it once (no Homebrew needed)."""
    for p in (shutil.which("cloudflared"), os.path.join(HOME, ".offaxis-robot", "bin", "cloudflared"),
              os.path.join(HOME, "cloudflared"), os.path.join(HOME, "Downloads", "cloudflared"), os.path.abspath("cloudflared")):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    if sys.platform != "darwin":
        return None
    arch = "arm64" if os.uname().machine == "arm64" else "amd64"
    d = os.path.join(HOME, ".offaxis-robot", "bin")
    os.makedirs(d, exist_ok=True)
    say("  Downloading Cloudflare's tunnel tool (one time only)...")
    tgz = os.path.join(d, "cf.tgz")
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-{arch}.tgz"
    if subprocess.run(["curl", "-fL", "--silent", "--show-error", "-o", tgz, url]).returncode != 0:
        return None
    subprocess.run(["tar", "-xzf", tgz, "-C", d])
    os.remove(tgz)
    p = os.path.join(d, "cloudflared")
    if os.path.isfile(p):
        os.chmod(p, 0o755)
        subprocess.run(["xattr", "-d", "com.apple.quarantine", p], stderr=subprocess.DEVNULL)
        return p
    return None


def serve_and_tunnel(port=8080):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm",
                          ".mjs": "text/javascript", ".js": "text/javascript", ".webmanifest": "application/manifest+json",
                          ".wav": "audio/wav", ".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".json": "application/json"}

        def log_message(self, *a):
            pass
    handler = functools.partial(Quiet, directory=OUT)
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cf = get_cloudflared()
    if not cf:
        say("\n  Couldn't get Cloudflare's tunnel tool (needed for a phone link). Check the internet and run this again.")
        say(f"\n  (The app is also on your Wi-Fi at http://{lan_ip()}:{port} but phones only allow the")
        say("   camera on https links, so use the tunnel.)  Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return
    say("\n  Opening a temporary secure link (takes a few seconds)...")
    proc = subprocess.Popen([cf, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = None
    t0 = time.time()
    for line in proc.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            break
        if time.time() - t0 > 60:
            break
    if not url:
        proc.terminate()
        raise SystemExit("!! The tunnel didn't start. Try again in a minute.")
    say("\n  ======================================================")
    say(f"   Open this on the Android phone (Chrome):\n\n      {url}\n")
    say("   It works from anywhere while this window stays open.")
    say("   Press Ctrl+C here to stop.")
    say("  ======================================================\n")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)


def deploy(project="my-robot"):
    npx = shutil.which("npx")
    if not npx:
        say("\n  To publish it permanently (free):")
        say("   1. Go to dash.cloudflare.com  ->  Workers & Pages  ->  Create  ->  Pages  ->  Upload assets")
        say(f"   2. Name the project (e.g. {project}) and drag the RobotApp folder from your Desktop in.")
        say("   3. You get a link like https://my-robot.pages.dev  - share that link.")
        say("  (Or install Node.js with  brew install node  and run this again with --deploy.)")
        return
    say("\n  Publishing to Cloudflare Pages (a browser window may ask you to log in the first time)...")
    subprocess.run([npx, "--yes", "wrangler", "pages", "deploy", OUT, "--project-name", project, "--branch", "main"])


if __name__ == "__main__":
    build()
    if "--test" in sys.argv:
        serve_and_tunnel()
    elif "--deploy" in sys.argv:
        deploy()
    else:
        say("\n  Next:")
        say("   * Try it on a phone right now:   python3 make_robot_app.py --test")
        say("   * Publish it permanently:        python3 make_robot_app.py --deploy")
        say("  On the phone: open the link in Chrome, tap Start, allow the camera,")
        say("  then Chrome menu (three dots) -> Add to Home screen.\n")
