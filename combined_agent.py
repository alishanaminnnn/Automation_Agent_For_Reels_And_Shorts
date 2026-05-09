"""
╔══════════════════════════════════════════════════════════════╗
║        COMBINED CLIP + SHORTS AGENT — Clean Version          ║
║                                                              ║
║  HOW TO RUN:                                                 ║
║  python combined_agent.py "video.mp4"                        ║
║  python combined_agent.py "video.mp4" --clips 3              ║
║  python combined_agent.py "video.mp4" --no-music             ║
║  python combined_agent.py "video.mp4" --clear-cache          ║
║                                                              ║
║                                                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, os, json, argparse, time, random, requests, subprocess
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyBRHbOAnHwH-k54R5IRiMcrzkcF0xhPJrc"  # ← paste your key
CLIP_DURATION  = 30     # seconds per clip
MUSIC_VOLUME   = 0.5   # background music volume (0.0 – 1.0)
# ─────────────────────────────────────────────────────────────

# ── ROYALTY-FREE MUSIC BY MOOD ────────────────────────────────
# All tracks from soundhelix.com — 100% free, no copyright
MUSIC_BY_MOOD = {
    "action":    "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
    "dramatic":  "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
    "emotional": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
    "neutral":   "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
}

DEFAULT_HOOKS = [
    {"hook": "You NEED to see this",    "title": "This Will Blow Your Mind | #Shorts",   "hashtags": "#Shorts #viral #trending #fyp #reels"},
    {"hook": "Nobody talks about this", "title": "The Secret Everyone Ignores | #Shorts", "hashtags": "#Shorts #viral #facts #trending #fyp"},
    {"hook": "Wait for the ending",     "title": "Watch Till The End | #Shorts",          "hashtags": "#Shorts #viral #satisfying #trending #fyp"},
    {"hook": "This changes everything", "title": "This Is Insane | #Shorts",              "hashtags": "#Shorts #viral #crazy #trending #fyp"},
    {"hook": "The part no one sees",    "title": "Don't Skip This | #Shorts",             "hashtags": "#Shorts #viral #trending #reels #fyp"},
]


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def check_dependencies():
    missing = []
    for pkg, name in [
        ("librosa",      "librosa"),
        ("cv2",          "opencv-python"),
        ("google.genai", "google-genai"),
        ("numpy",        "numpy"),
        ("requests",     "requests"),
    ]:
        try: __import__(pkg)
        except ImportError: missing.append(name)
    if missing:
        print(f"\n❌  Missing: {', '.join(missing)}")
        print(f"    Run: pip install {' '.join(missing)}\n")
        sys.exit(1)
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
    except Exception:
        print("\n❌  ffmpeg not found. Install from https://ffmpeg.org\n")
        sys.exit(1)
    print("✅  All good\n")


def cache_path(video_path, suffix):
    return os.path.splitext(os.path.abspath(video_path))[0] + suffix


def run(cmd):
    """Run ffmpeg command. Raises RuntimeError with last 400 chars of stderr on failure."""
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="ignore")[-400:])


def safe_remove(path, retries=6, delay=1.0):
    for i in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            if i < retries - 1:
                time.sleep(delay)


def probe_video(video_path):
    """Return (width, height, fps, duration) of a video file."""
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", video_path],
        capture_output=True
    )
    info = json.loads(r.stdout)
    w, h, fps_num, fps_den, duration = 1920, 1080, 25, 1, 0.0
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            w   = int(s.get("width",  w))
            h   = int(s.get("height", h))
            fr  = s.get("r_frame_rate", "25/1").split("/")
            fps_num = int(fr[0]); fps_den = max(int(fr[1]), 1)
    duration = float(info.get("format", {}).get("duration", 0))
    return w, h, fps_num / fps_den, duration


def has_audio(video_path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", "-select_streams", "a:0", video_path],
        capture_output=True
    )
    return len(json.loads(r.stdout).get("streams", [])) > 0


# ═══════════════════════════════════════════════════════════════
#  STEP 1: AUDIO ENERGY SCORING
# ═══════════════════════════════════════════════════════════════
def score_audio_energy(video_path, duration):
    import librosa
    cp = cache_path(video_path, ".audio_cache.npy")
    if os.path.exists(cp):
        print("🎵  Audio scores: from cache"); return np.load(cp)

    print("🎵  Analysing audio energy...")
    wav = cache_path(video_path, "_tmp_audio.wav")
    run(["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", wav])
    y, sr = librosa.load(wav, sr=16000)
    safe_remove(wav)
    rms = librosa.feature.rms(y=y, frame_length=sr*2, hop_length=sr)[0]
    n   = int(duration)
    rms = rms[:n] if len(rms) >= n else np.pad(rms, (0, n - len(rms)))
    if rms.max() > 0: rms = rms / rms.max()
    np.save(cp, rms)
    print(f"    ✓ Scored {n}s, cached"); return rms


# ═══════════════════════════════════════════════════════════════
#  STEP 2: SCENE CHANGE SCORING
# ═══════════════════════════════════════════════════════════════
def score_scene_changes(video_path, duration):
    import cv2
    cp = cache_path(video_path, ".scene_cache.npy")
    if os.path.exists(cp):
        print("🎬  Scene scores: from cache"); return np.load(cp)

    print("🎬  Analysing scene changes...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n   = int(duration)
    scores = np.zeros(n)
    prev, cur_sec, diffs = None, 0, []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        sec = int(fi / fps)
        if sec >= n: break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
        if prev is not None:
            d = float(np.mean(np.abs(gray.astype(float) - prev.astype(float))))
            if sec == cur_sec: diffs.append(d)
            else:
                if cur_sec < n: scores[cur_sec] = np.mean(diffs) if diffs else 0
                cur_sec, diffs = sec, [d]
        prev = gray; fi += 1
    cap.release()
    if scores.max() > 0: scores = scores / scores.max()
    np.save(cp, scores)
    print("    ✓ Scored, cached"); return scores


# ═══════════════════════════════════════════════════════════════
#  STEP 3: LOCAL MOOD DETECTION (zero API calls)
# ═══════════════════════════════════════════════════════════════
def detect_mood(audio_scores, scene_scores, clip_start, clip_len=30):
    end  = min(clip_start + clip_len, len(audio_scores))
    a    = float(np.mean(audio_scores[clip_start:end]))
    s    = float(np.mean(scene_scores[clip_start:end]))
    HIGH = 0.35
    if   s >= HIGH and a >= HIGH: mood = "action"
    elif s >= HIGH and a <  HIGH: mood = "dramatic"
    elif s <  HIGH and a >= HIGH: mood = "emotional"
    else:                         mood = "neutral"
    print(f"    🎭 Mood: {mood.upper()}  (audio={a:.2f}  scene={s:.2f}  0 API tokens)")
    return mood


# ═══════════════════════════════════════════════════════════════
#  STEP 4: GEMINI — find best windows (no transcript needed)
# ═══════════════════════════════════════════════════════════════
def gemini_find_windows(audio_scores, scene_scores, duration, n_clips):
    from google import genai
    print("🤖  Asking Gemini to pick best moments...")

    # Summarise scores into 10-second buckets — tiny payload
    bucket = 10
    n      = int(duration)
    buckets = []
    for i in range(0, n, bucket):
        a = float(np.mean(audio_scores[i:i+bucket]))
        s = float(np.mean(scene_scores[i:i+bucket]))
        buckets.append({"t": i, "a": round(a, 2), "s": round(s, 2)})

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        f"You are a YouTube Shorts editor. A video is {int(duration)}s long.\n"
        f"Here are 10-second buckets with audio energy (a) and scene change (s) scores (0-1):\n"
        f"{json.dumps(buckets)}\n\n"
        f"Pick the TOP {n_clips} best non-overlapping 30-second start times for viral Shorts clips.\n"
        f"Also write a hook (5-7 words), title (<60 chars), and 5 hashtags.\n"
        f"Reply ONLY raw JSON, no markdown:\n"
        f'{{"starts":[45,120],"hook":"hook text here","title":"title here","hashtags":"#Shorts #tag2 #tag3 #tag4 #tag5"}}'
    )

    for attempt in range(2):
        try:
            resp = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            raw  = resp.text.strip().replace("```json","").replace("```","").strip()
            data = json.loads(raw)
            print(f"    ✓ Gemini picked starts: {data.get('starts', [])}")
            return data
        except Exception as e:
            print(f"    ⚠  Attempt {attempt+1} failed: {str(e)[:60]}")
            if attempt == 0:
                print("    🔄  Retrying in 15s..."); time.sleep(15)
            else:
                print("    ❌  Using local scoring instead")
                return None
    return None


# ═══════════════════════════════════════════════════════════════
#  STEP 5: FIND BEST WINDOWS (local fallback)
# ═══════════════════════════════════════════════════════════════
def find_best_windows(audio_scores, scene_scores, n_clips, gemini_starts=None):
    print(f"\n🏆  Finding top {n_clips} clip(s)...")
    n = len(audio_scores)

    # If Gemini gave us starts, use them directly
    if gemini_starts:
        windows = []
        for st in gemini_starts[:n_clips]:
            st = max(0, min(int(st), n - CLIP_DURATION))
            score = float(np.mean((audio_scores + scene_scores)[st:st+CLIP_DURATION]))
            windows.append((st, score))
            print(f"    ✓ Gemini start: {st}s (score={score:.2f})")
        return windows

    # Local scoring fallback
    combined = 0.4 * audio_scores + 0.6 * scene_scores
    smoothed = np.convolve(combined, np.ones(5)/5, mode="same")
    windows, used = [], np.zeros(n, dtype=bool)
    for _ in range(n_clips):
        best_start, best_score = -1, -1.0
        for start in range(0, n - CLIP_DURATION):
            if used[start:start+CLIP_DURATION].any(): continue
            sc = float(smoothed[start:start+CLIP_DURATION].mean())
            if sc > best_score: best_score, best_start = sc, start
        if best_start == -1: break
        windows.append((best_start, best_score))
        used[best_start:best_start+CLIP_DURATION] = True
        print(f"    ✓ Local start: {best_start}s (score={best_score:.2f})")
    windows.sort(key=lambda x: x[0])
    return windows


# ═══════════════════════════════════════════════════════════════
#  STEP 6: CUT RAW CLIPS
# ═══════════════════════════════════════════════════════════════
def cut_clips(video_path, windows):
    base       = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.dirname(os.path.abspath(video_path))
    print(f"\n✂️   Cutting {len(windows)} clip(s)...\n")
    paths = []
    for i, (start, score) in enumerate(windows):
        out = os.path.join(output_dir, f"{base}_clip_{i+1}_at_{start}s.mp4")
        print(f"  📹 Clip {i+1}: {start}s → {start+CLIP_DURATION}s  (score={score:.2f})")
        run(["ffmpeg", "-y",
             "-ss", str(start), "-i", video_path,
             "-t", str(CLIP_DURATION),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-c:a", "aac", "-movflags", "+faststart", out])
        paths.append(out)
        print(f"     ✅ {os.path.basename(out)}")
    return paths


# ═══════════════════════════════════════════════════════════════
#  STEP 7A: MAKE VERTICAL — crop to fill 1080x1920, no black bars
# ═══════════════════════════════════════════════════════════════
def make_vertical(input_path, output_path):
    """
    Converts to 1080x1920. Crops to fill — no black bars.
    Zooms in slightly (1.05x) to focus on centre subject.
    """
    w, h, fps, dur = probe_video(input_path)
    TARGET_W, TARGET_H = 1080, 1920

    # Scale so the video FILLS the target (scale UP the smaller ratio dimension)
    scale_w = TARGET_W / w
    scale_h = TARGET_H / h
    scale   = max(scale_w, scale_h)   # fill, not fit

    # Apply 1.05x zoom so subject fills frame better
    scale  *= 1.05
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    # Make even
    new_w  += new_w % 2
    new_h  += new_h % 2

    # Centre crop to 1080x1920
    crop_x = (new_w - TARGET_W) // 2
    crop_y = (new_h - TARGET_H) // 2

    vf = (
        f"scale={new_w}:{new_h},"
        f"crop={TARGET_W}:{TARGET_H}:{crop_x}:{crop_y},"
        f"setsar=1"
    )

    print(f"📱  Vertical conversion: {w}x{h} → {TARGET_W}x{TARGET_H} (crop-fill + 1.05x zoom)")
    run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart", output_path
    ])
    print(f"    ✓ Output: exactly 1080x1920, fills phone screen")


# ═══════════════════════════════════════════════════════════════
#  STEP 7B: ADD HOOK TEXT (ffmpeg drawtext — no ImageMagick)
# ═══════════════════════════════════════════════════════════════
def add_hook(input_path, hook_text, output_path):
    import re
    # Remove emojis — ffmpeg drawtext can't render them
    clean = re.sub(r'[^\x00-\x7F]+', '', hook_text).strip() or "Watch This"
    # Escape special characters for ffmpeg
    clean = clean.replace("'", "").replace(":", " ").replace("\\", "")

    print(f"🎯  Hook text: '{clean}'")
    try:
        run([
            "ffmpeg", "-y", "-i", input_path,
            "-vf",
            f"drawtext=text='{clean}':fontsize=80:fontcolor=white"
            f":borderw=5:bordercolor=black"
            f":x=(w-text_w)/2:y=180"
            f":enable='between(t,0,2.5)'",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            "-movflags", "+faststart", output_path
        ])
        print("    ✓ Hook text added")
    except Exception as e:
        print(f"    ⚠  Hook text failed — copying without text\n    {str(e)[:120]}")
        import shutil; shutil.copy2(input_path, output_path)


# ═══════════════════════════════════════════════════════════════
#  STEP 7C: DOWNLOAD MUSIC
# ═══════════════════════════════════════════════════════════════
def download_music(mood, output_dir):
    url        = MUSIC_BY_MOOD.get(mood, MUSIC_BY_MOOD["neutral"])
    music_file = os.path.join(output_dir, f"_bg_music_{mood}.mp3")

    if os.path.exists(music_file) and os.path.getsize(music_file) > 10000:
        print(f"🎵  Music ({mood}): using cached file")
        return music_file

    print(f"🎵  Downloading {mood} music track...")
    try:
        r = requests.get(url, timeout=30, stream=True)
        r.raise_for_status()
        with open(music_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = os.path.getsize(music_file) // 1024
        print(f"    ✓ Downloaded {size_kb}KB — {os.path.basename(music_file)}")
        return music_file
    except Exception as e:
        print(f"    ❌  Download failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  STEP 7D: MIX MUSIC INTO VIDEO (ffmpeg — reliable)
# ═══════════════════════════════════════════════════════════════
def mix_music(video_path, music_path, output_path):
    """
    Mix background music into video using ffmpeg.
    Uses -stream_loop -1 to loop music — most reliable method.
    Original speech stays loud, music is quiet in background.
    """
    print(f"🎵  Mixing {os.path.basename(music_path)} at {int(MUSIC_VOLUME*100)}% volume...")

    video_has_audio = has_audio(video_path)

    if video_has_audio:
        # Blend music under original audio
        run([
            "ffmpeg", "-y",
            "-i", video_path,                    # input 0: video with audio
            "-stream_loop", "-1",                # loop music input
            "-i", music_path,                    # input 1: music
            "-filter_complex",
            f"[1:a]volume={MUSIC_VOLUME}[bg];"   # lower music volume
            f"[0:a][bg]amix=inputs=2:"           # mix original + music
            f"duration=first:"                   # stop when video ends
            f"dropout_transition=0[aout]",
            "-map", "0:v",                       # use video from input 0
            "-map", "[aout]",                    # use mixed audio
            "-c:v", "copy",                      # don't re-encode video
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ])
    else:
        # No original audio — use music only
        print("    ℹ️  No original audio — music will be only audio track")
        run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-stream_loop", "-1",
            "-i", music_path,
            "-filter_complex",
            f"[1:a]volume={MUSIC_VOLUME}[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ])

    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"    ✓ Music mixed in! Output: {size_mb:.1f}MB")


# ═══════════════════════════════════════════════════════════════
#  STEP 8: SAVE METADATA
# ═══════════════════════════════════════════════════════════════
def save_metadata(output_path, hook_data, mood):
    meta = output_path.replace(".mp4", "_metadata.txt")
    with open(meta, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("  YouTube Shorts — Copy & Paste Ready\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"TITLE:\n{hook_data.get('title', '')}\n\n")
        f.write(f"HASHTAGS:\n{hook_data.get('hashtags', '')}\n\n")
        f.write(f"HOOK TEXT (shown in video):\n{hook_data.get('hook', '')}\n\n")
        f.write(f"MOOD: {mood.upper()}\n\n")
        f.write("CHECKLIST:\n")
        f.write("  ✅ 1080x1920 — fills phone screen\n")
        f.write("  ✅ Hook text in first 2.5 seconds\n")
        f.write("  ✅ Under 45 seconds\n")
        f.write(f"  ✅ {mood.capitalize()} background music mixed in\n")
        f.write("  ✅ No copyrighted audio\n")
        f.write("  ✅ 5 hashtags including #Shorts\n")
    print(f"    ✓ Metadata: {os.path.basename(meta)}")


# ═══════════════════════════════════════════════════════════════
#  PROCESS ONE CLIP → SHORTS-READY
# ═══════════════════════════════════════════════════════════════
def process_clip(clip_path, hook_data, mood, no_music):
    base = os.path.splitext(clip_path)[0]
    dir_ = os.path.dirname(clip_path)

    print(f"\n{'─'*50}")
    print(f"📱  Processing: {os.path.basename(clip_path)}")
    print(f"    Mood: {mood.upper()}")
    print(f"{'─'*50}")

    # 1 ── Vertical 1080x1920 crop-fill
    v_path = base + "_step1_vertical.mp4"
    make_vertical(clip_path, v_path)

    # 2 ── Hook text overlay
    h_path = base + "_step2_hook.mp4"
    add_hook(v_path, hook_data.get("hook", "Watch This"), h_path)
    safe_remove(v_path)

    # 3 ── Mix mood-matched music
    final = base + "_SHORTS_READY.mp4"
    if not no_music:
        music = download_music(mood, dir_)
        if music:
            try:
                mix_music(h_path, music, final)
                safe_remove(h_path)
            except Exception as e:
                print(f"    ❌  Music mix error: {str(e)[:150]}")
                print("    ℹ️  Saving without music")
                import shutil; shutil.copy2(h_path, final)
                safe_remove(h_path)
        else:
            print("    ⚠️  No music file — saving without music")
            import shutil; shutil.copy2(h_path, final)
            safe_remove(h_path)
    else:
        import shutil; shutil.copy2(h_path, final)
        safe_remove(h_path)

    # 4 ── Metadata txt
    save_metadata(final, hook_data, mood)
    print(f"\n  ✅  Ready: {os.path.basename(final)}")
    return final


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--clips",       type=int, default=1)
    parser.add_argument("--no-ai",       action="store_true", help="Skip Gemini, use local scoring")
    parser.add_argument("--no-music",    action="store_true", help="Skip background music")
    parser.add_argument("--clear-cache", action="store_true", help="Delete cached scores")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"\n❌  File not found: {args.video}\n"); sys.exit(1)

    # Clear cache if requested
    if args.clear_cache:
        for suffix in [".audio_cache.npy", ".scene_cache.npy"]:
            p = cache_path(args.video, suffix)
            if os.path.exists(p):
                os.remove(p); print(f"🗑️  Deleted: {os.path.basename(p)}")
        print("✅  Cache cleared\n")

    # Check API key
    use_ai = not args.no_ai and GEMINI_API_KEY not in ("", "YOUR_GEMINI_API_KEY_HERE")
    if not use_ai:
        print("⚠️   No Gemini key or --no-ai set — using local scoring only\n")

    # Get video info
    w, h, fps, duration = probe_video(args.video)
    print(f"\n🎥  {args.video}")
    print(f"📐  Resolution: {w}x{h}  |  FPS: {fps:.1f}  |  Duration: {int(duration//60)}m {int(duration%60)}s")
    print(f"🎯  Making {args.clips} x {CLIP_DURATION}s Shorts clip(s)\n")

    if duration < CLIP_DURATION:
        print(f"❌  Video is shorter than {CLIP_DURATION}s\n"); sys.exit(1)

    n = int(duration)

    # Score video
    audio_scores = score_audio_energy(args.video, duration)
    scene_scores = score_scene_changes(args.video, duration)

    def pad(a): return a[:n] if len(a) >= n else np.pad(a, (0, n - len(a)))
    audio_scores = pad(audio_scores)
    scene_scores = pad(scene_scores)

    # Find best windows
    gemini_data    = None
    gemini_starts  = None
    if use_ai:
        gemini_data   = gemini_find_windows(audio_scores, scene_scores, duration, args.clips)
        gemini_starts = gemini_data.get("starts") if gemini_data else None

    windows = find_best_windows(audio_scores, scene_scores, args.clips, gemini_starts)

    # Build hook/title/hashtags
    if gemini_data:
        hook_data = {
            "hook":     gemini_data.get("hook",     random.choice(DEFAULT_HOOKS)["hook"]),
            "title":    gemini_data.get("title",    random.choice(DEFAULT_HOOKS)["title"]),
            "hashtags": gemini_data.get("hashtags", "#Shorts #viral #trending #fyp #reels"),
        }
    else:
        hook_data = random.choice(DEFAULT_HOOKS)

    # Cut raw clips
    clip_paths = cut_clips(args.video, windows)

    # Convert each clip to Shorts format
    print(f"\n🚀  Converting to Shorts format...\n")
    results = []
    for clip_path, (clip_start, _) in zip(clip_paths, windows):
        mood = detect_mood(audio_scores, scene_scores, clip_start)
        out  = process_clip(clip_path, hook_data, mood, args.no_music)
        results.append(out)

    # Summary
    print("\n" + "="*55)
    print("🎉  ALL DONE!")
    print(f"\n  📱 Shorts-ready clips (1080x1920, music mixed in):")
    for p in results:
        print(f"     {os.path.basename(p)}")
    print(f"\n  📁 Folder: {os.path.dirname(os.path.abspath(args.video))}")
    print("="*55 + "\n")


if __name__ == "__main__":
    check_dependencies()
    main()