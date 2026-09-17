# vidvoice

[![tests](https://github.com/codingstark-dev/vidvoice/actions/workflows/tests.yml/badge.svg)](https://github.com/codingstark-dev/vidvoice/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Turn a **silent screen recording** into a **narrated video**.

You record your screen with [cap.so](https://cap.so) (which already handles cursor
zoom and pan). You export an MP4. vidvoice sends that MP4 to Gemini, which watches
it and produces a script with timestamps — either *writing* a voice-over that
describes what happens on screen, or *transcribing* speech that is already there.
Each line is spoken by [Cartesia](https://cartesia.ai) in a voice and emotion you
pick, then merged back onto the original video at the exact moments it belongs.

```
cap.so export ──► Gemini ──► timed script ──► Cartesia ──► voice clips
                                                                  │
                        original video ───────────────────────────┤
                                                                  ▼
                                                          ffmpeg: place, mix, mux
                                                                  │
                                    narrated MP4 + SRT + VTT + TXT ◄┘
```

The video stream is **copied, never re-encoded** (unless the narration outlasts the
picture, in which case the last frame is frozen). Your zoom-in/zoom-out work from
cap.so survives untouched.

---

## Install

Requires **Python 3.10+** and **ffmpeg** on your PATH.

```bash
brew install ffmpeg        # macOS
# sudo apt install ffmpeg  # Debian/Ubuntu
```

### Make the command available everywhere

```bash
cd vidvoice
uv tool install -e ".[web]"      # or: pipx install -e ".[web]"
```

This installs `vidvoice` into `~/.local/bin` (already on most PATHs) and links it
to this checkout, so code edits take effect with no reinstall. You never need to
activate a virtualenv, and the command works from any directory.

> `uv tool install -e` recreates the project's `.venv` from only the extras you
> name. If you want the dev environment there too, run
> `uv pip install -e ".[dev]"` afterwards.

### Add your keys

```bash
cp .env.example .env
```

vidvoice resolves configuration in this order:

1. `$VIDVOICE_ENV_FILE`, if set
2. `~/.config/vidvoice/.env` — the user-level location
3. `.env` or `.env.local`, searching upwards from the current directory

Because the command is global, step 2 is normally what you want. Symlink it to
your project file so there is only one copy to edit:

```bash
mkdir -p ~/.config/vidvoice
ln -s ~/Projects/himanshu/vidvoice/.env ~/.config/vidvoice/.env
```

| Key | Where to get it | Used for |
| --- | --- | --- |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | watching the video, writing the script |
| `CARTESIA_API_KEY` | https://play.cartesia.ai/keys | synthesising the voice |

Check everything is wired up:

```bash
vidvoice doctor
```

You can also verify the whole pipeline **without any API keys** using `--dry-run`,
which substitutes a placeholder script and tones for the real APIs.

---

## Use it

### Web GUI

```bash
vidvoice gui
```

Opens `http://127.0.0.1:8765`. Drag a recording in, pick a voice and one of
Cartesia's 58 emotions, hit **Preview voice** to hear it, then **Generate
narration**. Progress is shown live and the result is playable in the page, with
download links for the video and every subtitle format.

### Command line

```bash
# Narrate one recording
vidvoice render demo.mp4

# Pick a voice and emotion
vidvoice render demo.mp4 --voice db6b0ed5-d5d3-463d-ae85-518a07d3c2b4 --emotion confident

# Audition a voice before committing to a render
vidvoice preview --voice <id> --emotion excited

# See the script and its word budget without spending on TTS
vidvoice analyze demo.mp4

# Narrate every recording in a folder
vidvoice batch ./recordings --recursive

# Try it with no API keys at all
vidvoice render demo.mp4 --dry-run
```

### Python

```python
from vidvoice import Settings, render

result = render(
    "demo.mp4",
    settings=Settings.load(),
    voice_id="db6b0ed5-d5d3-463d-ae85-518a07d3c2b4",
    emotion="calm",
    target_wpm=145,
)
print(result.output_path)      # demo_narrated.mp4
print(result.subtitles["srt"]) # demo_narrated.srt
print(result.warnings)         # anything that needs a human eye
```

---

## How the sync actually works

This is the part that decides whether the output is usable, so it is worth
understanding.

**The problem.** Gemini's idea of how long a sentence takes to say is routinely off
by a factor of two. It also only sees video at *one frame per second*, so its
timestamps are accurate to about a second and no better. Taking its timings
literally produces narration that drifts, races, or runs off the end.

**The approach.** Every line is synthesised separately, so its **real** duration can
be measured with ffprobe. Placement is then solved against those measurements, not
against the model's estimates. Three levers get applied, cheapest first:

1. **Budget the script up front.** From the video's length and your target pace,
   vidvoice computes how many words actually fit and tells Gemini that number *in
   the prompt*. A script that was never too long needs no fixing. (`--target-wpm`)

2. **Fit the global speaking rate.** Cartesia's `speed` parameter re-renders the
   whole script at one consistent pace. Because the entire performance shifts
   together, this sounds natural in a way per-line stretching never does. vidvoice
   predicts a rate from word counts, synthesises, measures the real durations, and
   corrects — converging in one or two passes. Clips are cached by
   `(text, voice, speed)`, so the second pass only pays for what changed.
   (`--fit exact|natural|off`)

3. **Adjust individual lines.** Any segment still too big for its own window is
   time-stretched with `rubberband` (or `atempo` where rubberband is not
   compiled in). Capped at `--max-speed` (default 1.35×), because past roughly
   1.4× a voice starts to sound like a legal disclaimer.

**Two invariants are enforced.** Clips never overlap — a line that runs long pushes
the next one later rather than talking over it — and the mix runs exactly as long as
the video, so nothing is clipped at the end.

After rendering, vidvoice re-measures the finished file and reports the true sync
offset, so you are never guessing whether it worked.

### When it cannot fit

If the script is genuinely too long for the video, vidvoice says so instead of
quietly mangling it:

```
warning: 2 segment(s) could not fit and will run long (worst: +00:03.100 on segment 4).
warning: Narration is 00:03.100 longer than the video.
```

Your options, in order of how good the result sounds:

- Re-run — the prompt now includes the word budget, so a fresh script is usually shorter.
- Lower `--target-wpm` if you want a calmer read, or raise it to say more per minute.
- `--tail pad` (default) freezes the last frame over the overhang.
- `--tail trim` cuts the narration to the video's length.
- `--tail error` refuses to render, useful in a batch job.

---

## What you get

For `demo.mp4`:

| File | What it is |
| --- | --- |
| `demo_narrated.mp4` | the video with the new voice track |
| `demo_narrated.srt` | subtitles, timed to where the audio **actually** sits |
| `demo_narrated.vtt` | the same, for browsers |
| `demo_narrated.txt` | plain transcript |
| `<work-dir>/demo/script.json` | the timed script, editable and reusable |
| `<work-dir>/demo/timeline.json` | per-segment placement: speed, drift, overflow |
| `<work-dir>/demo/report.json` | everything above in one summary |

The per-segment voice clips are kept in `<work-dir>/demo/audio/` on purpose:
they are the cache that makes a re-render cheap. Change the emotion or the pace
and only the clips that actually changed are re-synthesised; re-running the
length-fitting loop costs nothing.

Because the script is a plain JSON file and the intermediate stages are cached,
you can edit `script.json` by hand and re-run to hear the change without paying to
re-analyse the video.

---

## Modes

**`narration`** (default) — for silent recordings. Gemini watches what happens and
writes what a presenter would say, grounded in the real names of things on screen.
It is explicitly told not to invent features and not to read the UI aloud verbatim.

**`transcribe`** — for recordings that already have speech. Gemini transcribes it
verbatim with timings so you can re-voice it in a different voice. No word budget is
applied, because a transcript cannot be shortened.

---

## Voice and emotion

Cartesia exposes 58 emotions. List them:

```bash
vidvoice emotions              # the palette
vidvoice emotions --preview    # render a sample of each primary emotion
```

Use one per render with `--emotion`, or set a default in `.env`. Leave it unset and
the model reads the emotional subtext of the text itself. Emotion is passed at
synthesis time, so it is baked into the cached clip and changing it correctly
invalidates that cache.

---

## Configuration

Every setting can go in `.env` or the real environment. See `.env.example` for the
annotated list. The ones worth knowing:

| Variable | Default | Notes |
| --- | --- | --- |
| `VIDVOICE_TARGET_WPM` | `150` | pace used to budget the script against the video |
| `VIDVOICE_FIT_MODE` | `exact` | `exact`, `natural`, or `off` |
| `VIDVOICE_EMOTION` | *(unset)* | one of the 58 emotions |
| `VIDVOICE_SPEECH_SPEED` | `1.0` | Cartesia's own rate, 0.6–1.5 |
| `VIDVOICE_CARTESIA_MODEL` | `sonic-3.6` | |
| `VIDVOICE_GEMINI_MODEL` | `gemini-3.8-flash` | |
| `VIDVOICE_SAMPLE_RATE` | `48000` | 8000/16000/22050/24000/44100/48000 |
| `VIDVOICE_WORK_DIR` | `~/Library/Caches/vidvoice` | intermediates and the voice cache |

---

## Commands

| Command | Purpose |
| --- | --- |
| `render <video>` | narrate one video |
| `batch <dir>` | narrate a folder, continuing past failures |
| `analyze <video>` | script only — no TTS spend, shows the word budget |
| `voices` | list Cartesia voices |
| `preview` | audition a voice/emotion on one line |
| `emotions` | list the emotion palette, optionally previewing each |
| `doctor` | check ffmpeg, keys, models, and live API connectivity |
| `gui` | launch the local web interface |

Every command takes `--json` for scripting. Failures print a message, an
actionable hint, and exit non-zero.

---

## Development

```bash
uv pip install -e ".[dev]"
.venv/bin/python -m pytest          # 303 tests, ~40s
```

The suite needs **no API keys and no network**. It builds synthetic videos with
ffmpeg and runs the real pipeline against them, so timing, mixing and muxing are
all genuinely exercised. The two API clients are tested against
`httpx.MockTransport`, which asserts the exact wire format — the part most likely
to drift.

The tests that matter most are the invariants:

- narration **never overlaps itself**, asserted on both the plan and the rendered audio;
- audio **lands where the plan says**, verified by measuring the mixed track;
- a script too long for its video is **compressed**, not silently truncated;
- a script that *cannot* fit still renders, and says so.

### Layout

| Module | Responsibility |
| --- | --- |
| `models.py` | `Segment`/`Script`, timestamp parsing, the timeline solver |
| `media.py` | every ffmpeg/ffprobe call, behind testable functions |
| `gemini.py` | video upload, prompting, schema-constrained script |
| `cartesia.py` | TTS, voice listing, on-disk clip cache |
| `timing.py` | word budgeting and global-rate fitting |
| `pipeline.py` | stage orchestration and caching |
| `subtitles.py` | SRT/VTT/TXT export |
| `cli.py`, `web/` | the two front ends |

---

## Notes on the APIs

Both integrations were written against the live specifications and are pinned
deliberately:

- **Cartesia** requires a `Cartesia-Version` header whose enum currently accepts
  only `2026-08-14`. The voice specifier is a bare id or `{"id": ...}` — the older
  `{"mode": "id"}` form and inline embeddings were removed. `/tts/bytes` cannot
  return word timestamps, which is precisely why timing is derived from measured
  audio duration rather than requested from the API.
- **Gemini** video is sampled at 1 FPS with timestamps attached per second, so
  vidvoice asks for whole-second `MM:SS` rather than false millisecond precision.
  The request uses `responseJsonSchema`, not `responseSchema` — the latter
  serialises `propertyOrdering` incorrectly. `temperature` is not sent, as it is
  deprecated on current models.

Gemini keeps uploaded files for 48 hours; vidvoice deletes each one as soon as its
analysis finishes.
