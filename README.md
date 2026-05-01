# Diablo 2 Auto Farm — AI-Powered Autonomous Farming 🤖⚔️

> **Never click a monster again.** Your YOLO-powered bot sees, hunts, and kills — all on its own.

## ✨ Features

- **🔍 Real-time object detection** — YOLOv8 ONNX runs at 200-300ms per frame, spotting monsters, items, and terrain in real time
- **🎯 Smart target filtering** — Automatically distinguishes monsters from:
  - Your own character and mercenary (center-of-screen exclusion)
  - Fire pits, cactus, walls, and scenery (size/shape filtering)
  - Fake targets (kills the same thing 5× without it dying? Never attacks it again.)
- **🤖 Autonomous pathfinding** — No monster in sight? Randomly walks forward to explore. Enters caves, passes through doors, doesn't get stuck.
- **⚔️ Auto-attack** — Instant left-click attack on the nearest confirmed monster (200ms cooldown)
- **🖥️ Visual feedback** — OpenCV window shows real-time detection boxes, FPS, attack count
- **🎮 Works with any game** — Screen capture via `mss`, just point the script at your game window

## 📸 Demo

```
[class_14] → detected → attacked → dead → next target
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install mss opencv-python pyautogui onnxruntime numpy

# 2. Run (make sure Diablo 2 / Wine Desktop is open)
python3 diablo2_detect.py
```

**Controls:**
- `q` — Quit
- `Ctrl+C` — Stop
- The OpenCV window auto-docks at screen position (0,0)

## 🧠 How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  mss screencap │────▶│  YOLOv8 ONNX  │────▶│  Smart Filter │
│  (800×600)    │     │  (12MB model) │     │  (player/mer  │
└─────────────┘     └──────────────┘     │  c/fake/terr) │
                                          └──────┬───────┘
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  pyautogui   │◀────│  Attack cmd  │◀────│  Nearest      │
│  left-click  │     │  (0.2s cd)   │     │  monster      │
└─────────────┘     └──────────────┘     └──────────────┘
```

## 🎮 Recommended Use Cases

- **Act boss farming** — Andy, Meph, Baal runs on autopilot
- **Leveling** — AFK XP grind in any zone
- **Item hunting** — Filter can be extended to auto-loot class_6 items
- **Any old-school ARPG** — Works with any fullscreen window

## 🛠️ Configuration

Edit these constants at the top of `diablo2_detect.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `MONSTER_CLASS` | `"class_14"` | YOLO class ID for monsters |
| `ATTACK_COOLDOWN` | `0.2` | Seconds between attacks |
| `ROLE_CENTER_RADIUS` | `60` | Pixels around screen center to exclude as player |
| `HIT_CONFIRM_COUNT` | `5` | Attacks before marking target as fake |
| `MOVE_INTERVAL` | `3.0` | Seconds idle before auto-walk |

## 🧪 Training Your Own Model

This repo includes a pre-trained YOLOv8n model (20 classes, unknown training data — works surprisingly well on D2). For a dedicated Diablo 2 model:

1. Use `mss` to screenshot gameplay
2. Label with `labelImg`
3. Train with YOLOv10 (see `diablo2_object_detection.ipynb`)
4. Replace the `.onnx` file

## 📦 Files

```
diablo2_detect.py                 ← The bot (you are here)
yolov8n_relu_20class_zq.onnx      ← The model (12MB)
diablo2_object_detection.ipynb    ← Training notebook
runs/detect/train/weights/        ← Gem-detection model (bonus!)
```

## ⚠️ Disclaimer

This is an AI research project. Use responsibly and in accordance with game terms of service. The bot is visible on screen — don't leave it unattended on Battle.net.