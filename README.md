# Diablo 2 Auto Farm — AI-Powered Autonomous Farming 🤖⚔️

> **Never click a monster again.** Your YOLO-powered bot sees, hunts, and kills — all on its own.

Built for **Wine Desktop** on **ARM64 Linux** (harbour-containers on Sailfish OS). Works on any Linux with a windowed game.

## ✨ Features

- **🔍 Real-time object detection** — YOLOv8n ONNX runs at ~200-300ms per frame
- **🎯 Dedicated monster model** — Custom-trained 1-class YOLOv8n model focused on Diablo 2 monsters
- **🤖 Autonomous pathfinding** — No monster in sight? Randomly walks forward to explore
- **⚔️ Auto-attack** — Instant left-click attack on the nearest detected monster (200ms cooldown)
- **🖥️ Visual feedback** — OpenCV window shows real-time detection boxes, FPS, attack count
- **📝 Terminal-based annotation** — `label_focus2.py` keyboard-only labeler for curating training data
- **🎮 Works with any game** — Screen capture via `mss`, just point the script at your game window

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
│  mss screencap │────▶│  YOLOv8 ONNX  │────▶│  Attack cmd   │
│  (800×600)    │     │  (monster cls) │     │  (0.2s cd)    │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 ▼
┌─────────────┐     ┌──────────────┐
│  pyautogui   │◀────│  Nearest      │
│  left-click  │     │  monster      │
└─────────────┘     └──────────────┘
```

## 🏋️ Training Pipeline

This repo includes a complete training pipeline for a custom Diablo 2 monster detector:

```
screenshot (mss) → label_focus2.py (annotate) → train_monster.py (YOLO) → best.onnx → diablo2_detect.py
```

1. **Capture** — `capture_game.py` takes screenshots of gameplay
2. **Annotate** — `label_focus2.py` keyboard-only terminal labeler with auto-prelabeling
3. **Prelabel** — `prelabel2.py` batch pre-label with existing model
4. **Train** — `train_monster2.py` fine-tune YOLOv8n on custom data
5. **Export** — `export_onnx.py` convert trained model to ONNX
6. **Farm** — `diablo2_detect.py` use the model to auto-farm

## 📦 Files

```
├── diablo2_detect.py            ← The bot (entry point)
├── run-diablo2-detect.sh        ← Convenience runner (venv + DISPLAY)
│
├── label_focus2.py              ← Terminal-based annotation tool (keyboard only)
├── label_cli.py                 ← Alternative CLI annotator
├── label_simple.py              ← OpenCV GUI annotator (needs X11)
├── prelabel2.py                 ← Batch pre-labeling with ONNX model
│
├── train_monster2.py            ← YOLOv8n fine-tuning script
├── export_onnx.py               ← Export trained model to ONNX
├── split_val.py                 ← Train/val split
├── capture_game.py              ← Screenshot capture tool
│
├── runs/detect/monster_runs/
│   └── monster_finetune/        ← Trained model weights
│       └── weights/best.onnx    ← Active model (1-class monster)
│
└── monster_data/                ← Training dataset (images + YOLO labels)
    ├── images/train2/
    └── labels/train2/
```

## ⚙️ Configuration

Edit these constants at the top of `diablo2_detect.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `ATTACK_COOLDOWN` | `0.2` | Seconds between attacks |
| `MOVE_INTERVAL` | `3.0` | Seconds idle before auto-walk |

## 🧪 Training Your Own Model

```bash
# 1. Capture gameplay screenshots
python3 capture_game.py

# 2. Annotate with terminal labeler
python3 label_focus2.py

# 3. Train (adjust paths in the script)
python3 train_monster2.py

# 4. Export to ONNX
python3 export_onnx.py

# 5. Update model path in diablo2_detect.py and run
python3 diablo2_detect.py
```

## ⚠️ Disclaimer

This is an AI research project. Use responsibly and in accordance with game terms of service. The bot is visible on screen — don't leave it unattended on Battle.net.