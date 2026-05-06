#!/usr/bin/env python3
"""暗黑破坏神2 自动刷怪 — 截屏 + YOLO 检测 + 自动攻击"""
import mss
import cv2
import numpy as np
import time
import sys
import os
import subprocess

sys.path.insert(0, "/home/user/yolov8n-sfos/python")

from yolo_backend import YOLOv8Detector
MODEL = "/home/user/diablo2_object_dection_yolo_v10/runs/detect/monster_runs/monster_finetune/weights/best.onnx"
if not os.path.exists(MODEL):
    MODEL = "/home/user/diablo2_object_dection_yolo_v10/runs/detect/monster_runs/monster_finetune/weights/best.onnx"
det = YOLOv8Detector(MODEL, conf=0.4, iou=0.45)
print("Using ONNX YOLOv8 model")

import pyautogui
import random

MONSTER_CLASS = 0  # 专注模型只有1个类（monster），cls=0
ATTACK_BUTTON = "left"
ROLE_CENTER_RADIUS = 60
ATTACK_COOLDOWN = 0.2
MOVE_DURATION = 2.0
MOVE_INTERVAL = 3.0
HIT_CONFIRM_COUNT = 5  # 攻击同一目标5次还没死，判定为假目标


def get_window_rect(window_title="Wine Desktop"):
    result = subprocess.run(
        ["xdotool", "search", "--name", window_title],
        capture_output=True, text=True
    )
    if not result.stdout.strip():
        return None
    wid = result.stdout.strip().split("\n")[0]
    result = subprocess.run(
        ["xdotool", "getwindowgeometry", wid],
        capture_output=True, text=True
    )
    x, y, w, h = 0, 0, 0, 0
    for line in result.stdout.split("\n"):
        if "Position" in line:
            xy = line.split(":")[1].strip().split()[0]
            x, y = map(int, xy.split(","))
        if "Geometry" in line:
            wh = line.split(":")[1].strip()
            w, h = map(int, wh.split("x"))
    return {"left": x, "top": y, "width": w, "height": h}


def move_to_offset(offset, dx, dy):
    """鼠标点击窗口内位置行走（限制在游戏窗口内）"""
    cx = offset['left'] + offset['width'] // 2
    cy = offset['top'] + offset['height'] // 2
    # 限制点击位置在窗口内，留10px边距
    margin = 10
    x = max(offset['left'] + margin, min(cx + dx, offset['left'] + offset['width'] - margin))
    y = max(offset['top'] + margin, min(cy + dy, offset['top'] + offset['height'] - margin))
    pyautogui.moveTo(x, y, duration=0.1)
    pyautogui.click(button='left')
    time.sleep(0.1)


def attack_monster(d, offset):
    x = offset['left'] + (d['x1'] + d['x2']) // 2
    y = offset['top'] + (d['y1'] + d['y2']) // 2
    pyautogui.moveTo(x, y, duration=0.05)
    pyautogui.click(button=ATTACK_BUTTON)


print("Waiting for Wine Desktop window...")
window_rect = None
while window_rect is None:
    window_rect = get_window_rect("Wine Desktop")
    if window_rect is None:
        print("Window not found, retrying in 2s...")
        time.sleep(2)

print(f"Window found: {window_rect}")

with mss.mss() as sct:
    monitor = window_rect
    fps_counter = 0
    fps_time = time.time()
    fps = 0
    attack_count = 0
    last_attack = 0
    last_monster_seen = time.time()
    last_force_move = time.time()
    is_moving = False
    attack_targets = {}  # 记录每个目标被攻击次数 {id: count}

    print("Auto-farming started. Press Ctrl+C to stop.")
    try:
        while True:
            img = sct.grab(monitor)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

            dets, dt = det.infer(frame)

# 找怪物
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            monsters = []
            any_class14 = True if len(dets) > 0 else False
            for d in dets:
                obj_cx = (d['x1'] + d['x2']) // 2
                obj_cy = (d['y1'] + d['y2']) // 2
                dist = ((obj_cx - cx) ** 2 + (obj_cy - cy) ** 2) ** 0.5
                monsters.append((d, dist))

            # 过滤已判定为假目标（攻击多次没死）
            filtered_monsters = []
            for m, dist in monsters:
                mid = id(m)
                if mid in attack_targets and attack_targets[mid] >= HIT_CONFIRM_COUNT:
                    continue  # 假目标，跳过
                filtered_monsters.append((m, dist))
            monsters = filtered_monsters

            # 假目标不计入any_class14（不影响寻路）
            real_class14 = any_class14 and not (len(monsters) == 0 and any_class14)
            # 更准确：画面中是否有非假目标的class_14
            has_real_target = len(monsters) > 0
            has_class14_but_fake = any_class14 and not has_real_target

            # 自动攻击
            now = time.time()
            if monsters and (now - last_attack) > ATTACK_COOLDOWN:
                target, dist = min(monsters, key=lambda x: x[1])
                attack_monster(target, window_rect)
                last_attack = now
                last_monster_seen = now
                is_moving = False
                attack_count += 1
                print(f"  ATTACK #{attack_count} dist={dist:.0f} ({len(monsters)} monsters)")

                # 记录攻击次数
                tid = id(target)
                if tid not in attack_targets:
                    attack_targets[tid] = 0
                attack_targets[tid] += 1

                # 如果攻击HIT_CONFIRM_COUNT次还没消失，标记为假目标
                if attack_targets[tid] >= HIT_CONFIRM_COUNT:
                    print(f"  Fake target #{tid % 1000}, excluding")
                if attack_count <= 10 or attack_count % 30 == 0:
                    print(f"  Attack #{attack_count} dist={dist:.0f}px")

# 自动寻路
            if has_class14_but_fake:
                if not is_moving:
                    dx = random.randint(-150, 150)
                    dy = random.randint(-200, 50)
                    move_to_offset(window_rect, dx, dy)
                    is_moving = True
                    last_force_move = now
                    print(f"  Fake only ({dx:+d},{dy:+d})")
                elif (now - last_force_move) > 1.0:
                    is_moving = False
            elif not monsters and not any_class14 and (now - last_monster_seen) > MOVE_INTERVAL:
                if not is_moving:
                    dx = random.randint(-150, 150)
                    dy = random.randint(-200, 50)
                    move_to_offset(window_rect, dx, dy)
                    is_moving = True
                    last_monster_seen = now
                    print(f"  Walk ({dx:+d},{dy:+d})")
                elif (now - last_monster_seen) > MOVE_INTERVAL + MOVE_DURATION:
                    is_moving = False
                    last_monster_seen = now
            else:
                is_moving = False

            # 画框
            for d in dets:
                cv2.rectangle(frame,
                    (d['x1'], d['y1']), (d['x2'], d['y2']),
                    (0, 255, 0), 2)
                cv2.putText(frame, f"{d['name']} {d['conf']:.3f}",
                    (d['x1'], d['y1']-4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

            # FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            cv2.putText(frame, f"FPS:{fps} Att:{attack_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Diablo2 Auto Farm", frame)
            cv2.moveWindow("Diablo2 Auto Farm", 0, 0)  # 左上角显示
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print(f"Stopped. {attack_count} attacks performed.")