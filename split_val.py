#!/usr/bin/env python3
"""将一些训练集图片移到验证集（按比例）"""
import os, sys, random, shutil

MONSTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monster_data")
train_img = os.path.join(MONSTER_DIR, "images", "train")
val_img = os.path.join(MONSTER_DIR, "images", "val")
train_label = os.path.join(MONSTER_DIR, "labels", "train")
val_label = os.path.join(MONSTER_DIR, "labels", "val")
os.makedirs(val_label, exist_ok=True)

ratio = 0.2 if len(sys.argv) < 2 else float(sys.argv[1])

imgs = [f for f in os.listdir(train_img) if f.endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(imgs)
n = max(1, int(len(imgs) * ratio))
move_imgs = imgs[:n]

for f in move_imgs:
    base = os.path.splitext(f)[0]
    shutil.move(os.path.join(train_img, f), os.path.join(val_img, f))
    label_src = os.path.join(train_label, base + ".txt")
    if os.path.exists(label_src):
        shutil.move(label_src, os.path.join(val_label, base + ".txt"))

print(f"移动了 {n} 张图到验证集")