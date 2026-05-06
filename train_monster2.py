#!/usr/bin/env python3
"""把 train2 切分后训练, 用 onnxruntime 训练（不装pyTorch）"""
import os, sys, glob, random, shutil

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE, "monster_data")

# 把 train2 的东西放到训练集/验证集
train_img = os.path.join(SRC_DIR, "images", "train2")
train_lbl = os.path.join(SRC_DIR, "labels", "train2")

# 列出全部图片
imgs = sorted(glob.glob(os.path.join(train_img, "*.jpg")))
random.shuffle(imgs)
n_val = max(1, int(len(imgs) * 0.15))
val_imgs = imgs[:n_val]
train_imgs = imgs[n_val:]

# 验证集
val_img_dir = os.path.join(SRC_DIR, "images", "val")
val_lbl_dir = os.path.join(SRC_DIR, "labels", "val")
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

for path in val_imgs:
    base = os.path.splitext(os.path.basename(path))[0]
    # 移动图片
    os.rename(path, os.path.join(val_img_dir, os.path.basename(path)))
    # 移动标注
    lbl_src = os.path.join(train_lbl, base + ".txt")
    if os.path.exists(lbl_src):
        os.rename(lbl_src, os.path.join(val_lbl_dir, base + ".txt"))

print(f"训练集: {len(train_imgs)} 张")
print(f"验证集: {len(val_imgs)} 张")

# 写 data.yaml
yaml_path = os.path.join(SRC_DIR, "data_monster.yaml")
with open(yaml_path, "w") as f:
    f.write(f"train: {train_img}\n")
    f.write(f"val: {val_img_dir}\n")
    f.write(f"\nnc: 1\n")
    f.write(f"names: ['monster']\n")
print(f"data.yaml: {yaml_path}")

# 用 ultralytics 训练（已经装好了不炸系统）
print("\n开始训练...")
import subprocess
cmd = [
    sys.executable, "-m", "ultralytics.engine.trainer",
    "model=yolov8n.pt",
    f"data={yaml_path}",
    "epochs=200",
    "imgsz=640",
    "batch=16",
    "patience=30",
    "device=cpu",
    "project=monster_runs",
    "name=monster_v2",
    "exist_ok=True",
    "plots=True",
    "lr0=0.01",
]
print(f"命令: {' '.join(cmd)}")
subprocess.run(cmd, cwd=BASE, env={**os.environ})

# 导出 ONNX
best_pt = os.path.join(BASE, "monster_runs", "detect", "monster_v2", "weights", "best.pt")
if os.path.exists(best_pt):
    export_cmd = [
        sys.executable, "-m", "ultralytics.engine.exporter",
        f"model={best_pt}",
        "format=onnx",
        "simplify=True",
    ]
    print(f"导出 ONNX: {' '.join(export_cmd)}")
    subprocess.run(export_cmd, cwd=BASE)
    # 复制到项目根
    import shutil
    onnx_src = os.path.join(os.path.dirname(os.path.dirname(best_pt)), "best.onnx")
    if os.path.exists(onnx_src):
        shutil.copy(onnx_src, os.path.join(BASE, "monster_model.onnx"))
        print(f"模型已导出: monster_model.onnx")
else:
    print(f"训练失败？找不到 {best_pt}")