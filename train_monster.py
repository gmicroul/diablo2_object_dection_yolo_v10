#!/usr/bin/env python3
"""一键训练怪物检测模型"""
import subprocess
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
MONSTER_DIR = os.path.join(BASE, "monster_data")

# 1. 检查标注文件
train_img_dir = os.path.join(MONSTER_DIR, "images", "train")
train_label_dir = os.path.join(MONSTER_DIR, "labels", "train")
val_img_dir = os.path.join(MONSTER_DIR, "images", "val")
val_label_dir = os.path.join(MONSTER_DIR, "labels", "val")

if not os.path.exists(train_img_dir):
    print(f"错误: 找不到 {train_img_dir}")
    sys.exit(1)

train_imgs = [f for f in os.listdir(train_img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
labeled = 0
for img in train_imgs:
    base = os.path.splitext(img)[0]
    if os.path.exists(os.path.join(train_label_dir, base + ".txt")):
        labeled += 1

val_imgs = [f for f in os.listdir(val_img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
val_labeled = 0
for img in val_imgs:
    base = os.path.splitext(img)[0]
    if os.path.exists(os.path.join(val_label_dir, base + ".txt")):
        val_labeled += 1

print(f"训练集: {len(train_imgs)} 张图, {labeled} 张已标注")
print(f"验证集: {len(val_imgs)} 张图, {val_labeled} 张已标注")

if labeled == 0:
    print("错误: 训练集里没有标注文件，请先运行 label.py 标注")
    sys.exit(1)

# 2. 自动检测类别数
all_labels = []
for d in [train_label_dir, val_label_dir]:
    if os.path.exists(d):
        for f in os.listdir(d):
            if f.endswith(".txt"):
                with open(os.path.join(d, f)) as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if parts:
                            all_labels.append(int(parts[0]))
nc = max(set(all_labels)) + 1 if all_labels else 1
print(f"检测到 {nc} 个类别")

# 3. 自动生成类别名
class_names = []
for i in range(nc):
    class_names.append(f"class_{i}")

# 4. 写 data.yaml
yaml_path = os.path.join(MONSTER_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"train: {train_img_dir}\n")
    f.write(f"val: {val_img_dir}\n")
    f.write(f"\nnc: {nc}\n")
    f.write(f"names: {class_names}\n")
print(f"已生成 {yaml_path}")

# 5. 检查可用的预训练权重
pretrained = os.path.join(BASE, "runs", "detect", "train", "weights", "best.pt")
if not os.path.exists(pretrained):
    pretrained = "yolov8n.pt"  # 兜底，自动下载

# 6. 训练
cmd = [
    "yolo", "detect", "train",
    f"model={pretrained}",
    f"data={yaml_path}",
    "epochs=300",
    "imgsz=640",
    "batch=16",
    "patience=50",
    "device=cpu",
    "project=monster_runs",
    "name=monster_train",
    "exist_ok=True",
    "plots=True",
    "lr0=0.01",
    "amp=True",
]
print(f"\n训练命令: {' '.join(cmd)}\n")
subprocess.run(cmd, cwd=BASE)

# 7. 导出 ONNX
best_pt = os.path.join(BASE, "monster_runs", "detect", "monster_train", "weights", "best.pt")
if os.path.exists(best_pt):
    onnx_path = os.path.join(BASE, "monster_model.onnx")
    export_cmd = [
        "yolo", "export",
        f"model={best_pt}",
        "format=onnx",
        "simplify=True",
        f"imgsz=640",
    ]
    print(f"\n导出 ONNX: {' '.join(export_cmd)}\n")
    subprocess.run(export_cmd, cwd=BASE)
    print(f"\n完成! ONNX 模型: {onnx_path}")
else:
    print(f"\n训练未完成，找不到 {best_pt}")