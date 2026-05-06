#!/usr/bin/env python3
"""用现有ONNX模型对截图做预标注（检测到class_14怪物就标注）"""
import os
import sys
import glob
import onnxruntime
import numpy as np
import cv2

MODEL = "yolov8n_relu_20class_zq.onnx"
IMG_DIR = "monster_data/images/train"
LABEL_DIR = "monster_data/labels/train"
os.makedirs(LABEL_DIR, exist_ok=True)

CLASSES = ["class_0","class_1","class_2","class_3","class_4","class_5",
           "class_6","class_7","class_8","class_9","class_10","class_11",
           "class_12","class_13","class_14","class_15","class_16",
           "class_17","class_18","class_19"]

# 加载 ONNX
session = onnxruntime.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
_, _, input_h, input_w = session.get_inputs()[0].shape

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh, new_shape[0] - new_unpad[1] - dh
    left, right = dw, new_shape[1] - new_unpad[0] - dw
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def infer(frame, conf_thres=0.015, iou_thres=0.45):
    h0, w0 = frame.shape[:2]
    img, ratio, pad = letterbox(frame, (input_h, input_w))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]

    outputs = session.run(None, {input_name: img})[0][0]  # (84, 8400)
    boxes = []
    for i in range(outputs.shape[1]):
        pred = outputs[:, i]
        scores = pred[4:]
        cls_id = int(np.argmax(scores))
        conf = float(scores[cls_id])
        if conf < conf_thres:
            continue
        cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
        # 转换回原图坐标
        cx = (cx - pad[0]) / ratio
        cy = (cy - pad[1]) / ratio
        bw = bw / ratio
        bh = bh / ratio
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0, x2), min(h0, y2)
        boxes.append({
            'name': CLASSES[cls_id],
            'cls': cls_id,
            'conf': conf,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        })

    # NMS
    if not boxes:
        return []
    boxes.sort(key=lambda x: x['conf'], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if
                 calc_iou(best, b) < iou_thres]
    return keep

def calc_iou(a, b):
    ix1 = max(a['x1'], b['x1'])
    iy1 = max(a['y1'], b['y1'])
    ix2 = min(a['x2'], b['x2'])
    iy2 = min(a['y2'], b['y2'])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
    area_b = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
    return inter / (area_a + area_b - inter + 1e-6)

imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
print(f"处理 {len(imgs)} 张图片...")

detected_count = 0
for path in imgs:
    frame = cv2.imread(path)
    if frame is None:
        continue
    h, w = frame.shape[:2]
    dets = infer(frame, conf_thres=0.015)

    # 只保留 class_14 (怪物)
    monsters = [d for d in dets if d['cls'] == 14]
    # 也保留一些其他类可能值得看看的
    if not monsters:
        continue

    base = os.path.splitext(os.path.basename(path))[0]
    label_path = os.path.join(LABEL_DIR, base + ".txt")

    lines = []
    for d in monsters:
        xc = ((d['x1'] + d['x2']) / 2) / w
        yc = ((d['y1'] + d['y2']) / 2) / h
        bw = (d['x2'] - d['x1']) / w
        bh = (d['y2'] - d['y1']) / h
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    # 过滤尺寸过小的框（可能是噪声）
    valid = []
    for l in lines:
        parts = l.strip().split()
        bw, bh = float(parts[3]), float(parts[4])
        if bw * w > 10 and bh * h > 10:  # 至少10x10像素
            valid.append(l)

    if valid:
        with open(label_path, "w") as f:
            f.write("\n".join(valid))
        detected_count += 1

print(f"标注了 {detected_count} 张图片")
n_labels = len(glob.glob(os.path.join(LABEL_DIR, "*.txt")))
print(f"标注文件总数: {n_labels}")