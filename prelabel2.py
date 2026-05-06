#!/usr/bin/env python3
"""预标注 train2 目录的图片，只取 class_14（怪物）"""
import os, glob, cv2, numpy as np, onnxruntime

MODEL = "yolov8n_relu_20class_zq.onnx"
SRC = "monster_data/images/train2"
DST_LABEL = "monster_data/labels/train2"
os.makedirs(DST_LABEL, exist_ok=True)

session = onnxruntime.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
_, _, ih, iw = session.get_inputs()[0].shape

def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size/h, size/w)
    nw, nh = int(w*r), int(h*r)
    dw, dh = (size - nw)//2, (size - nh)//2
    img = cv2.resize(img, (nw, nh))
    return cv2.copyMakeBorder(img, dh, dh+((size-nh)%2), dw, dw+((size-nw)%2),
                              cv2.BORDER_CONSTANT, value=(114,114,114)), r, dw, dh

def iou(a, b):
    ax1, ay1 = a['cx']-a['bw']/2, a['cy']-a['bh']/2
    ax2, ay2 = a['cx']+a['bw']/2, a['cy']+a['bh']/2
    bx1, by1 = b['cx']-b['bw']/2, b['cy']-b['bh']/2
    bx2, by2 = b['cx']+b['bw']/2, b['cy']+b['bh']/2
    inter = max(0, min(ax2,bx2)-max(ax1,bx1)) * max(0, min(ay2,by2)-max(ay1,by1))
    area_a = a['bw']*a['bh']; area_b = b['bw']*b['bh']
    return inter / (area_a + area_b - inter + 1e-6)

imgs = sorted(glob.glob(os.path.join(SRC, "*.jpg")))
print(f"处理 {len(imgs)} 张...")

total_boxes = 0
labeled = 0
for path in imgs:
    img = cv2.imread(path)
    if img is None: continue
    h0, w0 = img.shape[:2]
    proc, r, px, py = letterbox(img)
    inp = proc.astype(np.float32).transpose(2,0,1)[np.newaxis,...] / 255.0
    out = session.run(None, {input_name: inp})[0][0]

    boxes = []
    for i in range(out.shape[1]):
        v = out[:, i]
        scores = v[4:]
        cls_id = int(np.argmax(scores))
        conf = float(scores[cls_id])
        if cls_id != 14 or conf < 0.015:  # 只取 class_14（怪物）
            continue
        cx = ((v[0] - px) / r) / w0 * 800
        cy = ((v[1] - py) / r) / h0 * 600
        bw = (v[2] / r) / w0 * 800
        bh = (v[3] / r) / h0 * 600
        boxes.append({'conf': conf, 'cx': cx, 'cy': cy, 'bw': bw, 'bh': bh})

    if not boxes:
        continue

    boxes.sort(key=lambda x: x['conf'], reverse=True)
    keep = []
    while boxes:
        b = boxes.pop(0)
        keep.append(b)
        boxes = [x for x in boxes if iou(b, x) < 0.45]

    # 过滤太小框
    valid = [b for b in keep if b['bw']*800 > 15 and b['bh']*600 > 15]

    if not valid:
        continue

    base = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(DST_LABEL, base + ".txt"), "w") as f:
        for b in valid:
            xc = b['cx'] / 800
            yc = b['cy'] / 600
            bw_n = b['bw'] / 800
            bh_n = b['bh'] / 600
            f.write(f"0 {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}\n")
    total_boxes += len(valid)
    labeled += 1

print(f"标注了 {labeled} 张，共 {total_boxes} 个怪物框")
print(f"标注文件: {len(glob.glob(os.path.join(DST_LABEL, '*.txt')))}")