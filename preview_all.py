#!/usr/bin/env python3
"""批量生成预标注预览图，快速评估标注质量"""
import os, glob, cv2, numpy as np

BASE = "/home/user/diablo2_object_dection_yolo_v10"
IMG_DIR = os.path.join(BASE, "monster_data/images/train2")
LABEL_DIR = os.path.join(BASE, "monster_data/labels/train2")
OUT_DIR = os.path.join(BASE, "preview_boxes")
os.makedirs(OUT_DIR, exist_ok=True)

# 模型
import onnxruntime as ort
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.enable_mem_pattern = False
sess_opts.intra_op_num_threads = 4

MODEL_PATH = os.path.join(BASE, "runs/detect/train/weights/best.onnx")
FALLBACK_MODEL = os.path.join(BASE, "yolov8n_relu_20class_zq.onnx")

def load_onnx(path):
    if not os.path.exists(path): return None
    try:
        return ort.InferenceSession(path, sess_opts, providers=['CPUExecutionProvider'])
    except:
        return None

session = load_onnx(MODEL_PATH) or load_onnx(FALLBACK_MODEL)
input_name = session.get_inputs()[0].name
inp_h, inp_w = session.get_inputs()[0].shape[2:4]

def prelabel(img):
    h, w = img.shape[:2]
    blob = cv2.resize(img, (inp_w, inp_h)).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, 0)
    outs = session.run(None, {input_name: blob})[0][0]
    scores = outs[4:, :]
    best_score = np.max(scores, axis=0)
    best_cls = np.argmax(scores, axis=0)
    mask = best_score > 0.3
    if not mask.any(): return []
    best_score = best_score[mask]
    outs = outs[:4, mask]
    result = []
    for i in range(outs.shape[1]):
        cx, cy, bw, bh = outs[:, i]
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        result.append((x1, y1, x2, y2, float(best_score[i])))
    # NMS
    keep = []
    for i in sorted(range(len(result)), key=lambda i: result[i][4], reverse=True):
        ok = True
        for j in keep:
            ix = max(result[i][0], result[j][0])
            iy = max(result[i][1], result[j][1])
            ix2 = min(result[i][2], result[j][2])
            iy2 = min(result[i][3], result[j][3])
            inter = max(0, ix2-ix) * max(0, iy2-iy)
            area_i = (result[i][2]-result[i][0]) * (result[i][3]-result[i][1])
            area_j = (result[j][2]-result[j][0]) * (result[j][3]-result[j][1])
            if inter / (area_i + area_j - inter) > 0.5:
                ok = False
                break
        if ok: keep.append(i)
    return [result[k] for k in keep]

# 获取已保存的标注文件列表和图片列表
lbl_files = {os.path.splitext(f)[0] for f in os.listdir(LABEL_DIR) if f.endswith('.txt')}
img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))

stats = {"total": 0, "has_label": 0, "empty_label": 0, "prelabel_only": 0, "missed_monsters": 0}

for i, img_path in enumerate(img_files):
    base = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None: continue
    h, w = img.shape[:2]
    
    # 加载已有标注
    existing_boxes = []
    lbl_path = os.path.join(LABEL_DIR, base + ".txt")
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    existing_boxes.append((x1, y1, x2, y2))
        stats["has_label"] += 1
    else:
        stats["empty_label"] += 1
    
    # 预标注
    pre = prelabel(img)
    
    # 统计预标注发现的怪物数
    stats["total"] += 1
    if pre and not existing_boxes:
        stats["prelabel_only"] += 1
    
    # 画图
    display = img.copy()
    
    # 已有标注（绿色）
    for x1, y1, x2, y2 in existing_boxes:
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)
    
    # 预标注（蓝色）— 如果已有标注就不画预标注了
    if not existing_boxes:
        for x1, y1, x2, y2, conf in pre:
            cv2.rectangle(display, (x1, y1), (x2, y2), (200, 0, 0), 2)
            cv2.putText(display, f"{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,0,0), 1)
    
    # 信息
    info = f"{base} | 已有框:{len(existing_boxes)} 预标:{len(pre)}"
    cv2.putText(display, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # 按比例缩小显示
    scale = 800 / max(w, h)
    if scale < 1:
        new_w, new_h = int(w*scale), int(h*scale)
        display = cv2.resize(display, (new_w, new_h))
    
    out_path = os.path.join(OUT_DIR, f"{i:04d}_{base}_boxes_{len(existing_boxes)}_pre_{len(pre)}.jpg")
    cv2.imwrite(out_path, display)

print(f"\n=== 统计 ===")
print(f"总图片: {stats['total']}")
print(f"有标注: {stats['has_label']}")
print(f"无标注: {stats['empty_label']}")
print(f"仅有预标注: {stats['prelabel_only']}")
print(f"\n预览图已生成: {OUT_DIR}/")
print(f"共 {len(os.listdir(OUT_DIR))} 张")
print(f"\n用看图工具打开 {OUT_DIR}/ 批量查看")