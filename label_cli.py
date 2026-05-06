#!/usr/bin/env python3
"""
终端标注器 v3 — 纯键盘，无OpenCV窗口
1. 显示图片路径和框信息
2. 按 v 键在 Wine Desktop 预览
3. 键盘调框

用法:
  python3 label_cli.py

命令（直接输入）:
  n       新增框（在图片中心放80x80框）
  d N     切换选中框到 N 号
  w/a/s/d 微调 1px
  W/A/S/D 微调 10px
  e       扩大 5px
  q       缩小 5px
  x       删除选中框
  v       在 Wine 下预览
  p       上一张
  SPACE   下一张（自动保存）
  s       保存
  r       重新预标注
  h       帮助
  quit    退出
"""

import os, sys, glob, cv2, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "monster_data/images/train2")
LABEL_DIR = os.path.join(BASE, "monster_data/labels/train2")
os.makedirs(LABEL_DIR, exist_ok=True)

# 模型
MODEL_PATH = os.path.join(BASE, "runs/detect/train/weights/best.onnx")
FALLBACK_MODEL = os.path.join(BASE, "yolov8n_relu_20class_zq.onnx")

import onnxruntime as ort
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.enable_mem_pattern = False
sess_opts.intra_op_num_threads = 4

def load_onnx(path):
    if not os.path.exists(path):
        return None
    try:
        return ort.InferenceSession(path, sess_opts, providers=['CPUExecutionProvider'])
    except:
        return None

session = load_onnx(MODEL_PATH) or load_onnx(FALLBACK_MODEL)
if session is None:
    print("[ERROR] 无模型")
    sys.exit(1)
input_name = session.get_inputs()[0].name
inp_h, inp_w = session.get_inputs()[0].shape[2:4]
print(f"模型: 640x640  | 图片目录: {IMG_DIR}")

# 图片列表
jpg_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
png_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
img_files = jpg_files + png_files
print(f"共 {len(img_files)} 张图片\n")

# Wine 预览路径
PREVIEW_PATH = os.path.expanduser("~/.wine/drive_c/users/user/Pictures/_label_preview.jpg")

def yolo_to_pixels(cx, cy, bw, bh, w, h):
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    return x1, y1, x2, y2

def pixels_to_yolo(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h
    return cx, cy, bw, bh

def prelabel(img):
    h, w = img.shape[:2]
    resized = cv2.resize(img, (inp_w, inp_h))
    blob = resized.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, 0)
    outs = session.run(None, {input_name: blob})[0][0]
    scores = outs[4:, :]
    best_score = np.max(scores, axis=0)
    best_cls = np.argmax(scores, axis=0)
    mask = best_score > 0.3
    if not mask.any():
        return []
    best_score = best_score[mask]
    best_cls = best_cls[mask]
    outs = outs[:4, mask]
    result = []
    for i in range(outs.shape[1]):
        cx, cy, bw, bh = outs[:, i]
        x1, y1, x2, y2 = yolo_to_pixels(cx, cy, bw, bh, w, h)
        result.append((x1, y1, x2, y2, int(best_cls[i]), float(best_score[i])))
    # 简单 NMS
    keep = []
    for i in sorted(range(len(result)), key=lambda i: result[i][5], reverse=True):
        ok = True
        for j in keep:
            x1 = max(result[i][0], result[j][0])
            y1 = max(result[i][1], result[j][1])
            x2 = min(result[i][2], result[j][2])
            y2 = min(result[i][3], result[j][3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            area_i = (result[i][2]-result[i][0]) * (result[i][3]-result[i][1])
            area_j = (result[j][2]-result[j][0]) * (result[j][3]-result[j][1])
            iou = inter / (area_i + area_j - inter)
            if iou > 0.5:
                ok = False
                break
        if ok:
            keep.append(i)
    return [result[k] for k in keep]

def load_labels(path, w, h):
    base = os.path.splitext(os.path.basename(path))[0]
    lbl_path = os.path.join(LABEL_DIR, base + ".txt")
    if not os.path.exists(lbl_path):
        return []
    boxes = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_pixels(cx, cy, bw, bh, w, h)
            boxes.append((x1, y1, x2, y2, cls_id))
    return boxes

def save_labels(path, boxes_pixel, w, h):
    base = os.path.splitext(os.path.basename(path))[0]
    lbl_path = os.path.join(LABEL_DIR, base + ".txt")
    with open(lbl_path, "w") as f:
        for x1, y1, x2, y2, cls_id in boxes_pixel:
            cx, cy, bw, bh = pixels_to_yolo(x1, y1, x2, y2, w, h)
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def load_image(idx_):
    global img, img_h, img_w, boxes, selected
    path = img_files[idx_]
    img = cv2.imread(path)
    if img is None:
        return None, 0, 0
    img_h, img_w = img.shape[:2]
    boxes = load_labels(path, img_w, img_h)
    if len(boxes) == 0:
        pre = prelabel(img)
        for x1, y1, x2, y2, cls_id, conf in pre:
            boxes.append((x1, y1, x2, y2, cls_id))
    selected = len(boxes) - 1 if boxes else -1
    return img, img_w, img_h

def save_preview():
    """保存带框的预览图到 Wine Pictures"""
    display = img.copy()
    for i, (x1, y1, x2, y2, cls_id) in enumerate(boxes):
        color = (0, 255, 0) if i == selected else (0, 200, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label = f"M{i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (x1, y1-th-4), (x1+tw+4, y1), color, -1)
        cv2.putText(display, label, (x1+2, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    info = f"[{idx+1}/{len(img_files)}] 框:{len(boxes)} 选中:{selected}  文件:{os.path.basename(img_files[idx])}"
    cv2.putText(display, info, (10, img_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.imwrite(PREVIEW_PATH, display)
    return info

def print_status():
    print(f"\n{'='*60}")
    print(f"  [{idx+1}/{len(img_files)}] 文件: {os.path.basename(img_files[idx])}")
    print(f"  框数: {len(boxes)}  选中: {selected}")
    for i, (x1, y1, x2, y2, c) in enumerate(boxes):
        sel = " <<<" if i == selected else ""
        print(f"    M{i}: ({x1},{y1})-({x2},{y2})  {x2-x1}x{y2-y1}{sel}")
    print(f"{'='*60}")

# ---------- 主循环 ----------
idx = 0
img = None
img_h = img_w = 0
boxes = []
selected = -1

load_image(0)
print_status()

while True:
    try:
        cmd = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n退出")
        break

    if not cmd:
        continue
    if cmd == 'quit' or cmd == 'q':
        break

    elif cmd == 'h':
        print("""
命令:
  n        新增框(中心80x80)
  d N      切换选中到 N 号框
  w/a/s/d  微调1px
  W/A/S/D  微调10px
  e        扩大5px
  q        缩小5px
  x        删除选中框
  v        在Wine Desktop预览
  p        上一张
  SPACE    下一张(自动保存)
  s        保存
  r        重新预标注
  h        帮助
  quit     退出
""")

    elif cmd == 'n':  # 画新框 — OpenCV selectROI
        print("在弹窗中拖框选怪物，ENTER确认，ESC取消...")
        # 显示当前图，让用户用 selectROI 画框
        display = img.copy()
        for i, (x1, y1, x2, y2, c) in enumerate(boxes):
            cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,0), 2)
        if len(boxes) > 0 and selected >= 0:
            x1, y1, x2, y2, _ = boxes[selected]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,255), 3)
        cv2.imshow("Drag to draw - ENTER confirm ESC cancel", display)
        rect = cv2.selectROI("Drag to draw - ENTER confirm ESC cancel", display, False, False)
        cv2.destroyAllWindows()
        x, y, w, h = [int(v) for v in rect]
        if w > 20 and h > 20:
            boxes.append((x, y, x+w, y+h, 0))
            selected = len(boxes) - 1
            print(f"新框 {selected}: ({x},{y})-({x+w},{y+h}) 大小 {w}x{h}")
        else:
            print("取消画框或框太小")

    elif cmd.startswith('d '):  # 切换选中
        try:
            n = int(cmd.split()[1])
            if 0 <= n < len(boxes):
                selected = n
            else:
                print(f"框号越界 (0-{len(boxes)-1})")
        except:
            print("用法: d 号码")

    elif cmd in ('w', 'a', 's', 'd', 'W', 'A', 'S', 'D'):
        if selected < 0:
            print("没有选中框")
            continue
        x1, y1, x2, y2, c = boxes[selected]
        step = 1 if cmd.islower() else 10
        if cmd.lower() == 'w':
            boxes[selected] = (x1, max(0,y1-step), x2, max(0,y2-step), c)
        elif cmd.lower() == 's':
            boxes[selected] = (x1, min(img_h-1,y1+step), x2, min(img_h-1,y2+step), c)
        elif cmd.lower() == 'a':
            boxes[selected] = (max(0,x1-step), y1, max(0,x2-step), y2, c)
        elif cmd.lower() == 'd':
            boxes[selected] = (min(img_w-1,x1+step), y1, min(img_w-1,x2+step), y2, c)
        print(f"M{selected} -> ({boxes[selected][0]},{boxes[selected][1]})-({boxes[selected][2]},{boxes[selected][3]})")

    elif cmd == 'e':  # 扩大
        if selected < 0:
            print("没有选中框")
            continue
        x1, y1, x2, y2, c = boxes[selected]
        boxes[selected] = (max(0,x1-5), max(0,y1-5), min(img_w-1,x2+5), min(img_h-1,y2+5), c)
        print(f"M{selected} 扩大")
    elif cmd == 'q':  # 缩小
        if selected < 0:
            print("没有选中框")
            continue
        x1, y1, x2, y2, c = boxes[selected]
        if x2 - x1 > 20 and y2 - y1 > 20:
            boxes[selected] = (min(img_w-1,x1+5), min(img_h-1,y1+5), max(0,x2-5), max(0,y2-5), c)
            print(f"M{selected} 缩小")
        else:
            print("框太小了不能再缩小")

    elif cmd == 'x':  # 删除
        if selected < 0:
            print("没有选中框")
            continue
        boxes.pop(selected)
        selected = min(selected, len(boxes)-1) if boxes else -1
        print(f"删除成功，剩余 {len(boxes)} 个框")

    elif cmd == 'v':  # 预览
        info = save_preview()
        print(f"预览已保存: {PREVIEW_PATH}")
        display = img.copy()
        for i, (x1, y1, x2, y2, c) in enumerate(boxes):
            cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,0), 2)
        if len(boxes) > 0 and selected >= 0:
            x1, y1, x2, y2, _ = boxes[selected]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,255), 3)
        info = f"[{idx+1}/{len(img_files)}] 框:{len(boxes)} 选中:{selected}  文件:{os.path.basename(img_files[idx])}  | 按任意键关"
        cv2.putText(display, info, (10, img_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imwrite(PREVIEW_PATH, display)
        cv2.imshow("Preview - press any key", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif cmd in ('p', 'P'):  # 上一张
        save_labels(img_files[idx], boxes, img_w, img_h)
        idx = max(0, idx - 1)
        load_image(idx)
        print_status()

    elif cmd in (' ', 'n', 'next'):  # 下一张
        # 注意：空格被 input() 吃掉，用 'next' 替代
        if cmd == ' ':
            continue  # input 不会得到空格
        if cmd == 'n':
            continue  # 已被新增框占用
        save_labels(img_files[idx], boxes, img_w, img_h)
        idx += 1
        if idx >= len(img_files):
            print("标注完成！")
            break
        load_image(idx)
        print_status()

    elif cmd == 's':  # 保存
        save_labels(img_files[idx], boxes, img_w, img_h)
        print(f"已保存 | {os.path.basename(img_files[idx])}")
    elif cmd == 'next':
        save_labels(img_files[idx], boxes, img_w, img_h)
        idx += 1
        if idx >= len(img_files):
            print("标注完成！")
            break
        load_image(idx)
        print_status()

    elif cmd == 'r':  # 重新预标
        pre = prelabel(img)
        boxes.clear()
        for x1, y1, x2, y2, cls_id, conf in pre:
            boxes.append((x1, y1, x2, y2, cls_id))
        selected = len(boxes) - 1 if boxes else -1
        print(f"预标注: {len(boxes)} 个框")
        print_status()

    else:
        print("未知命令，输入 h 看帮助")