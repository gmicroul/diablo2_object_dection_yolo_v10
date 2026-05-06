#!/usr/bin/env python3
"""
极简YOLO标注器 — 只用OpenCV
一张一张标，标完自动存YOLO格式

用法:
  cd /home/user/diablo2_object_dection_yolo_v10
  python3 label_simple.py

快捷键:
  R       : 画框模式（鼠标拖框，松手即标）
  X       : 删除选中框
  W/A/S/D : 微调选中框 1px
  E       : 扩大 5px
  Q       : 缩小 5px
  N       : 下一张（自动保存）
  P       : 上一张
  S       : 保存
  ESC/Q   : 退出
"""

import os, sys, glob, cv2, numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE, "monster_data/images/train2")
LABEL_DIR = os.path.join(BASE, "monster_data/labels/train2")
os.makedirs(LABEL_DIR, exist_ok=True)

img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
print(f"图片: {len(img_files)} 张")

# 状态
idx = 0
boxes = []
selected = -1
drawing = False
ix, iy = -1, -1
img_h, img_w = 0, 0
img = None
preview_only = False  # 预览模式不响应按键

WIN = "YOLO Label - [R]draw [N]next [P]prev [S]save [X]del [Q]uit"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 900, 675)

def load_img(i):
    global img, img_h, img_w, boxes, selected
    path = img_files[i]
    img = cv2.imread(path)
    if img is None:
        return False
    img_h, img_w = img.shape[:2]
    base = os.path.splitext(os.path.basename(path))[0]
    lbl_path = os.path.join(LABEL_DIR, base + ".txt")
    boxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw/2) * img_w)
                y1 = int((cy - bh/2) * img_h)
                x2 = int((cx + bw/2) * img_w)
                y2 = int((cy + bh/2) * img_h)
                boxes.append((x1, y1, x2, y2, int(parts[0])))
    selected = len(boxes) - 1 if boxes else -1
    return True

def save_img(i):
    path = img_files[i]
    base = os.path.splitext(os.path.basename(path))[0]
    lbl_path = os.path.join(LABEL_DIR, base + ".txt")
    with open(lbl_path, "w") as f:
        for x1, y1, x2, y2, cls_id in boxes:
            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            bw = abs(x2 - x1) / img_w
            bh = abs(y2 - y1) / img_h
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def redraw():
    display = img.copy()
    for i, (x1, y1, x2, y2, cls_id) in enumerate(boxes):
        color = (0,255,0) if i == selected else (0,200,0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        label = f"M{i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (x1, y1-th-4), (x1+tw+4, y1), color, -1)
        cv2.putText(display, label, (x1+2, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if i == selected:
            for px, py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2),((x1+x2)//2,y1),((x1+x2)//2,y2),(x1,(y1+y2)//2),(x2,(y1+y2)//2)]:
                cv2.circle(display, (px, py), 4, (0,255,255), -1)
    if drawing and ix>=0 and iy>=0:
        cv2.rectangle(display, (ix, iy), (mx, my), (255,0,0), 2)
    info = f"[{idx+1}/{len(img_files)}] 框:{len(boxes)} 选中:{selected}  {os.path.basename(img_files[idx])}"
    cv2.putText(display, info, (10, img_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.imshow(WIN, display)

# 鼠标事件
mx, my = 0, 0
def mouse_cb(event, x, y, flags, param):
    global ix, iy, drawing, mx, my, boxes, selected
    if y >= img_h: return
    mx, my = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        hit = -1
        for i, (bx1, by1, bx2, by2, _) in enumerate(boxes):
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                hit = i; break
        if hit >= 0:
            selected = hit
        else:
            selected = -1
            ix, iy = x, y
            drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        mx, my = x, y
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        x1, y1 = min(ix, mx), min(iy, my)
        x2, y2 = max(ix, mx), max(iy, my)
        if abs(x2-x1) > 20 and abs(y2-y1) > 20:
            boxes.append((x1, y1, x2, y2, 0))
            selected = len(boxes) - 1
        ix, iy = -1, -1

cv2.setMouseCallback(WIN, mouse_cb)

load_img(0)
redraw()

while True:
    key = cv2.waitKey(20) & 0xFF
    
    if key in (ord('q'), 27):
        break
    elif key == ord('n'):  # 下一张
        save_img(idx)
        idx += 1
        if idx >= len(img_files):
            print("完成！")
            break
        load_img(idx)
        cv2.setMouseCallback(WIN, mouse_cb)
        redraw()
    elif key == ord('p'):  # 上一张
        save_img(idx)
        idx = max(0, idx - 1)
        load_img(idx)
        cv2.setMouseCallback(WIN, mouse_cb)
        redraw()
    elif key == ord('s'):  # 保存
        save_img(idx)
    elif key == ord('x'):  # 删除
        if selected >= 0:
            boxes.pop(selected)
            selected = min(selected, len(boxes)-1) if boxes else -1
            redraw()
    elif key == ord('w'):  # 上移
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (x1, max(0,y1-1), x2, max(0,y2-1), c)
            redraw()
    elif key == ord('s') and not (cv2.getWindowProperty(WIN, 0) < 0):
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (x1, min(img_h-1,y1+1), x2, min(img_h-1,y2+1), c)
            redraw()
    elif key == ord('a'):
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (max(0,x1-1), y1, max(0,x2-1), y2, c)
            redraw()
    elif key == ord('d'):
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (min(img_w-1,x1+1), y1, min(img_w-1,x2+1), y2, c)
            redraw()
    elif key == ord('e'):  # 扩大
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (max(0,x1-5), max(0,y1-5), min(img_w-1,x2+5), min(img_h-1,y2+5), c)
            redraw()
    elif key == ord('q'):  # 缩小
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            if x2-x1 > 20 and y2-y1 > 20:
                boxes[selected] = (min(img_w-1,x1+5), min(img_h-1,y1+5), max(0,x2-5), max(0,y2-5), c)
                redraw()
    else:
        redraw()

cv2.destroyAllWindows()
print("结束")