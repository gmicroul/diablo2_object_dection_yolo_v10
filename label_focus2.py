#!/usr/bin/env python3
"""
专注标注器 v2 — 纯终端模式版
- 自动预标注（YOLO ONNX）
- 显示图片和框在终端
- 只用键盘操作

用法:
  python3 label_focus2.py

按键:
  n/SPACE  : 下一张
  p        : 上一张
  d        : 删除选中框
  r        : 刷新（重新预标注）
  s        : 保存当前（不切图）
  q/ESC    : 退出
  
  w/a/s/d  : 微调选中框 1px (上下左右)
  W/A/S/D  : 微调 10px（Shift+字母）
  
  j/k      : 切换选中框（下/上）
  +/=      : 增大选中框（向外扩2px）
  -/_      : 缩小选中框（向内缩2px）
  
状态行:
  图片序号/总数 | 框数 | 选中框 | 坐标 | 截图保存
"""

import os, sys, glob, cv2, numpy as np, textwrap

# ---------- 配置 ----------
IMG_DIR = "monster_data/images/train2"
LABEL_DIR = "monster_data/labels/train2"
os.makedirs(LABEL_DIR, exist_ok=True)

MODEL_PATH = "runs/detect/train/weights/best.onnx"
FALLBACK_MODEL = "yolov8n_relu_20class_zq.onnx"
CLASS_14_MODEL = "yolov8n_relu_20class_zq.onnx"

# 模型加载（ONNX Runtime）
import onnxruntime as ort
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.enable_mem_pattern = False
sess_opts.intra_op_num_threads = 4

def load_onnx(path):
    if not os.path.exists(path):
        return None
    try:
        s = ort.InferenceSession(path, sess_opts, providers=['CPUExecutionProvider'])
        return s
    except:
        return None

# 优先用训练后的模型，没有就用预标注模型
session_main = load_onnx(MODEL_PATH)
session_fb = load_onnx(FALLBACK_MODEL)
session_14 = load_onnx(CLASS_14_MODEL)
session = session_main or session_fb or session_14

if session is None:
    print("[ERROR] 找不到任何 ONNX 模型！")
    print(f"  试过: {MODEL_PATH}")
    print(f"  试过: {FALLBACK_MODEL}")
    print(f"  试过: {CLASS_14_MODEL}")
    sys.exit(1)

input_name = session.get_inputs()[0].name
inp_h, inp_w = session.get_inputs()[0].shape[2:4]

print(f"模型: {session._model_path}")
print(f"输入: {inp_w}x{inp_h}")

# ---------- 图片列表 ----------
jpg_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
png_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
img_files = jpg_files + png_files
print(f"共 {len(img_files)} 张图片")

# ---------- 标注工具 ----------
WINDOW = "Label Focus - [S]ave [Space]Next [Z]Undo [R]efresh [Q]uit"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 900, 675)

# 撤销栈：记录每张图片被切走时的标注状态
undo_stack = {}

# 全局状态
idx = 0
boxes = []        # [(x1,y1,x2,y2,conf), ...]
drawing = False
ix, iy = -1, -1
selected = -1
drag_mode = None  # "move" | "resize_tl" | "resize_tr" | "resize_bl" | "resize_br" | "resize_l" | "resize_r" | "resize_t" | "resize_b"
drag_off_x = 0
drag_off_y = 0

def yolo_to_pixels(yolo_boxes, w, h):
    """yolo归一化 -> 像素坐标 xyxy"""
    res = []
    for yb in yolo_boxes:
        cls_id, cx, cy, bw, bh = yb[:5]
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        res.append((x1, y1, x2, y2, cls_id))
    return res

def pixels_to_yolo(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = abs(x2 - x1) / w
    bh = abs(y2 - y1) / h
    return (cx, cy, bw, bh)

def prelabel(img):
    """用ONNX做预标注"""
    h, w = img.shape[:2]
    # resize
    resized = cv2.resize(img, (inp_w, inp_h))
    blob = resized.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, 0)
    
    outs = session.run(None, {input_name: blob})[0]  # (1,84,8400)
    outs = outs[0]  # (84,8400)
    
    # 后处理
    scores = outs[4:, :]  # (80,8400) 或 (20,8400)
    best_score = np.max(scores, axis=0)
    best_cls = np.argmax(scores, axis=0)
    
    # 过滤
    CONF = 0.3
    mask = best_score > CONF
    if not mask.any():
        return []
    
    best_score = best_score[mask]
    best_cls = best_cls[mask]
    outs = outs[:4, mask]  # (4,N)
    
    # 解算坐标 (cx,cy,w,h) -> xyxy
    scale_x = w / inp_w
    scale_y = h / inp_h
    result = []
    for i in range(outs.shape[1]):
        cx, cy, bw, bh = outs[:, i]
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cls_id = int(best_cls[i])
        conf = float(best_score[i])
        result.append((x1, y1, x2, y2, cls_id, conf))
    
    # NMS
    if result:
        keep = nms_boxes([r[:4] for r in result], [r[5] for r in result], 0.5)
        result = [result[k] for k in keep]
    
    return result

def nms_boxes(boxes, scores, iou_thresh=0.5):
    import math
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while indices:
        i = indices.pop(0)
        keep.append(i)
        to_remove = []
        for j in indices:
            x1 = max(boxes[i][0], boxes[j][0])
            y1 = max(boxes[i][1], boxes[j][1])
            x2 = min(boxes[i][2], boxes[j][2])
            y2 = min(boxes[i][3], boxes[j][3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            area_i = (boxes[i][2]-boxes[i][0]) * (boxes[i][3]-boxes[i][1])
            area_j = (boxes[j][2]-boxes[j][0]) * (boxes[j][3]-boxes[j][1])
            iou = inter / (area_i + area_j - inter)
            if iou > iou_thresh:
                to_remove.append(j)
        indices = [j for j in indices if j not in to_remove]
    return keep

def load_labels(img_path, img_w, img_h):
    """加载已有标注"""
    base = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(LABEL_DIR, base + ".txt")
    if not os.path.exists(lbl_path):
        return []
    boxes = []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw/2) * img_w)
            y1 = int((cy - bh/2) * img_h)
            x2 = int((cx + bw/2) * img_w)
            y2 = int((cy + bh/2) * img_h)
            boxes.append((x1, y1, x2, y2, cls_id))
    return boxes

def save_labels(img_path, boxes_pixel, img_w, img_h):
    base = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(LABEL_DIR, base + ".txt")
    with open(lbl_path, "w") as f:
        for x1, y1, x2, y2, cls_id in boxes_pixel:
            cx, cy, bw, bh = pixels_to_yolo(x1, y1, x2, y2, img_w, img_h)
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def clamp_rect(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return x1, y1, x2, y2

def load_image(idx_):
    global img, img_h, img_w, boxes, selected
    path = img_files[idx_]
    img = cv2.imread(path)
    if img is None:
        print(f"无法读取: {path}")
        return None, 0, 0
    img_h, img_w = img.shape[:2]
    # 加载已有标注
    boxes = load_labels(path, img_w, img_h)
    # 如果没有任何标注，做预标注
    if len(boxes) == 0:
        pre = prelabel(img)
        for x1, y1, x2, y2, cls_id, conf in pre:
            boxes.append((x1, y1, x2, y2, cls_id))
    selected = -1
    return img, img_w, img_h

def draw_image():
    """绘制画面"""
    display = img.copy()
    h, w = display.shape[:2]
    
    for i, (x1, y1, x2, y2, cls_id) in enumerate(boxes):
        color = (0, 255, 0) if i == selected else (0, 200, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        # 序号
        label = f"M{i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display, (x1, y1-th-4), (x1+tw+4, y1), color, -1)
        cv2.putText(display, label, (x1+2, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # 选中时显示8个控制点
        if i == selected:
            pts = [(x1,y1), (x2,y1), (x1,y2), (x2,y2),  # 四角
                   ((x1+x2)//2,y1), ((x1+x2)//2,y2),     # 上下中
                   (x1,(y1+y2)//2), (x2,(y1+y2)//2)]     # 左右中
            for px, py in pts:
                cv2.circle(display, (px, py), 4, (0, 255, 255), -1)
    
    if drawing and ix >= 0 and iy >= 0:
        cv2.rectangle(display, (ix, iy), (mx, my), (255, 0, 0), 2)
    
    # 信息栏
    info = f"[{idx+1}/{len(img_files)}] 怪物框: {len(boxes)} 选中: {selected}"
    h_info = 30
    info_bar = np.zeros((h_info, w, 3), dtype=np.uint8)
    cv2.putText(info_bar, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    display = np.vstack([display, info_bar])
    
    cv2.imshow(WINDOW, display)

def get_box_at(x, y, threshold=10):
    """找点击位置的框"""
    for i in reversed(range(len(boxes))):
        x1, y1, x2, y2, _ = boxes[i]
        # 扩展点击宽容
        if x1 - threshold <= x <= x2 + threshold and y1 - threshold <= y <= y2 + threshold:
            return i
    return -1

def get_resize_handle(x, y, box_idx, threshold=10):
    """检测是否点击到缩放手柄"""
    if box_idx < 0:
        return None
    x1, y1, x2, y2, _ = boxes[box_idx]
    handles = {
        "resize_tl": (x1, y1),
        "resize_tr": (x2, y1),
        "resize_bl": (x1, y2),
        "resize_br": (x2, y2),
        "resize_t": ((x1+x2)//2, y1),
        "resize_b": ((x1+x2)//2, y2),
        "resize_l": (x1, (y1+y2)//2),
        "resize_r": (x2, (y1+y2)//2),
    }
    for mode, (hx, hy) in handles.items():
        if abs(x - hx) <= threshold and abs(y - hy) <= threshold:
            return mode
    return None

def do_next():
    global idx, boxes, drawing, selected, mx, my, ix, iy
    # 保存当前
    if img_files and idx < len(img_files):
        path = img_files[idx]
        save_labels(path, boxes, img_w, img_h)
        undo_stack[idx] = [b[:] for b in boxes]
    # 下一张
    idx += 1
    if idx >= len(img_files):
        print("标注完成！")
        return False
    img_data = load_image(idx)
    if img_data[0] is None:
        return False
    drawing = False
    selected = -1
    return True

def do_undo():
    global boxes
    if idx in undo_stack:
        boxes = [b[:] for b in undo_stack[idx]]
        print(f"已撤销回 {len(boxes)} 个框")
    else:
        boxes = []
        print("无可撤销")

# ---------- 鼠标回调（简化稳定版） ----------
def mouse_cb(event, x, y, flags, param):
    global ix, iy, drawing, mx, my, boxes, selected, drag_mode, drag_off_x, drag_off_y
    
    if y >= img_h:
        mouse_cb.img_h = img_h
        return
    
    mx, my = x, y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 检查是否点击到已有框
        hit = -1
        for i in range(len(boxes)):
            x1, y1, x2, y2, _ = boxes[i]
            if x1 <= x <= x2 and y1 <= y <= y2:
                hit = i
                break
        
        if hit >= 0:
            # 选中已有框
            selected = hit
            # 检查是否点到边缘（8px内）— 缩放模式
            x1, y1, x2, y2, _ = boxes[hit]
            edge = 8
            on_edge = (abs(x-x1) <= edge or abs(x-x2) <= edge or 
                       abs(y-y1) <= edge or abs(y-y2) <= edge)
            if on_edge:
                drag_mode = "resize"
                drag_off_x = x
                drag_off_y = y
            else:
                # 移动模式
                drag_mode = "move"
                drag_off_x = x - x1
                drag_off_y = y - y1
        else:
            # 空白区域 — 画新框
            selected = -1
            ix, iy = x, y
            drawing = True
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            mx, my = x, y
        elif drag_mode == "move" and selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            bw = x2 - x1
            bh = y2 - y1
            nx = max(0, min(img_w-bw, x - drag_off_x))
            ny = max(0, min(img_h-bh, y - drag_off_y))
            boxes[selected] = (nx, ny, nx+bw, ny+bh, c)
        elif drag_mode == "resize" and selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            dx = x - drag_off_x
            dy = y - drag_off_y
            # 判断靠近哪边
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            edge = 8
            near_l = abs(drag_off_x - x1) <= edge
            near_r = abs(drag_off_x - x2) <= edge
            near_t = abs(drag_off_y - y1) <= edge
            near_b = abs(drag_off_y - y2) <= edge
            if near_l: x1 += dx
            if near_r: x2 += dx
            if near_t: y1 += dy
            if near_b: y2 += dy
            x1, y1, x2, y2 = clamp_rect(x1, y1, x2, y2, img_w, img_h)
            boxes[selected] = (x1, y1, x2, y2, c)
            drag_off_x, drag_off_y = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            x1, y1 = min(ix, mx), min(iy, my)
            x2, y2 = max(ix, mx), max(iy, my)
            if abs(x2-x1) > 20 and abs(y2-y1) > 20:
                boxes.append((x1, y1, x2, y2, 0))
                selected = len(boxes) - 1
            ix, iy = -1, -1
        else:
            drag_mode = None

mouse_cb.img_h = 0

cv2.setMouseCallback(WINDOW, mouse_cb)

# ---------- 主循环 ----------
load_image(0)
while True:
    draw_image()
    key = cv2.waitKey(20) & 0xFF
    
    if key == ord('q') or key == 27:  # q / ESC
        break
    elif key == ord(' '):  # 下一张
        if not do_next():
            break
    elif key == ord('p'):  # 上一张
        # 先保存当前
        save_labels(img_files[idx], boxes, img_w, img_h)
        undo_stack[idx] = [b[:] for b in boxes]
        idx = max(0, idx - 1)
        load_image(idx)
        selected = -1 if not boxes else (selected if selected < len(boxes) else len(boxes)-1)
    elif key == ord('s'):  # 保存
        save_labels(img_files[idx], boxes, img_w, img_h)
        undo_stack[idx] = [b[:] for b in boxes]
        print(f"已保存 {img_files[idx]}")
    elif key == ord('z'):  # 撤销
        do_undo()
    elif key == ord('r'):  # 重新预标注
        h, w = img.shape[:2]
        pre = prelabel(img)
        boxes.clear()
        for x1, y1, x2, y2, cls_id, conf in pre:
            boxes.append((x1, y1, x2, y2, cls_id))
        selected = -1
        print(f"预标注: {len(boxes)} 个框")

    # 删除键 — 支持多个键码
    elif key in (127, 8, 255, ord('x'), ord('X'), ord('.')):  # DEL, Backspace, 其他, x, X, .
        if selected >= 0:
            boxes.pop(selected)
            selected = min(selected, len(boxes)-1) if boxes else -1

    # 手动添加新框 — 用 selectROI
    elif key == ord('n'):  # 进入"新框"模式
        cv2.setMouseCallback(WINDOW, lambda *a: None)  # 临时取消鼠标回调
        rect = cv2.selectROI(WINDOW, img, False, False)  # (x,y,w,h)
        x, y, w, h = [int(v) for v in rect]
        if w > 20 and h > 20:
            boxes.append((x, y, x+w, y+h, 0))
            selected = len(boxes) - 1
            print(f"新框: ({x},{y})-({x+w},{y+h})")
        cv2.setMouseCallback(WINDOW, mouse_cb)  # 恢复鼠标回调
    elif key == ord('e'):  # 扩大选中框（向外扩5px）
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (max(0,x1-5), max(0,y1-5), min(img_w-1,x2+5), min(img_h-1,y2+5), c)
    elif key == ord('q'):  # 缩小选中框（向内缩5px）
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            if x2 - x1 > 20 and y2 - y1 > 20:
                boxes[selected] = (min(img_w-1,x1+5), min(img_h-1,y1+5), max(0,x2-5), max(0,y2-5), c)

    elif key == ord('w'):     # 上移
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (x1, max(0,y1-1), x2, max(0,y2-1), c)
    elif key == 87:          # W（大写）= Shift+W，上移10px
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (x1, max(0,y1-10), x2, max(0,y2-10), c)
    elif key == ord('a'):     # 左移
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (max(0,x1-1), y1, max(0,x2-1), y2, c)
    elif key == 65:          # A（大写）= Shift+A，左移10px
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (max(0,x1-10), y1, max(0,x2-10), y2, c)
    elif key == ord('d'):     # 右移
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (min(img_w-1,x1+1), y1, min(img_w-1,x2+1), y2, c)
    elif key == 68:          # D（大写）= Shift+D，右移10px
        if selected >= 0:
            x1, y1, x2, y2, c = boxes[selected]
            boxes[selected] = (min(img_w-1,x1+10), y1, min(img_w-1,x2+10), y2, c)

cv2.destroyAllWindows()
print("标注结束")