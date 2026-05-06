#!/usr/bin/env python3
"""
标注器（轻量版）— OpenCV鼠标画框，原生YOLO格式直接输出
用法: python3 label.py ../monster_data/images/train/
支持的快捷键:
  w         — 保存当前框并切换到下一张
  n         — 下一张（跳过当前）
  p         — 上一张
  方向键     — 微调框位置（上下左右移动选中框1px）
  +/-       — 微调框大小
  d         — 删除当前框
  q         — 保存退出
  Esc       — 不保存退出
"""
import cv2
import numpy as np
import os
import sys
import glob


class YOLOLabeler:
    def __init__(self, img_dir):
        self.img_dir = os.path.abspath(img_dir)
        self.label_dir = os.path.join(os.path.dirname(self.img_dir), "..", "labels", "train")
        os.makedirs(self.label_dir, exist_ok=True)
        # val 目录也建好，以后手动移
        val_label_dir = self.label_dir.replace("/train", "/val")
        os.makedirs(val_label_dir, exist_ok=True)

        self.exts = ("*.jpg", "*.png", "*.jpeg", "*.bmp")
        self.imgs = []
        for ext in self.exts:
            self.imgs.extend(glob.glob(os.path.join(self.img_dir, ext)))
            self.imgs.extend(glob.glob(os.path.join(self.img_dir, ext.upper())))
        self.imgs.sort()
        if not self.imgs:
            print(f"错误: {self.img_dir} 里没有图片")
            sys.exit(1)

        self.idx = 0
        self.class_names = []
        self.class_colors = {}
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.boxes = []        # [(cls_id, x1, y1, x2, y2), ...]
        self.selected = -1     # 选中的框索引
        self.img = None
        self.h, self.w = 0, 0
        self.drag_anchor = None  # (mx, my, bx1, by1, bx2, by2) 开始拖拽时的状态

    def get_next_color(self, idx):
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
            (0, 128, 255), (128, 0, 255), (255, 0, 128), (0, 255, 128),
        ]
        return colors[idx % len(colors)]

    def get_class_id(self, name):
        if name not in self.class_names:
            self.class_names.append(name)
            self.class_colors[name] = self.get_next_color(len(self.class_names) - 1)
        return self.class_names.index(name)

    def load_labels(self, img_path):
        """加载已有的YOLO标签"""
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base + ".txt")
        self.boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = int((xc - bw / 2) * self.w)
                        y1 = int((yc - bh / 2) * self.h)
                        x2 = int((xc + bw / 2) * self.w)
                        y2 = int((yc + bh / 2) * self.h)
                        self.boxes.append((cls, x1, y1, x2, y2))
                        # 确保类名存在
                        while len(self.class_names) <= cls:
                            name = f"class_{len(self.class_names)}"
                            self.class_names.append(name)
                            self.class_colors[name] = self.get_next_color(len(self.class_names) - 1)

    def save_labels(self, img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base + ".txt")
        with open(label_path, "w") as f:
            for cls, x1, y1, x2, y2 in self.boxes:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.w - 1, x2), min(self.h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                xc = (x1 + x2) / 2 / self.w
                yc = (y1 + y2) / 2 / self.h
                bw = (x2 - x1) / self.w
                bh = (y2 - y1) / self.h
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"  已保存: {label_path} ({len(self.boxes)}个框)")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 检查是否点击到已有框
            for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected = i
                    self.drag_anchor = (x, y, x1, y1, x2, y2)
                    return
            # 没点到框，开始画新框
            self.drawing = True
            self.ix, self.iy = x, y
            self.selected = -1

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_box = (self.ix, self.iy, x, y)
            elif self.selected >= 0 and self.drag_anchor:
                dx, dy = x - self.drag_anchor[0], y - self.drag_anchor[1]
                cls = self.boxes[self.selected][0]
                self.boxes[self.selected] = (
                    cls,
                    self.drag_anchor[2] + dx,
                    self.drag_anchor[3] + dy,
                    self.drag_anchor[4] + dx,
                    self.drag_anchor[5] + dy,
                )

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = min(self.ix, x), min(self.iy, y)
                x2, y2 = max(self.ix, x), max(self.iy, y)
                if x2 - x1 > 5 and y2 - y1 > 5:
                    self.boxes.append((0, x1, y1, x2, y2))
                    self.selected = len(self.boxes) - 1
                    print("  [+] 添加框, 按数字键切换类别")
            self.drag_anchor = None

    def run(self):
        cv2.namedWindow("label", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("label", 800, 600)
        cv2.setMouseCallback("label", self.mouse_callback)

        print(f"\n{'='*50}")
        print(f"标注器启动 | 共 {len(self.imgs)} 张图")
        print(f"{'='*50}")
        print("操作:")
        print("  [鼠标拖拽] 画新框")
        print("  [鼠标拖拽已有框] 移动")
        print("  [w] 保存+下一张  [n] 跳过")
        print("  [p] 上一张  [d] 删除选中框")
        print("  [0-9] 切换当前框类别  [q] 保存退出")
        print("  [Esc] 不保存退出")
        print(f"{'='*50}\n")

        while self.idx < len(self.imgs):
            path = self.imgs[self.idx]
            self.img = cv2.imread(path)
            if self.img is None:
                print(f"跳过: {path}")
                self.idx += 1
                continue
            self.h, self.w = self.img.shape[:2]
            self.load_labels(path)

            while True:
                display = self.img.copy()
                # 画所有框
                for i, (cls, x1, y1, x2, y2) in enumerate(self.boxes):
                    color = self.class_colors.get(
                        self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}",
                        (0, 255, 0)
                    ) if cls < len(self.class_names) else (0, 255, 0)
                    thick = 3 if i == self.selected else 2
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, thick)
                    label = f"{cls}:{self.class_names[cls] if cls < len(self.class_names) else cls}"
                    cv2.putText(display, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 画正在拖的框
                if self.drawing and hasattr(self, 'temp_box'):
                    x1, y1, x2, y2 = self.temp_box
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # 信息
                info = f"[{self.idx + 1}/{len(self.imgs)}] {os.path.basename(path)} | boxes:{len(self.boxes)}"
                if self.selected >= 0 and self.selected < len(self.boxes):
                    cls = self.boxes[self.selected][0]
                    info += f" | selected: cls={cls} ({self.class_names[cls] if cls < len(self.class_names) else '?'})"
                cv2.putText(display, info, (10, self.h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                cv2.imshow("label", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('w'):      # 保存+下一张
                    self.save_labels(path)
                    self.idx += 1
                    break
                elif key == ord('n'):    # 跳过
                    self.idx += 1
                    break
                elif key == ord('p'):    # 上一张
                    if self.idx > 0:
                        self.save_labels(path)
                        self.idx -= 1
                    break
                elif key == ord('d'):    # 删除选中框
                    if self.selected >= 0 and self.selected < len(self.boxes):
                        del self.boxes[self.selected]
                        self.selected = -1
                        print("  [-] 删除框")
                elif key == ord('q'):
                    self.save_labels(path)
                    cv2.destroyAllWindows()
                    print(f"\n标注完成! 共标注 {self.idx + 1} 张")
                    return
                elif key == 27:  # Esc
                    cv2.destroyAllWindows()
                    print("\n已取消，未保存当前图片")
                    return
                elif ord('0') <= key <= ord('9'):
                    cls = key - ord('0')
                    if self.selected >= 0 and self.selected < len(self.boxes):
                        x1, y1, x2, y2 = self.boxes[self.selected][1:]
                        self.boxes[self.selected] = (cls, x1, y1, x2, y2)
                        name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                        print(f"  类别切换为 {cls}:{name}")

        cv2.destroyAllWindows()
        print("\n所有图片标注完成!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 label.py <图片目录>")
        print("示例: python3 label.py monster_data/images/train/")
        sys.exit(1)
    YOLOLabeler(sys.argv[1]).run()