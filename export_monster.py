#!/usr/bin/env python3
"""一键导出训练好的模型为ONNX"""
import subprocess
import os
import sys
import glob

BASE = os.path.dirname(os.path.abspath(__file__))

# 找最新的 best.pt
pts = sorted(glob.glob(os.path.join(BASE, "monster_runs", "detect", "*", "weights", "best.pt")),
             key=os.path.getmtime, reverse=True)
if not pts:
    pts = sorted(glob.glob(os.path.join(BASE, "runs", "detect", "*", "weights", "best.pt")),
                 key=os.path.getmtime, reverse=True)

if not pts:
    print("错误: 没找到 best.pt 文件")
    sys.exit(1)

best_pt = pts[0]
print(f"使用模型: {best_pt}")

export_cmd = [
    "yolo", "export",
    f"model={best_pt}",
    "format=onnx",
    "simplify=True",
    "imgsz=640",
]
print(f"导出: {' '.join(export_cmd)}")
subprocess.run(export_cmd, cwd=BASE)

# 找导出的onnx
onnx_files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(best_pt)), "*.onnx"))
for f in onnx_files:
    dst = os.path.join(BASE, "monster_model.onnx")
    if os.path.abspath(f) != os.path.abspath(dst):
        import shutil
        shutil.copy2(f, dst)
        print(f"已复制到: {dst}")
        break