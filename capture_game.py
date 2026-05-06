#!/usr/bin/env python3
"""
自动截暗黑2游戏窗口并保存，每隔 0.5s 自动截一张。
按 Ctrl+C 结束。
"""
import subprocess
import os
import time
import sys
import struct
import numpy as np
import cv2

# 设置 DISPLAY
os.environ["DISPLAY"] = ":0"

DST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "monster_data", "images", "train")
os.makedirs(DST, exist_ok=True)
os.makedirs(DST.replace("/train", "/val"), exist_ok=True)

WID = "48234503"  # Wine Desktop 窗口 ID


def xwd_to_cv2(data):
    width = struct.unpack(">I", data[16:20])[0]
    height = struct.unpack(">I", data[20:24])[0]
    bpl = struct.unpack(">I", data[48:52])[0]
    pix_start = len(data) - bpl * height
    pix_data = data[pix_start:]

    arr = np.frombuffer(pix_data, dtype=np.uint8).reshape(height, bpl)
    img = arr[:, :width * 4].reshape(height, width, 4)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


existing = {f for f in os.listdir(DST) if f.endswith((".jpg", ".png"))}
count = len(existing)
print(f"已有 {count} 张截图")
print("开始自动截图！每0.5秒截一张。")
print("你去操作游戏走动/打怪，我来截图。")
print("按 Ctrl+C 停止...")
print()

try:
    while True:
        r = subprocess.run(
            ["xwd", "-id", WID],
            capture_output=True, check=True,
            env={**os.environ, "DISPLAY": ":0"}
        )
        fname = f"d2_{count:04d}.jpg"
        path = os.path.join(DST, fname)

        img = xwd_to_cv2(r.stdout)
        cv2.imwrite(path, img)
        count += 1
        sys.stdout.write(f"\r  已保存: {fname} (共{count}张)")
        sys.stdout.flush()
        time.sleep(0.5)

except KeyboardInterrupt:
    print(f"\n\n完成！共截取 {count} 张图片到 {DST}")
except Exception as e:
    print(f"\n错误: {e}")