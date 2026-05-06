#!/usr/bin/env python3
"""把train2的图片全部复制到Wine Pictures目录"""
import os, shutil, glob

src = "/home/user/diablo2_object_dection_yolo_v10/monster_data/images/train2"
dst = os.path.expanduser("~/.wine/drive_c/users/user/Pictures/train2")
os.makedirs(dst, exist_ok=True)

for f in sorted(glob.glob(os.path.join(src, "*.jpg"))):
    shutil.copy2(f, dst)

print(f"已复制 {len(os.listdir(dst))} 张到 {dst}")