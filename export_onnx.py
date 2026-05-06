from ultralytics import YOLO
model = YOLO('/home/user/diablo2_object_dection_yolo_v10/runs/detect/monster_runs/monster_finetune/weights/best.pt')
model.export(format='onnx', simplify=True)
print("ONNX导出完成")