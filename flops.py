from ultralytics import YOLOv10
import torch

#model = YOLOv10.from_pretrained('jameslahm/yolov10n')
model = YOLOv10('yolov10n.yaml')
#model = YOLOv10('yolov10x.yaml')
#model = YOLOv10('yolov10_dev_mob.yaml')
#model = YOLOv10('yolov10_dev_egis.yaml')
#model = YOLOv10('yolov10_dev_egisv2.yaml')
#model = YOLOv10('yolov10_dev_egisv2_mono.yaml')
#model = YOLOv10('yolov10_dev_alcor.yaml')
#model = YOLOv10('yolov10_dev_alcorv2.yaml')
#model = YOLOv10('yolov10_dev_alcorv2_mono.yaml')
#model = YOLOv10('yolov10_dev_egisv3_mono.yaml')
#model = YOLOv10('yolov10_dev_egisv4_mono.yaml')
#model = YOLOv10('yolov10_dev_alcor_vdet.yaml')
"""
wgt = torch.load('/home/kylai/llm/yolov10/runs/detect/train137/weights/last.pt')
for k in wgt['model'].model.state_dict():
    print(k)
exit(0)
for k in model.model.model.state_dict():
    print(k)
exit(0)"""
model.model.model[-1].export = True
model.model.model[-1].format = 'onnx'
del model.model.model[-1].cv2
del model.model.model[-1].cv3
model.fuse()
print(model)
"""
a = None
for _ in range(5):
    a = model.predict('/home/share/datasets/COCO/COCO2017/coco/images/val2017/000000001296.jpg')
print(type(a[0]))
"""