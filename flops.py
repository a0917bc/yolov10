from ultralytics import YOLOv10
import torch
import sys

# model = YOLOv10.from_pretrained('jameslahm/yolov10n')
# model = YOLOv10('yolov10n.yaml')
# model = YOLOv10('yolov10_egis.yaml')

# model = YOLOv10('yolov10x.yaml')
# model = YOLOv10('yolov10_dev_mob.yaml')
# model = YOLOv10('yolov10_dev_egis.yaml')
# model = YOLOv10('yolov10_dev_egisv2.yaml')
# model = YOLOv10('yolov10_dev_egisv2_mono.yaml')
# model = YOLOv10('yolov10_dev_alcor.yaml')
# model = YOLOv10('yolov10_dev_alcorv2.yaml')
# model = YOLOv10('yolov10_dev_alcorv2_mono.yaml')
# model = YOLOv10('yolov10_dev_egisv3_mono.yaml')
# model = YOLOv10('yolov10_dev_egisv4_mono.yaml')
# model = YOLOv10('yolov10_dev_alcor_vdet.yaml')
# model = YOLOv10('yolov10_eos_dev20.yaml',task="detect")
# model = YOLOv10('yolov10_eos_dumbhead.yaml', task="detect")
# model = YOLOv10('/home/trivenzhou/yolov10_250321/runs/detect/train37/weights/best.pt') # Use system arguments to determine the path
# import pdb;pdb.set_trace()
model = YOLOv10(sys.argv[1])
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
# input = torch.randn(1, 1, 160, 128)
# print(model(input))
exit(0)
CUSTOM_QCFG = torch.quantization.QConfig(
    activation=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.observer.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine
    ),
    weight=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.observer.MovingAverageMinMaxObserver,
        quant_min=-64,
        quant_max=63,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric
    )
)
model.model.train()
model.model.qconfig = CUSTOM_QCFG
torch.quantization.prepare_qat(model.model, inplace=True)
model.model.eval()
torch.quantization.convert(model.model, inplace=True)
#print(model)

"""
a = None
for _ in range(5):
    a = model.predict('/home/share/datasets/COCO/COCO2017/coco/images/val2017/000000001296.jpg')
print(type(a[0]))
"""