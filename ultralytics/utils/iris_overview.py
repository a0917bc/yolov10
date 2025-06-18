import torch.nn as nn
import torch
from timm.layers import DropPath
import cv2
import numpy as np
import time
import tqdm


W = 160
H = 160


CUSTOM_QCFG = torch.quantization.QConfig(
    activation=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.observer.MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    ),
    weight=torch.quantization.FakeQuantize.with_args(
        observer=torch.quantization.observer.MovingAverageMinMaxObserver,
        quant_min=-64,
        quant_max=63,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
    ),
)


def stack_mono(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.stack([img, img, img], axis=2)
    return img


def tensor_yuv422(img):
    r, g, b = img[0, :, :].int(), img[1, :, :].int(), img[2, :, :].int()
    y = torch.bitwise_right_shift(128 + 55 * r + 183 * g + 18 * b, 8)
    u = -29 * r + -99 * g + 128 * b + 32896
    v = 128 * r + -116 * g + -12 * b + 32896
    uv = torch.zeros_like(r)
    uv[:, 0::2] = torch.bitwise_right_shift(u[:, 0::2] + u[:, 1::2], 9)
    uv[:, 1::2] = torch.bitwise_right_shift(v[:, 0::2] + v[:, 1::2], 9)

    y = y.clip(0, 255).to(torch.uint8)
    uv = uv.clip(0, 255).to(torch.uint8)
    yuv422_img = torch.stack((y, uv), dim=0)  # Shape: (2, H, W)
    return yuv422_img


def loadnresize_image(img):
    h, w, _ = img.shape
    r = min(H / h, W / w)
    rh = int(h * r)
    rw = int(w * r)
    img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(
        img,
        (H + 1 - rh) // 2,
        (H - rh) // 2,
        (W + 1 - rw) // 2,
        (W - rw) // 2,
        cv2.BORDER_CONSTANT,
        0,
    )  # TBLR to 320,256
    return img


def preprocess_image(img, mono, yuv, device):
    # https://github.com/THU-MIG/yolov10/blob/main/ultralytics/models/yolo/detect/val.py#L45
    if mono:
        img = stack_mono(img)
    img = img.transpose(2, 0, 1)  # C,H,W
    img = np.ascontiguousarray(img[::-1])
    img = torch.from_numpy(img)
    if yuv:
        img = tensor_yuv422(img)
    img = img.to(device, non_blocking=True)
    img = img.float() / 256
    return img


class ConvBNReLU(nn.Sequential):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, q=False
    ):
        padding = (kernel_size - 1) // 2
        # print(in_planes, out_planes, kernel_size, stride, groups)
        if q:
            super(ConvBNReLU, self).__init__(
                torch.quantization.QuantStub(),
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size,
                    stride,
                    padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes, momentum=0.1),
                nn.ReLU(inplace=False),
            )
        else:
            super(ConvBNReLU, self).__init__(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size,
                    stride,
                    padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes, momentum=0.1),
                nn.ReLU(inplace=False),
            )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, drop_path=0):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, momentum=0.1),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.skip_add = nn.quantized.FloatFunctional()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.drop_path(self.conv(x)))
        else:
            return self.conv(x)


class DWConvbnrelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x = self.conv(x)
        return x


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.qf = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.qf.cat(x, self.d)


class EgisDetect(nn.Module):
    def __init__(self, ch, class_num, conf=0.45, h=8, w=10):
        super().__init__()
        kernel_size = 3
        self.regression = 4
        self.class_num = class_num
        self.one2one_cv2 = nn.ModuleList(
            nn.Sequential(
                DWConvbnrelu(x, ch[0], kernel_size),
                DWConvbnrelu(ch[0], ch[0], kernel_size),
                nn.Conv2d(ch[0], self.regression, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = nn.ModuleList(
            nn.Sequential(
                DWConvbnrelu(x, ch[0], kernel_size),
                DWConvbnrelu(ch[0], ch[0], kernel_size),
                nn.Conv2d(ch[0], self.class_num, 1),
            )
            for x in ch
        )
        self.classifier = nn.Linear(40, 16)
        # Consider the fixed size only
        self.anchors, self.strides = (
            x.transpose(0, 1) for x in make_anchors(h, w, 3, 0.5)
        )
        self.max_det = 16
        self.conf = conf
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        classification = self.dequant(self.classifier(x[0].mean([2, 3])))
        cls = []
        box = []
        B = x[0].shape[0]
        for i in range(3):
            box.append(
                self.dequant(self.one2one_cv2[i](x[i])).view(B, self.regression, -1)
            )
            cls.append(
                self.dequant(self.one2one_cv3[i](x[i])).view(B, self.class_num, -1)
            )

        box = torch.cat(box, 2)
        cls = torch.cat(cls, 2)
        y = self.postprocess(box, cls)
        return y, torch.argmax(classification, dim=1)

    def postprocess(self, boxes, cls):
        boxes = decode_boxes(boxes, self.anchors.unsqueeze(0)) * self.strides
        cls = cls.sigmoid()

        # Determine Conf
        max_scores = cls.amax(dim=1)
        max_scores, index = torch.topk(max_scores, self.max_det, dim=1)
        index = index.unsqueeze(1)
        boxes = torch.gather(boxes, dim=2, index=index.repeat(1, boxes.shape[1], 1))
        cls = torch.gather(cls, dim=2, index=index.repeat(1, cls.shape[1], 1))

        # Determine Label
        scores, index = torch.topk(cls.flatten(1), self.max_det, dim=-1)
        scores = scores.unsqueeze(1)
        labels = index % self.class_num
        labels = labels.unsqueeze(1)
        index = index // self.class_num
        index = index.unsqueeze(1)
        boxes = boxes.gather(dim=2, index=index.repeat(1, boxes.shape[1], 1))

        # Conf Threshold
        mask = scores > self.conf
        boxes = boxes[mask.repeat(1, boxes.shape[1], 1)]
        boxes = boxes.view(1, 4, -1)
        cls = cls[mask.repeat(1, cls.shape[1], 1)]
        scores = scores[mask]
        labels = labels[mask]

        return boxes, cls, scores, labels


class InvertedResidualalcor(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, drop_path=0):
        super(InvertedResidualalcor, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, q=False))

        layers.extend(
            [
                ConvBNReLU(
                    hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, q=False
                ),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, momentum=0.1),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.skip_add = nn.quantized.FloatFunctional()
        # self.norm = DynamicTanh(c = oup)

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.drop_path(self.conv(x)))
            # return self.norm(self.skip_add.add(x, self.drop_path(self.conv(x))))

        else:
            x = self.conv(x)
            return x


class EgisDetect2(nn.Module):
    def __init__(self, nc=8, ch=[40]):  # ch depends on the model, dev202122...
        super().__init__()
        self.max_det = 16
        bk_out = ch[0]
        self.regression = 1
        self.up4x = torch.nn.Sequential(
            nn.Upsample(scale_factor=2),
            InvertedResidualalcor(ch[0], 32, 1, 4),
            nn.Upsample(scale_factor=2),
            InvertedResidualalcor(32, 32, 1, 2),
            nn.Upsample(scale_factor=2),
            InvertedResidualalcor(32, 24, 1, 2),
        )
        kernel_size = 3
        ch_lmt = 64
        ch = [24]

        self.one2one_cv2 = nn.ModuleList(
            nn.Sequential(
                DWConvbnrelu(x, ch_lmt, kernel_size),
                DWConvbnrelu(ch_lmt, ch_lmt, kernel_size),
                nn.Conv2d(ch_lmt, 4 * self.regression, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = nn.ModuleList(
            nn.Sequential(
                DWConvbnrelu(x, ch_lmt, kernel_size),
                DWConvbnrelu(ch_lmt, ch_lmt, kernel_size),
                nn.Conv2d(ch_lmt, nc, 1),
            )
            for x in ch
        )

        self.classifier = torch.nn.Sequential(
            nn.Linear(bk_out, 128), nn.ReLU(), nn.Linear(128, 16)
        )
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        cls = self.dequant(self.classifier(x.mean([2, 3])))
        # x = [self.up4x(x[0])]
        return cls


def make_anchors(
    h: int = 8, w: int = 10, strides: int = 3, grid_cell_offset: float = 0.5
):
    """Generate anchors from features.

    h,w: shape at last stride
    """
    strides = [32, 16, 8, 4][:strides]
    anchor_points = []
    stride_tensor = []
    dtype = torch.float
    for stride in strides:
        sx = torch.arange(end=w, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")  # torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype))
        h *= 2
        w *= 2

    return torch.cat(anchor_points), torch.cat(stride_tensor)


BOXCOLOR = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0), 3: (225, 225, 0)}


def draw_box(img, boxes, label, classification):
    img = cv2.copyMakeBorder(img, 0, 64, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    lt = (255, 255)
    rb = (255, 255)
    for i in range(boxes.shape[-1]):
        box = boxes[..., i][0]
        lt = (int(box[0]), int(box[1]))
        rb = (int(box[2]), int(box[3]))
        cv2.rectangle(img, lt, rb, BOXCOLOR[label[i].item()], 3, cv2.LINE_AA)

    cv2.putText(
        img,
        str(classification),
        (0, 280),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        str(lt) + str(rb),
        (50, 280),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return img


def decode_boxes(distance, anchor_points, dim=1):
    """Transform distance(ltrb) to box(xyxy)."""
    lt, rb = distance.split([2, 2], dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class IRIS(nn.Sequential):
    def __init__(self, conf):
        super().__init__(
            ConvBNReLU(1, 16, 3, 2, 1, True),  # 0
            InvertedResidual(16, 8, 1, 1),  # 1
            InvertedResidual(8, 20, 2, 10),  # 2p0
            InvertedResidual(20, 20, 1, 3.6),  # 3
            InvertedResidual(20, 24, 2, 5.4),  # 4p1
            InvertedResidual(24, 24, 1, 6),  # 5
            InvertedResidual(24, 48, 2, 6),  # 6p2
            InvertedResidual(48, 48, 1, 4),  # 7
            InvertedResidual(48, 24, 1, 2),  # 8
            InvertedResidual(24, 24, 1, 6),  # 9p3
            InvertedResidual(24, 40, 2, 6),  # 10p4
            EgisDetect2(nc=8, ch=[40]),  # 11
        )

    def fuse(self):
        for m in self.modules():
            if isinstance(m, InvertedResidual):
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.ao.quantization.fuse_modules_qat(
                            m.conv, [str(idx), str(idx + 1)], inplace=True
                        )

            if isinstance(m, ConvBNReLU):
                if len(m) == 3:
                    torch.ao.quantization.fuse_modules_qat(
                        m, ['0', '1', '2'], inplace=True
                    )
                else:
                    torch.ao.quantization.fuse_modules_qat(
                        m, ['1', '2', '3'], inplace=True
                    )

            if isinstance(m, DWConvbnrelu):
                torch.ao.quantization.fuse_modules_qat(m.conv, ['1', '2'], inplace=True)
            if (
                isinstance(m, torch.nn.Sequential)
                and isinstance(m[0], torch.nn.Linear)
                and len(m) == 3
            ):
                torch.ao.quantization.fuse_modules_qat(m, ['0', '1'], inplace=True)

    def forward(self, x):
        tmp = []
        for i in range(11):
            x = self[i](x)
        x = self[11](x)
        return x


def del_one2many_wgt(wgt):
    del_list = []
    for k in wgt:
        if '.cv' in k:
            del_list.append(k)
    for k in del_list:
        del wgt[k]
    return wgt


def webcam_demo(model, device, fbf, slow):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = loadnresize_image(frame)
        img = preprocess_image(frame, mono=args.mono, yuv=args.yuv, device=device)
        box, classification = model(img.unsqueeze(0))
        classification = classification.item()
        box, cls, conf, label = box
        img = draw_box(frame, box, label, classification)
        cv2.imshow('demo', img)
        if fbf:
            if input("pause") == "999":
                break
        if slow:
            time.sleep(0.1)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')
    for data, target in tqdm.tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = model(data)
        test_loss += lossLayer(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    top1_acc = 100.0 * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.3f}%\n'.format(test_loss, top1_acc))
    return test_loss, top1_acc


def evaluate(model, device):
    from ultralytics.utils.dataloader import create_loader
    from ultralytics.utils.distribution import get_rank, get_world_size
    import os
    datasets_folder = 'datasets/acer_test/'
    for person_dir in os.listdir(datasets_folder):
        if person_dir == 'train':
            continue
        print(f"Evaluating {person_dir}")
        data_loader_val = create_loader(
            # data_root=['datasets/ACR/v2.7/val'], # /home/trivenzhou/yolov10_250321/datasets/acer_test/EddieTan/5/gc6133
            data_root=[f'{datasets_folder}{person_dir}'], # 'datasets/ACR/v2.7'
            mono=True,
            train='val',
            num_tasks=get_world_size(),
            rank=get_rank(),
            batch_size=128,
            num_workers=12,
            jlist=None,
            root_j=None,
            auto_augment=False,
            rotation=False,
            val_size=[120, 160],
            dataset_type="default",
            paste_root=None,
        )
        test(model, device, data_loader_val)
    data_loader_val = create_loader(
        data_root=['datasets/ACR/v2.7/val'],
        mono=True,
        train='val',
        num_tasks=get_world_size(),
        rank=get_rank(),
        batch_size=128,
        num_workers=12,
        jlist=None,
        root_j=None,
        auto_augment=False,
        rotation=False,
        val_size=[120, 160],
        dataset_type="default",
        paste_root=None,
    )
    return test(model, device, data_loader_val)
    


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--wgt", type=str)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--wgt_type", action='store_true')
    parser.add_argument("--q", action='store_true')
    parser.add_argument("--fbf", action='store_true')
    parser.add_argument("--slow", action='store_true')
    parser.add_argument("--mono", action='store_true')
    parser.add_argument("--yuv", action='store_true')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = IRIS(conf=0)

    if args.q:
        model.fuse()
        model.train()
        model.qconfig = CUSTOM_QCFG
        torch.quantization.prepare_qat(model, inplace=True)

    wgt = torch.load(args.wgt, map_location=torch.device('cpu'))
    if args.wgt_type:
        wgt = wgt['model'].model.state_dict()

    # model.load_state_dict(wgt, strict=True)
    wgt = del_one2many_wgt(wgt)
    model.load_state_dict(wgt, strict=True)

    if args.q:
        model.eval()
        torch.quantization.convert(model, inplace=True)

    model.to(device)
    print("Use device: ", device)
    evaluate(model, device)
