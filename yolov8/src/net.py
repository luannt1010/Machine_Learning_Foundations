from .backbone import YOLOv8BackBone
from .head import YOLOv8Head
from .neck import YOLOv8Neck
import torch.nn as nn

class YOLOv8(nn.Module):
    def __init__(self, num_classes, d=0.33, w=0.25, r=2, reg_max=16):
        super().__init__()

        self.reg_max = reg_max
        self.num_classes = num_classes
        self.backbone = YOLOv8BackBone(d, w, r)
        self.neck = YOLOv8Neck(d, w, r)
        self.head = YOLOv8Head(num_classes, d, w, r, self.reg_max)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        preds = self.head(p3, p4, p5)
        return preds

def get_model(num_classes, suffix="n", reg_max=16):
    suffix = suffix.lower()
    if suffix not in ["n", "s", "m", "l", "x"]:
        print("Invalid suffix!")
        return

    d, w ,r = 0.33, 0.25, 2
    if suffix == "s":
        d, w, r = 0.33, 0.50, 2.0
    elif suffix == "m":
        d, w, r = 0.67, 0.75, 1.5
    elif suffix == "l":
        d, w, r = 1.0, 1.0, 1.0
    elif suffix == "x":
        d, w, r = 1.0, 1.25, 1.0

    model = YOLOv8(num_classes, d, w, r, reg_max)
    print(f"Load model YOLOv8{suffix} successfully!")
    return model

# if __name__ == "__main__":
#     model = get_model(80, "x")
#     print(model.head.reg_max)
#     # print(model)
#     x = torch.randn(1, 3, 1024, 1024)
#
#     preds = model(x)
#
#     for i, (k, v) in enumerate(preds.items()):
#         cls_logits, bbox_logits = v[0], v[1]
#         print(f"Scale {i}, {k.upper()}")
#         print("cls:", cls_logits.shape)
#         print("bbox:", bbox_logits.shape)
#     total_params = sum([p.numel() for p in model.parameters()])
#     print(f"Tổng số lượng tham số là: {total_params}")
