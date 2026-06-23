import math
import torch.nn as nn
from .backbone import Detect

class YOLOv8Head(nn.Module):
    def __init__(self, num_classes, d=0.33, w=0.25, r=2, reg_max=16):
        super().__init__()
        self.w = w
        self.d = d
        self.r = r
        self.reg_max = reg_max

        c_in_head1 = math.ceil(256*self.w)
        self.head1 = Detect(c_in_head1, num_classes, self.reg_max)

        c_in_head2 = math.ceil(512*self.w)
        self.head2 = Detect(c_in_head2, num_classes, self.reg_max)

        c_in_head3 = math.ceil(512*self.w*self.r)
        self.head3 = Detect(c_in_head3, num_classes, self.reg_max)


    def forward(self, p3, p4, p5):
        cls_logits_p3, reg_logits_p3 = self.head1(p3)
        cls_logits_p4, reg_logits_p4 = self.head2(p4)
        cls_logits_p5, reg_logits_p5 = self.head3(p5)
        return {"p3": [cls_logits_p3, reg_logits_p3],
                "p4": [cls_logits_p4, reg_logits_p4],
                "p5": [cls_logits_p5, reg_logits_p5]}


# a = torch.randn(2, 64, 20, 20)
# B, C, H, W = a.shape
# print(a.shape)
# a = a.view(a.shape[0], a.shape[1], a.shape[2]*a.shape[3])
# print(a.shape)
