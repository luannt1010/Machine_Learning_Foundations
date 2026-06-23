import math
from .backbone import C2F, ConvBlock
import torch
import torch.nn as nn

class Upsampler(nn.Module):
    def __init__(self):
        super().__init__()

        self.upsampler = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.upsampler(x)

class TopDownFPN(nn.Module):
    def __init__(self, d=0.33, w=0.25, r=2):
        super().__init__()

        self.d = d
        self.w = w
        self.r = r

        self.up_sampler = Upsampler()

        n_c2f1 = math.ceil(self.d*3)
        c_in_c2f1 = math.ceil(512*self.w*(1+self.r))
        c_out_c2f1 = math.ceil(512*self.w)
        self.c2f1 = C2F(c_in_c2f1, c_out_c2f1, n_c2f1, False)

        c_in_c2f2 = math.ceil(768*self.w)
        c_out_c2f2 = math.ceil(256*self.w)
        self.c2f2 = C2F(c_in_c2f2, c_out_c2f2, n_c2f1, False)

    def forward(self, p5, p4, p3):
        p5_upped = self.up_sampler(p5)
        p5p4 = torch.concat([p5_upped, p4], dim=1)
        p5p4 = self.c2f1(p5p4)
        p5p4_upped = self.up_sampler(p5p4)
        p5p4p3 = torch.concat([p5p4_upped, p3], dim=1)
        p5p4p3 = self.c2f2(p5p4p3)
        return p5, p5p4, p5p4p3


class BottomUpPAN(nn.Module):
    def __init__(self, d=0.33, w=0.25, r=2):
        super().__init__()
        self.d = d
        self.w = w
        self.r = r

        c_in_conv1 = math.ceil(256*self.w)
        self.conv1 = ConvBlock(c_in_conv1, c_in_conv1, 3, 2, 1)

        c_in_c2f1 = math.ceil(768*self.w)
        c_out_c2f1 = math.ceil(512*self.w)
        n_c2f1 = math.ceil(3*self.d)
        self.c2f1 = C2F(c_in_c2f1, c_out_c2f1, n_c2f1, False)

        c_in_conv2 = math.ceil(512*self.w)
        self.conv2 = ConvBlock(c_in_conv2, c_in_conv2, 3, 2, 1)

        c_in_c2f2 = math.ceil(512*self.w*(1+self.r))
        c_out_c2f2 = math.ceil(512*self.w*self.r)
        self.c2f2 = C2F(c_in_c2f2, c_out_c2f2, n_c2f1, False)

    def forward(self, p3, p4, p5):
        p3_extracted = self.conv1(p3)
        p3p4 = torch.concat([p3_extracted, p4], dim=1)
        p3p4 = self.c2f1(p3p4)
        p3p4_extracted = self.conv2(p3p4)
        p3p4p5 = torch.concat([p3p4_extracted, p5], dim=1)
        p3p4p5 = self.c2f2(p3p4p5)
        return p3, p3p4, p3p4p5

class YOLOv8Neck(nn.Module):
    def __init__(self, d=0.33, w=0.25, r=2):
        super().__init__()

        self.top_down_fpn = TopDownFPN(d, w, r)
        self.bottom_up_pan = BottomUpPAN(d, w, r)

    def forward(self, p3, p4, p5):
        p5, p4, p3 = self.top_down_fpn(p5, p4, p3)
        p3, p4, p5 = self.bottom_up_pan(p3, p4, p5)
        return p3, p4, p5


# if __name__ == "__main__":
#     a = torch.rand(1, math.ceil(512*0.25*2.0), 20, 20)
#     b = torch.rand(1, math.ceil(512*0.25), 40, 40)
#     c = torch.rand(1, math.ceil(256*0.25), 80, 80)
#     # ab = torch.concat([a,b], dim=1)
#     # print(ab.shape)
#     head = TopDownFPN()
#     p5, p5p4, p5p4p3 = head(a, b, c)
#     print(p5.shape)
#     print(p5p4.shape)
#     print(p5p4p3.shape)
#     print("---")
#     c = torch.rand(1, math.ceil(256*0.25), 80, 80)
#     d = torch.rand(1, math.ceil(512*0.25), 40, 40)
#     e = torch.rand(1, math.ceil(512*0.25*2.0), 20, 20)
#     head2 = BottomUpPAN()
#     p3, p3p4, p3p4p5 = head2(c, d, e)
#     print(p3.shape)
#     print(p3p4.shape)
#     print(p3p4p5.shape)
#
#     f = torch.rand(1, math.ceil(512*0.25*2.0), 20, 20)
#     print(f.shape)
#     print(Upsampler()(f).shape)






