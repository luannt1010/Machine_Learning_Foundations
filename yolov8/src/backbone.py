import math
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=2, p=1):
        super().__init__()

        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.SiLU())

    def forward(self, x):
        return self.block(x)

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=0, pool_k=5):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv2d(out_channels*4, out_channels, kernel_size=k, stride=s, padding=p)
        self.max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=pool_k//2)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.max_pool(out1)
        out3 = self.max_pool(out2)
        out4 = self.max_pool(out3)
        concat = torch.concat([out1, out2, out3, out4], dim=1)
        return self.conv2(concat)

class BottleNeck(nn.Module):
    def __init__(self, in_channels:int , out_channels:int, k=3, s=1, p=1, shortcut=False):
        super().__init__()

        self.in_c = in_channels
        self.out_c = out_channels
        self.shortcut = self.make_shortcut() if shortcut else None
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=k, stride=s, padding=p)

    def make_shortcut(self):
        shortcut = nn.Sequential(nn.Conv2d(self.in_c, self.out_c, kernel_size=3, stride=1, padding=1),
                             nn.BatchNorm2d(self.out_c))
        return shortcut

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut is not None:
            if x.shape[1] != identity.shape[1]:
                identity = self.shortcut(identity)
            x = x + identity
        return x

class Detect(nn.Module):
    def __init__(self, in_channels, num_classes, reg_max=16, k=3, s=1, p=1):
        super().__init__()

        self.reg_max = reg_max
        self.cls = nn.Sequential(ConvBlock(in_channels, in_channels, k=k, s=s, p=p),
                                 ConvBlock(in_channels, in_channels, k=k, s=s, p=p),
                                 nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0))
        self.bbox = nn.Sequential(ConvBlock(in_channels, in_channels, k=k, s=s, p=p),
                                 ConvBlock(in_channels, in_channels, k=k, s=s, p=p),
                                 nn.Conv2d(in_channels=in_channels, out_channels=4*self.reg_max, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        cls_logits = self.cls(x)
        reg_logits = self.bbox(x)
        return cls_logits, reg_logits

class C2F(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=False, k=1, s=1, p=0, e=0.5):
        super().__init__()

        self.n = n
        self.c_expanded = int(out_channels * e)

        self.conv_block1 = ConvBlock(in_channels, out_channels, k, s, p)
        self.n_bottleneck = nn.ModuleList(BottleNeck(self.c_expanded, self.c_expanded, k=k, s=s, p=p, shortcut=shortcut) for _ in range(n))
        self.conv_block2 = ConvBlock(self.c_expanded*(self.n+2), out_channels, k, s, p)

    def forward(self, x):
        x = self.conv_block1(x)
        split1, split2 = torch.chunk(x, 2, 1)
        fm = [split1, split2]
        out = split2
        for bottleneck in self.n_bottleneck:
            out = bottleneck(out)
            fm.append(out)
        concat = torch.concat(fm, dim=1)
        return self.conv_block2(concat)

class YOLOv8BackBone(nn.Module):
    def __init__(self, d=0.33, w=0.25, r=2.0, shortcut=True):
        super().__init__()
        self.d = d
        self.w = w
        self.r = r
        self.shortcut = shortcut

        c_p1_out = math.ceil(64*self.w)
        self.p1 = nn.Conv2d(in_channels=3, out_channels=c_p1_out, kernel_size=3, stride=2, padding=1)

        c_p2_out = math.ceil(128*self.w)
        n_p2 = math.ceil(3*self.d)
        self.p2 = nn.Sequential(nn.Conv2d(in_channels=c_p1_out, out_channels=c_p2_out, kernel_size=3, stride=2, padding=1),
                                C2F(in_channels=c_p2_out, out_channels=c_p2_out, n=n_p2, shortcut=self.shortcut))

        c_p3_out = math.ceil(256*self.w)
        n_p3 = math.ceil(6*self.d)
        self.p3 = nn.Sequential(nn.Conv2d(in_channels=c_p2_out, out_channels=c_p3_out, kernel_size=3, stride=2, padding=1),
                                C2F(in_channels=c_p3_out, out_channels=c_p3_out, n=n_p3, shortcut=self.shortcut))

        c_p4_out = math.ceil(512*self.w)
        self.p4 = nn.Sequential(nn.Conv2d(in_channels=c_p3_out, out_channels=c_p4_out, kernel_size=3, stride=2, padding=1),
                                C2F(in_channels=c_p4_out, out_channels=c_p4_out, shortcut=self.shortcut))

        c_p5_out = math.ceil(512*self.w*self.r)
        self.p5 = nn.Sequential(nn.Conv2d(in_channels=c_p4_out, out_channels=c_p5_out, kernel_size=3, stride=2, padding=1),
                                C2F(in_channels=c_p5_out, out_channels=c_p5_out, shortcut=self.shortcut))

        self.sppf = SPPF(in_channels=c_p5_out, out_channels=c_p5_out)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        out = self.sppf(p5)
        return p3, p4, out

# if __name__ == "__main__":
#     x = torch.randn(1, 3, 640, 640)
#
#     model = BackBone(d=0.33, w=0.25, r=2.0, shortcut=True)
#
#     p3, p4, p5 = model(x)
#
#     print("p3:", p3.shape)
#     print("p4:", p4.shape)
#     print("p5:", p5.shape)



