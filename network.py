import torch.nn as nn

class DSResBlock(nn.Module):
    """Depthwise-separable residual block for MFCC spectrogram inputs.

    Structure:
        - Depthwise 3x3 convolution (groups=c_in) with optional spatial downsampling via `stride`.
        - BatchNorm + ReLU.
        - Pointwise 1x1 convolution to change channel dimension (c_in -> c_out).
        - BatchNorm.
        - Residual skip connection is applied only when shapes match (c_in == c_out and stride == (1, 1)).
        - Final ReLU.

    This block is used to keep the model lightweight on edge devices by replacing a standard
    3x3 conv with a depthwise + pointwise decomposition.

    Args:
        c_in: Input channels.
        c_out: Output channels.
        stride: (stride_time, stride_freq) for the depthwise conv.
    """
    def __init__(self, c_in: int, c_out: int, stride=(1, 1)):
        super().__init__()
        self.use_skip = (c_in == c_out) and (stride == (1, 1))

        self.dw = nn.Conv2d(
            c_in, c_in, 
            kernel_size=3, 
            stride=stride, 
            padding=1,
            groups=c_in, 
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(c_in)
        self.act1 = nn.ReLU()

        self.pw = nn.Conv2d(
            c_in, c_out, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act2 = nn.ReLU()

    def forward(self, x):
        out = self.act1(self.bn1(self.dw(x)))
        out = self.bn2(self.pw(out))
        if self.use_skip:
            out = out + x
        out = self.act2(out)
        return out


    def make_model(num_classes=2, width_mult=0.35, dropout=0.0, stem_downsample_freq=True):
        base = 256
        c0 = max(16, int(base * width_mult))

        stem_stride = (2, 1) if stem_downsample_freq else (1, 1)

        return nn.Sequential(
            nn.Conv2d(1, c0, kernel_size=3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(),

            DSResBlock(c0, c0, stride=(1, 1)),
            DSResBlock(c0, c0, stride=(1, 1)),
            DSResBlock(c0, c0, stride=(1, 1)),
            DSResBlock(c0, c0, stride=(1, 1)),
            DSResBlock(c0, c0, stride=(1, 1)),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c0, num_classes),
        )

    def freeze_bn(m):
        if isinstance(m,nn.modules.batchnorm._BatchNorm):
            m.eval()