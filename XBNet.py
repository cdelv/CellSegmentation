import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGN(nn.Module):
    def __init__(self, in_ch, out_ch, n_group, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)
        self.gn = nn.GroupNorm(num_groups=n_group, num_channels=out_ch)
    def forward(self, x):
        return self.gn(self.conv(x))

class DownBlock(nn.Module):
    """Dense-residual 5-layer block + downsample by 2."""
    def __init__(self, in_ch, out_ch, n_group):
        super().__init__()
        nf = in_ch
        gc = nf // 4
        # 5 dense convs
        self.conv1 = nn.Sequential(ConvGN(nf,    gc, n_group), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(ConvGN(nf+gc, gc, n_group), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(ConvGN(nf+2*gc, gc, n_group), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(ConvGN(nf+3*gc, gc, n_group), nn.ReLU(inplace=True))
        self.conv5 = ConvGN(nf+4*gc, nf, n_group)  # no activation
        # then downsample
        self.down = nn.Sequential(
            nn.ReLU(inplace=True),
            ConvGN(in_ch, out_ch, n_group, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], dim=1))
        x3 = self.conv3(torch.cat([x, x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        # scaled residual
        x6 = x + 0.2 * x5
        # downsample
        return self.down(x6)

class ASPP(nn.Module):
    """ASPP with 1×1, dilated 3×3, image-pool branches + residual skip."""
    def __init__(self, in_ch, out_ch, n_group, rates=(6,12,18)):
        super().__init__()
        mid = in_ch // 2

        # 1×1 conv branch
        self.conv1 = nn.Sequential(
            ConvGN(in_ch, mid, n_group, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        # dilated conv branches
        self.atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, mid, kernel_size=3, padding=r, dilation=r, bias=False),
                nn.GroupNorm(n_group, mid),
                nn.ReLU(inplace=True)
            ) for r in rates
        ])
        # image-pooling branch
        self.img_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            ConvGN(in_ch, mid, n_group, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        # project back to out_ch
        total = 1 + len(rates) + 1
        self.exit = nn.Sequential(
            nn.Conv2d(mid * total, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(n_group, out_ch)
        )
        # residual skip
        self.skip = (ConvGN(in_ch, out_ch, n_group, kernel_size=1, padding=0)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        size = x.shape[-2:]
        feats = [self.conv1(x)]                             # 1×1
        feats += [b(x) for b in self.atrous]                # dilated 3×3s
        img = self.img_pool(x)                              # pooled
        img = F.interpolate(img, size=size,
                            mode='bilinear', align_corners=False)
        feats.append(img)

        cat = torch.cat(feats, dim=1)
        out = self.exit(cat)
        res = self.skip(x)
        return F.relu(out + res, inplace=True)

class UpBlock(nn.Module):
    """PixelShuffle ↑2 + bridge conv + 5-layer dense-residual path."""
    def __init__(self, in_ch, out_ch, n_group):
        super().__init__()
        # 1) upsample by 2 via PixelShuffle
        #    in_ch must be = out_ch * 2
        self.up = nn.Sequential(
            nn.PixelShuffle(2),                         # → in_ch/4 channels
            nn.GroupNorm(n_group, out_ch//2),           # out_ch//2 == in_ch/4
            nn.ReLU(inplace=True)
        )
        # 2) project skip connection down to out_ch//2
        self.bridge = nn.Sequential(
            nn.Conv2d(out_ch, out_ch//2, kernel_size=1, bias=False),
            nn.GroupNorm(n_group, out_ch//2),
            nn.ReLU(inplace=True)
        )
        # 3) dense-residual path (5 convs)
        nf = out_ch               # after concat(up,bridge)
        gc = nf // 4
        self.conv1 = nn.Sequential(ConvGN(nf,      gc, n_group), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(ConvGN(nf+gc,   gc, n_group), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(ConvGN(nf+2*gc, gc, n_group), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(ConvGN(nf+3*gc, gc, n_group), nn.ReLU(inplace=True))
        self.conv5 = ConvGN(nf+4*gc, nf, n_group)  # no activation

    def forward(self, x, skip):
        u  = self.up(x)                    # (B, out_ch//2, H*2, W*2)
        br = self.bridge(skip)             # (B, out_ch//2, H*2, W*2)
        x0 = torch.cat([u, br], dim=1)     # (B, out_ch,  H*2, W*2)

        x1 = self.conv1(x0)
        x2 = self.conv2(torch.cat([x0, x1], dim=1))
        x3 = self.conv3(torch.cat([x0, x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x0, x1, x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x0, x1, x2, x3, x4], dim=1))

        # scaled residual add
        return F.relu(x0 + 0.2 * x5, inplace=True)
        
class XBNet(nn.Module):
    def __init__(self, in_ch=1, base_chns = 16, n_group = 4, input_size = (256, 256), output_size = (256, 256), out_ch = 3):
        super(XBNet, self).__init__()
        self.pool_input = nn.AdaptiveAvgPool2d(output_size=input_size)
        self.pool_out = nn.AdaptiveAvgPool2d(output_size=output_size)

        self.entry = nn.Sequential(
            ConvGN(in_ch, base_chns, n_group, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            ConvGN(base_chns, base_chns, n_group, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.down_path = nn.ModuleList()
        ch = base_chns
        for _ in range(4):
            self.down_path.append(DownBlock(ch, ch*2, n_group))
            ch *= 2

        # ASPP bottleneck
        self.aspp = ASPP(ch, ch, n_group)

        # up path
        self.up_path = nn.ModuleList()
        for _ in range(4):
            self.up_path.append(UpBlock(ch, ch//2, n_group))
            ch //= 2
            
        # final 1×1 conv and output resize
        self.last     = nn.Conv2d(ch, out_ch, kernel_size=1)
        
    def forward(self, x):
        x = self.pool_input(x)

        x = self.entry(x)
        
        # down‐channel
        skips = [x]
        for i, down in enumerate(self.down_path):
            x = down(x)
            # XBNet kept only the first 3 skip maps after entry:
            if i < 3:
                skips.append(x)

        x = self.aspp(x)

        for up, skip in zip(self.up_path, reversed(skips)):
            x = up(x, skip)

        x = self.last(x)

        return self.pool_out(x)