import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.warplayer import warp
from model.head import Head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True, groups=groups),        
        nn.PReLU(out_planes)
    )


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )


class Resblock(nn.Module):
    def __init__(self, c, dilation=1):
        super(Resblock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1),
            nn.PReLU(c),
            nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1),
        )
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.prelu = nn.PReLU(c)

    def forward(self, x):
        y = self.conv(x)
        return self.prelu(y * self.beta + x)


class RoundSTE(torch.autograd.Function):
    """
    Deterministic straight-through estimator for gating.
    Forward: deterministic rounding at 0.5 threshold (0 or 1).
    Backward: pass-through gradient (identity), which stabilizes training
    compared to pure Bernoulli sampling.
    """
    @staticmethod
    def forward(ctx, x):
        return (x >= 0.5).float()

    @staticmethod
    def backward(ctx, grad):
        return grad
    

class RecurrentBlock(nn.Module):
    def __init__(self, c, dilation=1, depth=6):
        super(RecurrentBlock, self).__init__()
        # conv_stem expects concatenated input channels (3*c + 6 + 1)
        self.conv_stem = conv(3*c+6+1, c, 3, 1, 1, groups=1)
        self.conv_backbone = torch.nn.ModuleList([])
        self.depth = depth
        for i in range(depth):
            self.conv_backbone.append(Resblock(c, dilation))
        
    def forward(self, x, i0, i1, flow, timestep, convflow, getscale):
        # downscale flow for feature warping
        flow_down = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False)
        i0_warp = warp(i0, flow_down[:, :2] * 0.5)
        i1_warp = warp(i1, flow_down[:, 2:4] * 0.5)
        x_cat = torch.cat((x, flow_down, i0_warp, i1_warp, timestep), 1)

        # Use deterministic rounding STE to compute discrete-like scale gating but keep
        # a stable gradient path.
        scale_raw = getscale(x_cat)
        scale = RoundSTE.apply(scale_raw).unsqueeze(2).unsqueeze(3)

        feat = 0
        # branch for full resolution
        cond_full = (scale.shape[0] != 1) or (scale[:, 0:1].mean() >= 0.5 and scale[:, 1:2].mean() >= 0.5)
        if cond_full:
            x0 = self.conv_stem(x_cat)
            for i in range(self.depth):
                x0 = self.conv_backbone[i](x0)
            feat = feat + x0 * scale[:, 0:1] * scale[:, 1:2]

        # branch for 0.5x -> upsample
        cond_half = (scale.shape[0] != 1) or (scale[:, 0:1].mean() < 0.5 and scale[:, 1:2].mean() >= 0.5)
        if cond_half:
            x1 = self.conv_stem(F.interpolate(x_cat, scale_factor=0.5, mode="bilinear", align_corners=False))
            for i in range(self.depth):
                x1 = self.conv_backbone[i](x1)
            feat = feat + F.interpolate(x1, scale_factor=2.0, mode="bilinear", align_corners=False) * (1 - scale[:, 0:1]) * scale[:, 1:2]

        # branch for 0.25x -> upsample
        cond_quarter = (scale.shape[0] != 1) or (scale[:, 1:2].mean() < 0.5)
        if cond_quarter:
            x2 = self.conv_stem(F.interpolate(x_cat, scale_factor=0.25, mode="bilinear", align_corners=False))
            for i in range(self.depth):
                x2 = self.conv_backbone[i](x2)
            feat = feat + F.interpolate(x2, scale_factor=4.0, mode="bilinear", align_corners=False) * (1 - scale[:, 1:2])

        return feat, convflow(feat) + flow, i0_warp, i1_warp, scale

class Flownet(nn.Module):
    def __init__(self, block_num, c=64):
        super(Flownet, self).__init__()
        self.convimg = Head(c)
        self.shuffle = conv(2*c, c, 3, 1, 1, groups=1)
        self.convblock = torch.nn.ModuleList([])
        self.block_num = block_num
        self.convflow = nn.Sequential(
            nn.Conv2d(c, 4*6, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.getscale = nn.Sequential(
            conv(3*c+6+1, c, 1, 1, 0),
            conv(c, c, 1, 2, 0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c, 2),
            nn.Sigmoid()
        )
        for i in range(self.block_num):
            self.convblock.append(RecurrentBlock(c, 1, 2))

    def extract_feat(self, x):
        i0 = self.convimg(x[:, :3])
        i1 = self.convimg(x[:, 3:6])
        feat = self.shuffle(torch.cat((i0, i1), 1))
        return feat, i0, i1
        
    def forward(self, i0, i1, feat, timestep, flow):
        flow_list = []
        feat_list = []
        scale_list = []
        for i in range(self.block_num):
            feat, flow, w0, w1, scale = self.convblock[i](feat, i0, i1, flow, timestep, self.convflow, self.getscale)
            flow_list.append(flow)
            feat_list.append(feat)
            scale_list.append(scale)
        return flow_list, feat_list, torch.cat(scale_list, 1)
        
class SAFA(nn.Module):
    """SAFA high-level module (fixed):
    - Use a Flownet instance internally to provide consistent feature extraction
      and recurrent blocks.
    - Single unified forward() implementation (no duplicate definitions).
    - Deterministic STE gating (RoundSTE) is used in RecurrentBlock for stability.
    """
    def __init__(self, flownet_blocks=1, c=96):
        super(SAFA, self).__init__()
        # create a Flownet configured to match channel sizes used in original file
        self.flownet = Flownet(block_num=flownet_blocks, c=c)
        # expose the first recurrent block for direct usage if needed
        self.block = self.flownet.convblock[0]
        self.lastconv = conv(c*3+1, 3, 3, 1, 1)

    def inference(self, lowres, timestep=0.5):
        """Public inference helper. Supports a single scalar timestep or a list of timesteps.
        Returns a list of interpolated images for each requested timestep.
        """
        if isinstance(timestep, list):
            merged = []
            for t in timestep:
                result = self.forward(lowres, t, training=False)
                merged.append(result)
            return merged
        else:
            # Handle common case: single scalar or two-step (e.g., 3x interpolation)
            if isinstance(timestep, (float, int)):
                return self.forward(lowres, float(timestep), training=False)
            else:
                # assume tensor-like but we normalize to float
                return self.forward(lowres, float(timestep.item()), training=False)

    def forward(self, lowres, timestep=0.5, training=False):
        """Unified forward for training and inference.
        - lowres: tensor with concatenated frames in channel dim (e.g., [B, 6, H, W])
        - timestep: scalar in [0,1] or tensor; if tensor, will be reshaped appropriately.
        Returns:
          - during training: (flow_cat, soft_scale, merged_list) as in original intent
          - during inference: single interpolated image tensor
        """
        img0 = lowres[:, :3]
        img1 = lowres[:, -3:]

        # normalize timestep to tensor on correct device
        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, dtype=torch.float32, device=lowres.device)
        timestep = timestep.reshape(1, 1, 1, 1).repeat(img0.shape[0], 1, img0.shape[2], img0.shape[3])
        timestep = F.interpolate(timestep, scale_factor=0.5, mode="bilinear", align_corners=False)

        # Use flownet to extract features consistently
        feat, i0, i1 = self.flownet.extract_feat(torch.cat((img0, img1), 1))

        # run the recurrent block(s)
        flow_list, feat_list, soft_scale = self.flownet(i0, i1, feat, timestep, (lowres[:, :6] * 0).detach())

        # take the final flow and produce warped images + residual correction
        flow_sum = flow_list[-1]
        warped_i0 = warp(img0, flow_sum[:, :2])
        warped_i1 = warp(img1, flow_sum[:, 2:4])
        mask = torch.sigmoid(flow_sum[:, 4:5])
        warped = warped_i0 * mask + warped_i1 * (1 - mask)

        flow_down = F.interpolate(flow_sum, scale_factor=0.5, mode="bilinear", align_corners=False)
        w0 = warp(i0, flow_down[:, :2] * 0.5)
        w1 = warp(i1, flow_down[:, 2:4] * 0.5)
        img = self.lastconv(torch.cat((timestep, w0, w1), 1))
        result = torch.clamp(warped + img, 0, 1)

        if training:
            # if training, produce additional diagnostic outputs similar to original
            merged = []
            soft_scale_list = []
            # create a list of timesteps for multi-step supervision (if desired)
            one = 1 - timestep * 0
            timestep_list = [timestep * 0, one / 8, one / 8 * 2, one / 8 * 3, one / 8 * 4, one / 8 * 5, one / 8 * 6, one / 8 * 7, one]
            for ts in timestep_list:
                flow_list_t, feat_list_t, soft_scale_t = self.flownet(i0, i1, feat, ts, (lowres[:, :6] * 0).detach())
                flow_sum_t = flow_list_t[-1]
                warped_i0_t = warp(img0, flow_sum_t[:, :2])
                warped_i1_t = warp(img1, flow_sum_t[:, 2:4])
                mask_t = torch.sigmoid(flow_sum_t[:, 4:5])
                warped_t = warped_i0_t * mask_t + warped_i1_t * (1 - mask_t)
                flow_down_t = F.interpolate(flow_sum_t, scale_factor=0.5, mode="bilinear", align_corners=False)
                w0_t = warp(i0, flow_down_t[:, :2] * 0.5)
                w1_t = warp(i1, flow_down_t[:, 2:4] * 0.5)
                img_t = self.lastconv(torch.cat((ts, w0_t, w1_t), 1))
                merged.append(torch.clamp(warped_t + img_t, 0, 1))
                soft_scale_list.append(soft_scale_t)
            # return flows concatenated in width, a representative soft_scale and merged list
            flow_cat = torch.cat(flow_list, 3)
            return flow_cat, soft_scale_list[1] if len(soft_scale_list) > 1 else soft_scale_list[0], merged
        else:
            return result
