import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import MySequential, LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

# --- UNCHANGED HELPER MODULES ---

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 3, 1, 1, groups=c // 4, bias=True)
        
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class ConvGRUCell(nn.Module):
    """
    A Convolutional GRU cell to pass hidden states between recursive steps.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.reset_gate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.update_gate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)
        self.out_gate = nn.Conv2d(in_channels + hidden_channels, hidden_channels, kernel_size, padding=self.padding)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        
        h_hat = torch.tanh(self.out_gate(torch.cat([x, h_prev * reset], dim=1)))
        h_new = h_prev * (1 - update) + h_hat * update
        return h_new

class RecursiveNAFBlockSR(nn.Module):

    def __init__(self, c, fusion_module=None, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        self.fusion = fusion_module
        self.state_updater = ConvGRUCell(in_channels=c, hidden_channels=c, kernel_size=3)

    def forward(self, feats, hidden_states):
        processed_feats = tuple(self.blk(x) for x in feats)
        
        if self.fusion:
            processed_feats = self.fusion(*processed_feats)

        new_hidden_states = tuple(self.state_updater(feat, h) for feat, h in zip(processed_feats, hidden_states))
        
        return new_hidden_states, new_hidden_states


class DisparityEstimator(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels // 2, 1, 3, 1, 1)
        )

    def forward(self, x_l, x_r):
        return self.estimator(torch.cat([x_l, x_r], dim=1))

class DisparityGuidedFusion(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.disparity_estimator_r2l = DisparityEstimator(c)
        self.disparity_estimator_l2r = DisparityEstimator(c)
        self.fusion_conv_l = nn.Conv2d(c * 2, c, 3, 1, 1)
        self.fusion_conv_r = nn.Conv2d(c * 2, c, 3, 1, 1)
        
        self.gamma_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma_r = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x_l, x_r):
        B, C, H, W = x_l.shape
        
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2).float().to(x_l.device)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) # Shape: (B, H, W, 2)

        disparity_r2l = self.disparity_estimator_r2l(x_l, x_r) # Shape: (B, 1, H, W)
        new_grid_r2l = grid.clone()
        new_grid_r2l[..., 0] = new_grid_r2l[..., 0] - disparity_r2l.squeeze(1) # Apply horizontal shift
        norm_grid_r2l = new_grid_r2l * 2 / torch.tensor([W-1, H-1], dtype=torch.float32).to(x_l.device) - 1
        
        warped_r = F.grid_sample(x_r, norm_grid_r2l, mode='bilinear', padding_mode='border', align_corners=True)
        fused_l = self.fusion_conv_l(torch.cat([x_l, warped_r], dim=1))
        out_l = x_l + fused_l * self.gamma_l
        
        disparity_l2r = self.disparity_estimator_l2r(x_r, x_l)
        new_grid_l2r = grid.clone()
        new_grid_l2r[..., 0] = new_grid_l2r[..., 0] + disparity_l2r.squeeze(1) # Apply positive shift
        norm_grid_l2r = new_grid_l2r * 2 / torch.tensor([W-1, H-1], dtype=torch.float32).to(x_l.device) - 1
        
        warped_l = F.grid_sample(x_l, norm_grid_l2r, mode='bilinear', padding_mode='border', align_corners=True)
        fused_r = self.fusion_conv_r(torch.cat([x_r, warped_l], dim=1))
        out_r = x_r + fused_r * self.gamma_r

        return out_l, out_r


class InnovativeNAFNetSR(nn.Module):
    def __init__(self, up_scale=4, width=64, num_recursive_steps=5, img_channel=3, drop_out_rate=0.):
        super().__init__()
        self.dual = True 
        self.up_scale = up_scale
        
        # Initial feature extraction
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, bias=True)
        
        fusion_module = DisparityGuidedFusion(c=width)
        
        self.recursive_body = RecursiveNAFBlockSR(c=width, fusion_module=fusion_module, drop_out_rate=drop_out_rate)
        self.num_recursive_steps = num_recursive_steps
        
        self.body_start = MySequential(*[NAFBlock(c=width) for _ in range(2)])
        self.body_end = MySequential(*[NAFBlock(c=width) for _ in range(2)])
        
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale ** 2, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(up_scale)
        )

    def forward(self, inp):
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bicubic', align_corners=False)
        
        inp = inp.chunk(2, dim=1)
        
        feats = [self.intro(x) for x in inp]
        
        feats = tuple(self.body_start(feat) for feat in feats)
        
        hidden_states = feats
        
        for _ in range(self.num_recursive_steps):
            feats, hidden_states = self.recursive_body(feats, hidden_states)
        
        feats = tuple(self.body_end(feat) for feat in feats)
        
        out = torch.cat([self.up(x) for x in feats], dim=1)
        out = out + inp_hr
        return out

class INAFSSR(Local_Base, INAFNetSR):
    def __init__(self, *args, train_size=(1, 6, 30, 90), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        INAFNetSR.__init__(self, *args, img_channel=3, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

if __name__ == '__main__':
    # Configuration for the I model
    width = 64
    num_recursive_steps = 5 # How many times to apply the stateful recursive block
    train_size = (1, 6, 30, 90) # (Batch, Channels, Height, Width)

    net = INAFSSR(
        up_scale=4, 
        train_size=train_size, 
        fast_imp=True, 
        width=width, 
        num_recursive_steps=num_recursive_steps
    )

    inp_shape = (6, 64, 64) # (Stereo Channels, Height, Width)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    print(f"I-NAFSSR ({num_recursive_steps} recursive steps, width={width})")
    print(f"  Parameters: {params}")
    print(f"  MACs (G): {float(macs[:-4])}")

    # --- Test forward pass ---
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        output = net(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    assert output.shape == (1, 6, 256, 256)
    print("Forward pass successful!")