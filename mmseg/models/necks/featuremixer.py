import torch
from mmcv.cnn import ConvModule
import torch.nn as nn
import numpy as np
from mmseg.registry import MODELS

class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)
    

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, num_channels, use_ln=True):
        super(MixerBlock, self).__init__()
        self.use_ln = use_ln
        if use_ln:
            self.ln_token = nn.LayerNorm(num_channels)
            self.ln_channel = nn.LayerNorm(num_channels)
        self.token_mix = MlpBlock(num_tokens, num_tokens * 2)
        self.channel_mix = MlpBlock(num_channels, num_channels * 2)
    
    def forward(self, x):
        if self.use_ln:
            out = self.ln_token(x)
        else:
            out = x
        out = out.transpose(-1, -2)
        x = x + self.token_mix(out).transpose(-1, -2)
        if self.use_ln:
            out = self.ln_channel(x)
        else:
            out = x
        x = x + self.channel_mix(out)
        return x
    

class MLP(nn.Module):
    def __init__(self,
                 embedding_dim_in,
                 hidden_dim=None,
                 embedding_dim_out=None,
                 activation=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim_in
        embedding_dim_out = embedding_dim_out or embedding_dim_in
        self.fc1 = nn.Linear(embedding_dim_in, hidden_dim)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
    
    
@MODELS.register_module()
class MLPFPN(nn.Module):
    """MLPFPN

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_dim=8,
                 start_index=1,
                 start_stage=0,
                 end_stage=4,
                 feat_channels=[8, 16, 128],
                 mixer_count=1,
                 norm_cfg=None,
                 init_cfg=None,
                 linear_reduction=False):
        super(MLPFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_index = start_index
        self.num_ins = len(in_channels)
        self.mixer_count = mixer_count
        self.patch_dim = patch_dim
        self.start_stage = start_stage
        self.end_stage = end_stage
        self.feat_channels = feat_channels
        self.linear_reduction = linear_reduction
        self.backlinks=[]

        pc = int(np.sum([self.feat_channels[i] * 2**(2*(self.num_ins-1 - i)) for i in range(len(feat_channels))]))
        self.intprL = nn.Linear(pc, (self.patch_dim**2)*int(self.out_channels/16))

        self.intpr = nn.ModuleList()
        for i in range(len(self.feat_channels)):
            if self.linear_reduction:
                tokens = 2**(2*(self.num_ins-1 - i))
                self.intpr.append(nn.Linear(self.in_channels[i] * tokens, self.feat_channels[i] * tokens))
            else:
                self.intpr.append(ConvModule(self.in_channels[i], self.feat_channels[i], 1))

        self.mixers = None
        if self.mixer_count > 0:
            self.mixers = nn.Sequential(*[
                MixerBlock(self.patch_dim**2, self.out_channels) for i in range(self.mixer_count)
            ])

    def window_partition(self, x, window_size, channel_last=True):
        """
        Args:
            x: (B, W, H, C)
            window_size (int): window size
        Returns:
            windows: (B, num_windows, window_size * window_size, C)
            :param channel_last: if channel is last dim
        """
        if not channel_last:
            x = x.permute(0, 3, 2, 1)
        B, W, H, C = x.shape
        x = x.view(B, W // window_size, window_size, H // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, -1, window_size * window_size, C)
        return windows

    def window_reverse(self, windows, window_size, W, H):
        """
        Args:
            windows: (B, num_windows, window_size*window_size, C)
            window_size (int): Window size
            W (int): Width of image
            H (int): Height of image
        Returns:
            x: (B, C, W, H)
        """
        B = windows.shape[0]
        x = windows.view(B, W // window_size, H // window_size, window_size, window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, W, H)
        return x

    def init_weights(self):
        pass

    def forward(self, inputs):

        B, _, H4, W4 = inputs[0].shape
        parts = []
        for i in range(len(self.feat_channels)):
            if self.linear_reduction:

                part = self.window_partition(inputs[i], 2**(self.num_ins-1 - i), channel_last=False)
                part = torch.flatten(part, -2)
                part = self.intpr[i](part)
            else:
                part = self.intpr[i](inputs[i])
                part = self.window_partition(part, 2 ** (self.num_ins - 1 - i), channel_last=False)
                part = torch.flatten(part, -2)
            parts.append(part)

        out = torch.cat(parts, dim=-1)
        out = self.intprL(out)

        outputs = out.view(B, -1, self.patch_dim**2, self.out_channels)
        
        if self.mixers is not None:
            outputs = self.mixers(outputs)
            
        B, W, H, C = outputs.shape
        
        # if self.training:
        #     # outputs_final = outputs.view(B, C * 4, H, -1)
        #     outputs_final = outputs.view(B, -1, int(H / 4), int(H / 2))
        # else:
        #     outputs_final = outputs.view(B, -1, int(H / 2), H)

        if self.training:
            outputs_final = outputs.view(B, C, H, W)
        else:
            outputs_final = outputs.view(B, C, H * 2, -1)

        # outputs_final = outputs.view(B, C * 4, H , -1)
        # outputs_final = outputs.view(B, C, H, W)

        return tuple([outputs_final])
