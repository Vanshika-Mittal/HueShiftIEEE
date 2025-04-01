import torch
import torch.nn as nn
from models.embeddings import get_time_embedding
from models.blocks import DownBlock, MidBlock, UpBlock


class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """
    # im_channels will be the no of input channels (latent channels)
    def __init__(self, in_channels, out_channels, model_config, condition=False):
        super().__init__()
        self.condition = condition
        self.down_channels = model_config['DOWN_CHANNELS'] # [256, 384, 512, 768]
        self.mid_channels = model_config['MID_CHANNELS'] # [768, 512]
        self.t_emb_dim = model_config['TIME_EMB_DIM'] # 512
        self.down_sample = model_config['DOWN_SAMPLE'] # [True, True, True]
        self.num_down_layers = model_config['NUM_DOWN_LAYERS'] # 2
        self.num_mid_layers = model_config['NUM_MID_LAYERS'] # 2
        self.num_up_layers = model_config['NUM_UP_LAYERS'] # 2
        self.attns = model_config['ATTN'] # [True, True, True]
        self.norm_channels = model_config['NORM_CHANNELS'] # 32
        self.num_heads = model_config['NUM_HEADS'] # 16
        self.conv_out_channels = model_config['CONV_OUT_CHANNELS'] # 128
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1


        # Spatial Conditioning
        if self.condition:
            self.cond_channels = model_config['CONDITION']['COND_CHANNELS']
            self.conv_in_concat = nn.Conv2d(in_channels + self.cond_channels,
                                            self.down_channels[0], kernel_size=3, padding=1)
        else:
            self.conv_in = nn.Conv2d(in_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        # only change is to add time embeddings
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], 
                                        t_emb_dim=self.t_emb_dim,down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        attn=self.attns[i],
                                        norm_channels=self.norm_channels))

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlock("unet",self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                                    self.t_emb_dim, up_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_up_layers,
                                        attn=self.attns[i],
                                        norm_channels=self.norm_channels))
        
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t, cond_in = None):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        if cond_in is not None:
            # cond_in = self.cond_conv_in(cond_in)
            cond_in = nn.functional.interpolate(size = x.shape[-2:])
            x = torch.concat([x, cond_in],dim=1)
            out = self.conv_in_concat(x)

        else:
            out = self.conv_in(x)

        # B x C1 x H x W
        
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        down_outs = []
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
        
        for mid in self.mids:
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out