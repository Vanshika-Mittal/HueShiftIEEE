import torch
import torch.nn as nn
from models.blocks import DownBlock, MidBlock, UpBlock


class VAE(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.im_channels = model_config['IM_CHANNELS'] # 3
        self.down_channels = model_config['DOWN_CHANNELS'] # [64, 128, 256, 256]
        self.mid_channels = model_config['MID_CHANNELS'] # [256, 256]
        self.down_sample = model_config['DOWN_SAMPLE'] # [True, True. True]
        self.num_down_layers = model_config['NUM_DOWN_LAYERS'] # 2
        self.num_mid_layers = model_config['NUM_MID_LAYERS'] # 2
        self.num_up_layers = model_config['NUM_UP_LAYERS'] # 2
        
        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config['DOWN_UP_ATTN'] # [False, False, False]
        
        # Latent Dimension
        self.z_channels = model_config['Z_CHANNELS'] # 3
        self.norm_channels = model_config['NORM_CHANNELS'] # 32
        self.num_heads = model_config['NUM_HEADS'] # 4
        
        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        self.up_sample = list(reversed(self.down_sample))

        self.silu = nn.SiLU()
        
        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(self.im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        # Downblock + Midblock
        self.encoder_downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.encoder_downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels))
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], 2*self.z_channels, kernel_size=3, padding=1)
        
        # Latent Dimension is 2*Latent because we are predicting mean & variance
        self.pre_quant_conv = nn.Conv2d(2*self.z_channels, 2*self.z_channels, kernel_size=1)
        ####################################################
        

        ##################### Decoder ######################
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))
        
        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.decoder_ups = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_ups.append(UpBlock("vae", self.down_channels[i], self.down_channels[i - 1],
                                               t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               attn=self.attns[i - 1],
                                               norm_channels=self.norm_channels))
        
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], self.im_channels, kernel_size=3, padding=1)
    
    def encode(self, x):
        out = self.encoder_conv_in(x)
        for down in self.encoder_downs:
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = self.silu(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape).to(device=x.device)
        return sample, out, mean, logvar
        
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for up in self.decoder_ups:
            out = up(out)

        out = self.decoder_norm_out(out)
        out = self.silu(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x):
        z, encoder_output, mean, logvar = self.encode(x)
        out = self.decode(z)
        return out, encoder_output, mean, logvar