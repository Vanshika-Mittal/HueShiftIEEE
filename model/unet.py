import torch
import torch.nn as nn


def get_time_embedding(time_steps, temb_dim):
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    factor = 10000 ** (
        torch.arange(
            start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device
        )
        / (temb_dim // 2)
    )
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        down_sample,
        num_heads,
        num_layers,
        attn,
        norm_channels,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.t_emb_dim = t_emb_dim

        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        norm_channels, in_channels if i == 0 else out_channels
                    ),
                    nn.SiLU(),
                    nn.Conv2d(
                        in_channels if i == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
                for i in range(num_layers)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList(
                [
                    nn.Sequential(nn.SiLU(), nn.Linear(self.t_emb_dim, out_channels))
                    for _ in range(num_layers)
                ]
            )

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, stride=1, padding=1
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)]
            )
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels if i == 0 else out_channels, out_channels, kernel_size=1
                )
                for i in range(num_layers)
            ]
        )

        self.down_sample_conv = (
            nn.Conv2d(out_channels, out_channels, 4, 2, 1)
            if self.down_sample
            else nn.Identity()
        )

    def forward(self, x, t_emb=None):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        out = self.down_sample_conv(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, model_config):
        super().__init__()
        self.down_channels = model_config["DOWN_CHANNELS"]
        self.mid_channels = model_config["MID_CHANNELS"]
        self.t_emb_dim = model_config["TIME_EMB_DIM"]
        self.down_sample = model_config["DOWN_SAMPLE"]
        self.num_down_layers = model_config["NUM_DOWN_LAYERS"]
        self.num_mid_layers = model_config["NUM_MID_LAYERS"]
        self.num_up_layers = model_config["NUM_UP_LAYERS"]
        self.attns = model_config["ATTN"]
        self.norm_channels = model_config["NORM_CHANNELS"]
        self.num_heads = model_config["NUM_HEADS"]
        self.conv_out_channels = model_config["CONV_OUT_CHANNELS"]

        self.conv_in = nn.Conv2d(
            in_channels, self.down_channels[0], kernel_size=3, padding=1
        )

        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
        )

        self.downs = nn.ModuleList(
            [
                DownBlock(
                    self.down_channels[i],
                    self.down_channels[i + 1],
                    t_emb_dim=self.t_emb_dim,
                    down_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    num_layers=self.num_down_layers,
                    attn=self.attns[i],
                    norm_channels=self.norm_channels,
                )
                for i in range(len(self.down_channels) - 1)
            ]
        )

        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(
            self.conv_out_channels, out_channels, kernel_size=3, padding=1
        )

    def forward(self, x, t):
        out = self.conv_in(x)

        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        for down in self.downs:
            out = down(out, t_emb)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out
