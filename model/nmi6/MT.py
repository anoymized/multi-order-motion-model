import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class GRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(GRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, hidden, x, shape):
        # horizontal
        b, l, c = hidden.shape
        h, w = shape
        hidden = hidden.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        hx = torch.cat([hidden, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * hidden, x], dim=1)))
        hidden = (1 - z) * hidden + z * q

        # vertical
        hx = torch.cat([hidden, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * hidden, x], dim=1)))
        hidden = (1 - z) * hidden + z * q

        return hidden.flatten(-2).permute(0, 2, 1)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        #
        # y_embed = (y_embed / 2) ** 2
        # x_embed = (x_embed / 2) ** 2

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            # using an exponential
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class ImageEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, 1, H, W]
        b, _, h, w = x.size()
        # repeat the image to 128 channels
        x = x.repeat(1, self.num_pos_feats, 1, 1)

        # to one dimension
        # x_embed = x.view(b, self.num_pos_feats, -1)
        x_embed = x * self.scale
        # using an exponential
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed / dim_t[None, :, None, None]
        pos_x = torch.cat((pos_x[:, 0::2, :, :].sin(), pos_x[:, 1::2, :, :].cos()), dim=1)

        return pos_x


def feature_add_position(feature0, image=None, feature_channels=None, scale=0.45):
    temp = torch.mean(abs(feature0))
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)
    # position = PositionalEncodingPermute2D(feature_channels)(feature0)
    position = pos_enc(feature0)
    feature0 = feature0 + (temp * position / position.mean()) * scale * torch.pi

    if image is not None:
        emb = ImageEmbeddingSine(num_pos_feats=feature_channels)
        position = emb(image)
        feature0 = feature0 + (temp * position / position.mean()) * scale * torch.pi * 5

    feature0 = feature0 * temp / torch.mean(abs(feature0), dim=(1, 2, 3), keepdim=True)
    return feature0


def feature_add_image_content(feature0, add_fea, scale=0.4):
    temp = torch.mean(abs(feature0))
    position = add_fea
    feature0 = feature0 + (temp * position / position.mean()) * scale * torch.pi
    feature0 = feature0 * temp / torch.mean(abs(feature0), dim=(1, 2, 3), keepdim=True)
    return feature0


class AttUp(nn.Module):
    def __init__(self,
                 c=512
                 ):
        super(AttUp, self).__init__()
        self.proj = nn.Linear(c, c, bias=False)
        self.norm = nn.LayerNorm(c)
        self.conv = nn.Sequential(nn.Conv2d(2 * c, c, kernel_size=1, stride=1, padding=0),
                                  nn.GELU(),
                                  nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
                                  nn.GELU(),
                                  nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
                                  nn.GELU()
                                  )
        self.gru = SepConvGRU(c, c)

    def forward(self, att, message, shape):
        # q, k, v: [B, L, C]
        b, l, c = att.shape
        h, w = shape
        message = self.norm(self.proj(message)).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        att = att.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        message = self.conv(torch.cat([att, message], dim=1))
        att = self.gru(att, message).flatten(-2).permute(0, 2, 1)
        # [B, H*W, C]
        return att


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=1,
                 no_ffn=False,
                 ffn_dim_expansion=4
                 ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.no_ffn = no_ffn
        # multi-head attention
        self.att_proj = nn.Sequential(nn.Linear(d_model, d_model, bias=False), nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model, bias=False))
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.gru = GRU(d_model, d_model)
        self.attn_updater = AttUp(d_model)
        self.drop = nn.Dropout(p=0.8)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, att, value,
                shape, iteration=0):
        # source, target: [B, L, C]
        max_exp_scale = 3 * torch.pi
        # single-head attention
        B, L, C = value.shape
        val_proj = self.v_proj(value)
        att_proj = self.att_proj(att)  # [B, L, C]
        norm_fac = torch.sum(att_proj ** 2, dim=-1, keepdim=True) ** 0.5
        scale = max_exp_scale * torch.sigmoid(torch.mean(att_proj, dim=[-1, -2], keepdim=True)) + 1
        A = torch.exp(scale * torch.matmul(att_proj / norm_fac, att_proj.permute(0, 2, 1) / norm_fac.permute(0, 2, 1)))
        A = A / A.max()
        # I = torch.eye(A.shape[-1], device=A.device).unsqueeze(0)
        # # A[I.repeat(B, 1, 1) == 1] = 1e-6  # remove self-prop
        D = torch.sum(A, dim=-1, keepdim=True)
        D = 1 / (torch.sqrt(D) + 1e-6)  # normalized node degrees
        A = D * A * D.transpose(-1, -2)

        # A = torch.softmax(A , dim=2)  # [B, L, L]
        message = torch.matmul(A, val_proj)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)
        if not self.no_ffn:
            message = self.mlp(torch.cat([value, message], dim=-1))
            message = self.norm2(message)

        # if iteration > 2:
        #     message = self.drop(message)

        att = self.attn_updater(att, message, shape)
        value = self.gru(value, message, shape)
        return value, att, A


class FeatureTransformer(nn.Module):
    def __init__(self,
                 num_layers=6,
                 d_model=128
                 ):
        super(FeatureTransformer, self).__init__()
        self.d_model = d_model
        # self.layers = nn.ModuleList([TransformerLayer(self.d_model, no_ffn=False, ffn_dim_expansion=2)
        #                              for i in range(num_layers)])
        self.layers = TransformerLayer(self.d_model, no_ffn=False, ffn_dim_expansion=2)
        self.re_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.num_layers = num_layers
        self.norm_sigma = nn.Parameter(torch.tensor(1.0, requires_grad=True), requires_grad=True)
        self.norm_k = nn.Parameter(torch.tensor(1.8, requires_grad=True), requires_grad=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def normalize(self, x):  # TODO
        sum_activation = torch.mean(x, dim=[1, 2], keepdim=True) + torch.square(self.norm_sigma)
        x = self.norm_k.abs() * x / sum_activation
        return x

    def forward(self, feature0, image=None):
        feature_list = []
        attn_list = []
        attn_viz_list = []
        b, c, h, w = feature0.shape
        assert self.d_model == c
        value = feature0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        att = feature_add_position(feature0, image, c)
        att = att.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        for i in range(self.num_layers):
            value, att, A = self.layers(att=att, value=value, shape=[h, w], iteration=i)
            value_decode = self.normalize(
                torch.square(self.re_proj(value)))  # map to motion energy, Do use normalization here
            # print("value_decode",value_decode.abs().mean())
            attn_list.append(att.view(b, h, w, c).permute(0, 3, 1, 2).contiguous())
            attn_viz_list.append(A.reshape(b, h, w, h, w).contiguous())
            feature_list.append(value_decode.view(b, h, w, c).permute(0, 3, 1, 2).contiguous())
        # reshape back
        return feature_list, attn_list, attn_viz_list

    def forward_save_mem(self, feature0, image=None):
        feature_list = []
        attn_list = []
        attn_viz_list = []
        b, c, h, w = feature0.shape
        assert self.d_model == c
        value = feature0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        att = feature_add_position(feature0, image, c)
        att = att.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        for i in range(self.num_layers):
            value, att, _ = self.layers(att=att, value=value, shape=[h, w], iteration=i)
            value_decode = self.normalize(
                torch.square(self.re_proj(value)))  # map to motion energy, Do use normalization here
            # print("value_decode",value_decode.abs().mean())
            attn_list.append(att.view(b, h, w, c).permute(0, 3, 1, 2).contiguous())
            feature_list.append(value_decode.view(b, h, w, c).permute(0, 3, 1, 2).contiguous())
        # reshape back
        return feature_list, attn_list

    @staticmethod
    def demo():
        import time
        frame_list = torch.randn([4, 256, 64, 64], device="cuda")
        model = FeatureTransformer(6, 256).cuda()
        for i in range(100):
            start = time.time()
            output = model(frame_list)

            torch.mean(output[-1][-1]).backward()
            end = time.time()
            print(end - start)
            print("#================================++#")


if __name__ == '__main__':
    FeatureTransformer.demo()
