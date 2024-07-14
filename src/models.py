from   copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from   einops import rearrange, repeat
from   scipy.optimize import linear_sum_assignment


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def count_parameters(net: torch.nn.Module):
    return sum(p.numel() for p in net.parameters())
      

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 32768):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos):
        freqs = torch.einsum('i , j -> i j', pos, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    rot_dim, T      = freqs.shape[-1], t.shape[-2]
    freqs           = freqs[-T:]
    t, t_unrotated  = t[..., :rot_dim], t[..., rot_dim:]
    t               = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_unrotated), dim = -1)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head=None, dropout=0.0):
        super().__init__()
        self.dim        = dim
        self.dim_head   = default(dim_head, dim//heads)
        self.heads      = heads
        self.drop       = dropout
        self.to_q       = nn.Linear(dim, self.dim_head*heads,   bias=False)
        self.to_kv      = nn.Linear(dim, self.dim_head*heads*2, bias=False)
        self.to_out     = nn.Linear(self.dim_head*heads, dim,   bias=False)
      
    def forward(self, x, context, input_mask, mask_context, rotary_emb):
        q       = self.to_q(x)
        k, v    = self.to_kv(context).chunk(2, dim=-1)    
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q, k    = map(lambda t: apply_rotary_pos_emb(t, rotary_emb) if exists(rotary_emb) else t, (q, k))

        mask = rearrange(mask_context, 'b j -> b 1 1 j') if exists(mask_context) else None

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.0, is_causal=False)
        x = rearrange(x, 'b h t d -> b t (h d)')
        x = self.to_out(x) 
        x = x.masked_fill(~input_mask.unsqueeze(-1), 0.) if exists(input_mask) else x
        return x


def FeedForward(dim, ff_mult, drop=0.1):
    return nn.Sequential(nn.Linear(dim, int(dim*ff_mult)),
                         nn.GELU(),
                         nn.Dropout(drop),
                         nn.Linear(int(dim*ff_mult), dim),
                         nn.Dropout(drop))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, ff_mult, dropout=0.0):
        super().__init__()
        self.ff     = FeedForward(dim, ff_mult, dropout)
        self.attn   = Attention(dim, heads, dropout=dropout)
        self.norms  = nn.ModuleList([RMSNorm(dim) for _ in range(2)])

    def forward(self, x, mask, rotary_emb):
        x = x + self.sa_block(self.norms[0](x), mask, rotary_emb)
        x = x + self.ff(self.norms[1](x))
        return x
    
    def sa_block(self, x, mask, rotary_emb):
        return self.attn(x, x, mask, mask, rotary_emb)
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, heads, ff_mult, dropout=0.0):
        super().__init__()
        self.sa_attn    = Attention(dim, heads, dropout=dropout)
        self.ca_attn    = Attention(dim, heads, dropout=dropout)
        self.ff         = FeedForward(dim, ff_mult, dropout)
        self.norms      = nn.ModuleList([RMSNorm(dim) for _ in range(3)])

    def forward(self, x, context, mask, mask_context, rotary_emb):
        x = x + self.sa_block(self.norms[0](x), mask, rotary_emb)
        x = x + self.ca_block(self.norms[1](x), context, mask, mask_context)
        x = x + self.ff(self.norms[2](x))
        return x
    
    def sa_block(self, x, mask, rotary_emb):
        return self.sa_attn(x, x, mask, mask, rotary_emb)
    
    def ca_block(self, x, context, mask, mask_context):
        return self.ca_attn(x, context, mask, mask_context, None)
    

class Transformer(nn.Module):
    def __init__(self, layers, dim_out=None):
        super().__init__()
        attn_layer    = layers[0].sa_attn if hasattr(layers[0], 'sa_attn') else layers[0].attn
        dim, dim_head = attn_layer.dim, attn_layer.dim_head
        self.layers     = nn.ModuleList(layers)
        self.rot_emb    = RotaryEmbedding(dim_head)
        self.proj       = nn.Linear(dim, dim_out) if exists(dim_out) else nn.Identity()
        self.norm       = RMSNorm(dim)

    def forward(self, x, context=None, mask=None, mask_context=None):
        # Rotary embedding
        rotary_emb = self.rot_emb(torch.arange(x.shape[1], device=x.device, dtype=x.dtype))

        # Transformer
        for l in self.layers:
            x = l(x, context, mask, mask_context, rotary_emb)
        
        # Project
        x = self.proj(self.norm(x))

        return x
    

class Residual(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
    def forward(self, x):
        return x + self.f(x)


def Repeat(module, N):
    return nn.Sequential(*[deepcopy(module) for _ in range(N)])


def Conv(c1, c2, k=1, s=1, p=None, g=None):
    return nn.Sequential(nn.Conv2d(c1, c2, k, s, default(p,k//2), groups=default(g,1), bias=False), nn.BatchNorm2d(c2), nn.SiLU(True))


def Bottleneck(c1, c2=None, k=(3, 3), shortcut=True, g=1, e=0.5):
    c2  = default(c2, c1)
    c_  = int(c2 * e)
    net = nn.Sequential(Conv(c1, c_, k[0]), Conv(c_, c2, k[1], g=g))
    return Residual(net) if shortcut else net


class CspBlock(nn.Module):
    def __init__(self, c1, c2, f=1, e=1, n=1):
        super().__init__()
        c_      = int(c2 * f)  # hidden channels
        self.d  = Conv(c1, c_, 3, s=2)
        self.c1 = Conv(c_, c2, 1)
        self.c2 = Conv(c_, c2, 1)
        self.m  = Repeat(Bottleneck(c2, e=e, k=(1,3)), n)
        self.c3 = Conv(c2, c2, 1)
        self.c4 = Conv(2*c2, c_, 1)
    def forward(self, x):
        x = self.d(x)
        a = self.c1(x)
        b = self.c3(self.m(self.c2(x)))
        x = self.c4(torch.cat([b, a], 1))
        return x


class CspDarknet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Conv(3, 32, 3)
        self.b1   = CspBlock( 32,  64, f=1, e=0.5, n=1)
        self.b2   = CspBlock( 64,  64, f=2, e=1,   n=2)
        self.b3   = CspBlock(128, 128, f=2, e=1,   n=8)
        self.b4   = CspBlock(256, 256, f=2, e=1,   n=8)
        self.b5   = CspBlock(512, 512, f=2, e=1,   n=4)
    def forward(self, x):
        p8  = self.b3(self.b2(self.b1(self.stem(x))))
        p16 = self.b4(p8)
        p32 = self.b5(p16)
        return p8, p16, p32
    

def bbox_iou(box1, box2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = map(lambda t: t.transpose(-2,-1), box2.chunk(4, -1))
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
            ) / 4  # center dist**2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class DeTr(nn.Module):
    def __init__(self, num_classes, max_dets=100):
        super().__init__()
        dim             = 512
        self.enc        = CspDarknet()
        self.proj       = nn.Linear(1024, dim)
        self.dec        = Transformer([TransformerDecoderLayer(dim=dim, heads=4, ff_mult=2.0, dropout=0.) for _ in range(4)])
        self.q          = nn.Parameter(torch.randn(max_dets, dim))
        self.proj_bbox  = nn.Linear(dim, 4)
        self.proj_cls   = nn.Linear(dim, num_classes)

    def forward(self, x, targets=None):
        x   = self.enc(x)[-1]
        x   = self.proj(rearrange(x, 'b f h w -> b (h w) f'))
        q   = self.dec(repeat(self.q, 'd f -> b d f', b=x.shape[0]), x)
        box = self.proj_bbox(q).relu_()
        cls = self.proj_cls(x)

        if exists(targets):
            B, D = targets.shape[0], targets.shape[1]
            ious = bbox_iou(targets[...,:4], box, CIoU=True)
        return torch.cat((box, cls.sigmoid()), 2)    