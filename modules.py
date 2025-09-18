import numpy as np
import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm

class LearnedAggregationLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim: int = dim // num_heads
        self.scale = qk_scale if qk_scale is not None else head_dim**-0.5

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.id = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, N, C = x.shape
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        q = self.q(cls_tokens).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = self.id(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls.squeeze()

class ProtProjector(nn.Module):
    def __init__(self, target_dim, latent_dim, activation, num_heads=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.proj = LearnedAggregationLayer(target_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.input_proj = nn.Linear(target_dim, latent_dim)
        self.res = nn.Sequential(
            nn.LayerNorm(target_dim),
            activation(),
            nn.Linear(target_dim, latent_dim),
            nn.Dropout(proj_drop),
            nn.LayerNorm(latent_dim),
            activation(),
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(proj_drop),
            nn.LayerNorm(latent_dim),
            activation(),
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(proj_drop),
            nn.LayerNorm(latent_dim),
        )

        self.non_linearity = activation()

    def forward(self, x):
        proj = self.proj(x)
        proj = self.res(proj)
        return self.non_linearity(proj)

class DrugProjector(nn.Module):
    def __init__(self, target_dim, latent_dim, activation, num_heads=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.proj = LearnedAggregationLayer(target_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.input_proj = nn.Linear(target_dim, latent_dim)
        self.res = nn.Sequential(
            nn.LayerNorm(target_dim),
            activation(),
            nn.Linear(target_dim, latent_dim),
            nn.Dropout(proj_drop),
            nn.LayerNorm(latent_dim),
            activation(),
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(proj_drop),
            nn.LayerNorm(latent_dim),
            activation(),
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(proj_drop),
            nn.LayerNorm(latent_dim),
        )

        self.non_linearity = activation()

    def forward(self, x):
        proj = self.proj(x)
        proj = self.res(proj)
        return self.non_linearity(proj)

class BAN(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3):
        super(BAN, self).__init__()

        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout)
        if 1 < k:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out <= self.c:
            self.h_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

        self.bn = nn.BatchNorm1d(h_dim)

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if 1 < self.k:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.p_net(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, v, q, softmax=False):
        v_num = v.size(1)
        q_num = q.size(1)
        if self.h_out <= self.c:
            v_ = self.v_net(v)
            q_ = self.q_net(q)
            att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.h_mat, v_, q_)) + self.h_bias
        else:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)
            att_maps = self.h_net(d_.transpose(1, 2).transpose(2, 3))
            att_maps = att_maps.transpose(2, 3).transpose(1, 2)
        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, v_num * q_num), 2)
            att_maps = p.view(-1, self.h_out, v_num, q_num)
        logits = self.attention_pooling(v_, q_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits_i = self.attention_pooling(v_, q_, att_maps[:, i, :, :])
            logits += logits_i
        logits = self.bn(logits)
        return logits, att_maps

class FCNet(nn.Module):
    def __init__(self, dims, act='ReLU', dropout=0):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if 0 < dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if '' != act:
                layers.append(getattr(nn, act)())
        if 0 < dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if '' != act:
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.bn2 = nn.BatchNorm1d(input_dim // 2)
        self.fc3 = nn.Linear(input_dim // 2, input_dim // 4)
        self.bn3 = nn.BatchNorm1d(input_dim // 4)
        self.output = nn.Linear(input_dim // 4, 1)

    def forward(self, x):
        x = self.bn1(torch.relu(self.fc1(x)))
        x = self.bn2(torch.relu(self.fc2(x)))
        x = self.bn3(torch.relu(self.fc3(x)))
        x = torch.sigmoid(self.output(x))
        return x

class Clip(nn.Module):
    def __init__(self, temperature=0.05):
        super(Clip, self).__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, features_1, features_2):
        N = features_1.size()[0]
        cat_features_1 = torch.cat([features_1, features_2])
        cat_features_2 = torch.cat([features_2, features_1])
        features_1 = cat_features_1 / cat_features_1.norm(dim=1, keepdim=True)
        features_2 = cat_features_2 / cat_features_2.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_f1 = logit_scale * features_1 @ features_2.t()
        labels = torch.arange(2 * N).long().to(logits_per_f1.device)
        loss = self.loss_fun(logits_per_f1, labels) / 2
        return loss, logits_per_f1