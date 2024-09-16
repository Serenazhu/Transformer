from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math  


class SelfAttention(nn.Module):

    def __init__(self,config, causal=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # the embedding dimension is split across different attention heads.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.causal = causal
        if self.causal:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        qk = (q@k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        if self.causal:
            att = qk.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F. softmax(att, dim=-1)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, q, k, v):
        B, T, C = q.size()
        
        q = self.q_proj(q).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.k_proj(k).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.v_proj(v).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        qk = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(qk, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config, causal=False)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        attn = self.ln_1(self.attn(x) + x)
        attn = self.ln_2(self.mlp(x) + x) 
        return attn

class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config, causal = True)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x, encoder_output):
        x = self.ln_1(self.attn(x) + x)
        x = self.ln_2(self.cross_attn(x, encoder_output, encoder_output) + x)
        x = self.ln_3(self.mlp(x) + x) 
        return x


@dataclass
class TransformerConfig:
    block_size: int = 512
    vocab_size: int = 37000
    n_encoder_layer: int = 6
    n_decoder_layer: int = 6
    n_head: int = 8
    n_embd: int = 512

class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h_encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_encoder_layer)]),
            h_decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_decoder_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)