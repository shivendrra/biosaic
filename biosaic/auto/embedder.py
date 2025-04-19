import torch
import torch.nn as nn

class ModelConfig:
  DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  A            = 4        # DNA alphabet
  C            = 21       # 21 letter for amino acid & 4 for dna
  d_msa        = 128
  d_pair       = 64
  n_heads      = 8
  n_blocks     = 4

class RowAttention(nn.Module):
  def __init__(self, d_msa, n_heads):
    super().__init__()
    self.attn = nn.MultiheadAttention(d_msa, n_heads, batch_first=True)
  def forward(self, msa):  # msa: (B, N, L, d_msa)
    B, N, L, D = msa.shape
    x = msa.view(B*L, N, D)  # treat each position across sequences as a sequence
    out, _ = self.attn(x, x, x)
    return out.view(B, N, L, D)

class ColAttention(nn.Module):
  def __init__(self, d_msa, n_heads):
    super().__init__()
    self.attn = nn.MultiheadAttention(d_msa, n_heads, batch_first=True)
  def forward(self, msa):
    B, N, L, D = msa.shape
    x = msa.permute(0,2,1,3).reshape(B* N, L, D)  # each sequence across positions
    out, _ = self.attn(x, x, x)
    return out.view(B, L, N, D).permute(0,2,1,3)

class TriMulUpdate(nn.Module):
  def __init__(self, d_pair):
    super().__init__()
    self.linear_a = nn.Linear(d_pair, d_pair)
    self.linear_b = nn.Linear(d_pair, d_pair)
  def forward(self, pair):
    # pair: (B, L, L, d_pair)
    left = self.linear_a(pair)    # (B,L,L,d)
    right= self.linear_b(pair)    # (B,L,L,d)
    # outer product along one axis
    # simplistic: new_pair[i,j] += sum_k left[i,k] * right[k,j]
    return pair + torch.einsum("bikd,bkjd->bijd", left, right)

class Block(nn.Module):
  def __init__(self, d_msa, d_pair, n_heads):
    super().__init__()
    self.row_attn = RowAttention(d_msa, n_heads)
    self.col_attn = ColAttention(d_msa, n_heads)
    self.tri_mul = TriMulUpdate(d_pair)
    # plus feed‑forwards, layernorms, gating, etc.

  def forward(self, msa, pair):
    msa = msa + self.row_attn(msa)
    msa = msa + self.col_attn(msa)
    pair= pair + self.tri_mul(pair)
    return msa, pair

class Evoformer(nn.Module):
  def __init__(self, params: ModelConfig):
    """
      A: alphabet size (e.g. 4 for DNA, 21 for protein)
      C: number of initial pair features
    """
    super().__init__()
    self.embed_msa  = nn.Linear(ModelConfig.A, ModelConfig.d_msa)
    self.embed_pair = nn.Linear(ModelConfig.C, ModelConfig.d_pair)
    self.blocks     = nn.ModuleList([
      Block(ModelConfig.d_msa, ModelConfig.d_pair, ModelConfig.n_heads)
      for _ in range(ModelConfig.n_blocks)
    ])
    # for masked token prediction
    self.msa_out = nn.Linear(ModelConfig.d_msa, ModelConfig.A)
  def forward(self, msa, pair):
    # msa: (B, N, L, A); pair: (B, L, L, C)
    msa  = self.embed_msa(msa)
    pair = self.embed_pair(pair)
    for blk in self.blocks:
      msa, pair = blk(msa, pair)
    # return logits for each msa position
    return self.msa_out(msa), pair