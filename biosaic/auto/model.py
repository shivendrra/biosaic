class ModelConfig:
  d_model: int= 128
  in_dim: int= 4
  n_embed: int=512
  beta: float=0.25
  n_heads: int= 8
  n_layers: int= 8

import torch
import torch.nn as nn

class encoder(nn.Module):
  def __init__(self, _in, d_model, n_layers, n_heads):
    super().__init__()
    self.embed = nn.Linear(_in, d_model)
    self.encoder = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads),
      num_layers=n_layers)

  def forward(self, x):
    x = self.embed(x)
    x = x.permute(1, 0, 2)  # (L, B, d_model)
    z_e = self.encoder(x) # Transformer encoding
    return z_e.permute(1, 0, 2) # Back to (B, L, vocab_size)

class decoder(nn.Module):
  def __init__(self, d_model, _out, n_layers, n_heads):
    super().__init__()
    self.decoder = nn.TransformerDecoder(
      nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads),
      num_layers=n_layers
    )
    self.fc_out = nn.Linear(d_model, _out)  # Output logits

  def forward(self, z_q):
    z_q = z_q.permute(1, 0, 2)  # (L, B, d_model)
    x_recon = self.decoder(z_q, z_q)  # Transformer decoding
    x_recon = self.fc_out(x_recon.permute(1, 0, 2))  # Back to (B, L, vocab_size)
    return x_recon

class Quantizer(nn.Module):
  def __init__(self, n_embed, d_model, beta):
    super().__init__()
    self.n_embed, self.d_model, self.beta = n_embed, d_model, beta
    self.embeddings = nn.Embedding(n_embed, d_model)
    self.embeddings.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

  def forward(self, z_e):
    z_e_flat = z_e.reshape(-1, self.d_model)
    distances = torch.cdist(z_e_flat, self.embeddings.weight)
    encoding_indices = torch.argmin(distances, dim=1)
    z_q = self.embeddings(encoding_indices).view(z_e.shape)
    loss = self.beta * torch.mean((z_q.detach() - z_e) ** 2) + torch.mean((z_e.detach() - z_q) ** 2)

    z_q = z_e + (z_q - z_e).detach()
    return z_q, loss, encoding_indices.view(z_e.shape[:-1])

class DNA_VQVAE(nn.Module):
  def __init__(self, args: ModelConfig):
    super().__init__()
    self.encoder = encoder(args.in_dim, args.d_model, args.n_layers, args.n_heads)
    self.vq_layer = Quantizer(args.n_embed, args.d_model, args.beta)
    self.decoder = decoder(args.d_model, args.in_dim, args.n_layers, args.n_heads)

  def forward(self, x):
    z_e = self.encoder(x)
    z_q, vq_loss, indices = self.vq_layer(z_e)
    x_recon = self.decoder(z_q)
    return x_recon, vq_loss, indices