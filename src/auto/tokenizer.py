import torch
from torch.nn import functional as F
from .model import DNA_VQVAE, ModelConfig

DNA_VOCAB = {"A": 0, "T": 1, "C": 2, "G": 3}
INDEX_TO_DNA = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DNA_VQVAE(ModelConfig).to(device)

class tokenizer:
  def __init__(self):
    self.vocab = DNA_VOCAB
    self.ids_to_dna = INDEX_TO_DNA
    self.device = device
    self.model = model

  def __str__(self):
    return f"\t/Biosaic VQ-VAE tokenizer v1.0.0/\t"

  def dna_to_onehot(self, seq):
    seq_idx = [DNA_VOCAB[char] for char in seq]
    one_hot = F.one_hot(torch.tensor(seq_idx), num_classes=4)
    return one_hot.float()

  def onehot_to_dna(self, logits):
    decoded_out = torch.argmax(logits, dim=-1)
    decoded = ''.join(self.ids_to_dna[idx.item()] for idx in decoded_out.squeeze(0))
    return decoded

  def encode(self, seq: str):
    one_hot_seq = self.dna_to_onehot(seq).unsqueeze(0).to(device)
    _, _, tokens = self.model(one_hot_seq)
    return tokens.sequeeze(0).cpu().numpy()

  def decode(self, tokens: list):
    tokens = torch.tensor(tokens, dtype=torch.long).to(device)
    z_q = self.model.vq_layer.embeddings(tokens)

    with torch.no_grad():
      logits = self.model.decoder(z_q)
    decoded = self.onehot_to_dna(logits)
    return decoded