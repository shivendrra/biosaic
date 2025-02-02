import torch
from torch.nn import functional as F
from .model import DNA_VQVAE, ModelConfig

DNA_VOCAB = {"A": 0, "T": 1, "C": 2, "G": 3}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DNA_VQVAE(ModelConfig).to(device)

def dna_to_onehot(seq):
  seq_idx = [DNA_VOCAB[char] for char in seq]
  one_hot = F.one_hot(torch.tensor(seq_idx), num_classes=4)
  return one_hot.float()

def tokenize_dna(seq):
  one_hot_seq = dna_to_onehot(seq).unsqueeze(0).to(device)  # Add batch dim
  with torch.no_grad():
    _, _, tokens = model(one_hot_seq)
  return tokens.squeeze(0).cpu().numpy()  # Remove batch dim

def detokenize_dna(tokens):
  tokens = torch.tensor(tokens, dtype=torch.long).to(device)  # convert to tensor
  z_q = model.vq_layer.embedding(tokens)  # convert tokens to embeddings

  with torch.no_grad():
    logits = model.decoder(z_q)  # decode to DNA sequence logits
    decoded_onehot = torch.argmax(logits, dim=-1)  # most likely bases

  # mapping one-hot indices back to DNA characters
  INDEX_TO_DNA = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
  decoded_seq = ''.join(INDEX_TO_DNA[idx.item()] for idx in decoded_onehot.squeeze(0))  
  return decoded_seq

class tokenizer:
  def __init__(self):
    self.vocab = DNA_VOCAB