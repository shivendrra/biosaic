import torch
from torch.nn import functional as F
from .model import DNA_VQVAE, ModelConfig
import biosaic
from typing import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VQTokenizer:
  def __init__(self, kmer:int, device:str="cpu"):
    self.device, self.kmer = device, kmer
    if kmer < 6:
      self._tokenizer = biosaic.tokenizer(encoding=biosaic.get_encodings[kmer-1])
    else:
      raise ValueError(f"KMer size till 5 only supported!!")
    self._model = DNA_VQVAE(ModelConfig).to(self.device)
    model_path = '/content/drive/MyDrive/biosaic_30m.pth'
    self._model.load_state_dict(torch.load(model_path, map_location=self.device))
    self._model.eval()

  def __str__(self):
    return f"\t/Biosaic VQ-VAE tokenizer v1.0.1/\t"

  def tokens_to_onehot(self, ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
    # Convert list of token IDs into one-hot encoded tensor of shape (N, vocab_size)
    if isinstance(ids, list):
      ids = torch.tensor(ids, dtype=torch.long)
    return F.one_hot(ids, num_classes=self.n_classes).float() # shape (L, n_classes)

  def onehot_to_tokens(self, one_hot: torch.Tensor) -> List[int]:
    # Convert one-hot tensor back to list of token IDs
    if one_hot.dim() != 2 or one_hot.size(1) != self.tokenizer.vocab_size:
      raise ValueError(f"Expected one-hot of shape (N, {self.tokenizer.vocab_size})")
    return torch.argmax(one_hot, dim=-1).tolist()

  def encode(self, seq):
    seq = self._tokenizer.encode(seq)
    one_hot_seq = self.tokens_to_onehot(seq).unsqueeze(0).to(self.device)
    _, _, tokens = self._model(one_hot_seq)
    return tokens.sequeeze(0).cpu().numpy()

  def decode(self, ids):
    tokens = torch.tensor(tokens, dtype=torch.long).to(self.device)
    z_q = self._model.vq_layer.embeddings(tokens)
    with torch.no_grad():
      logits = self._model.decoder(z_q)
    decoded = self.onehot_to_tokens(logits)
    return self._tokenizer.decode(ids)

from .embedder import Evoformer, ModelConfig

# --- alphabet & reverse map (example: protein 20 aa + gap) ---
AMINO_ACIDS = [
  'A','R','N','D','C','Q','E','G','H','I',
  'L','K','M','F','P','S','T','W','Y','V','-'  # 21st for gap/pad
]
VOCAB = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
ID2AA = {i: aa for aa, i in VOCAB.items()}

class AFTokenizer:
  def __init__(self, model_path: str, msa_size: int, seq_len: int, pair_feat_dim: int, d_msa: int = 128, d_pair: int = 64, n_heads: int = 8, n_blocks: int = 4):
    """
    model_path: path to saved AlphaFoldTokenizer weights
    msa_size:  number of sequences in MSA input
    seq_len:   length of each sequence
    pair_feat_dim: number of pair features (C)
    """
    self.model = Evoformer(ModelConfig).to(DEVICE)
    self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    self.model.eval()

    self.msa_size = msa_size
    self.seq_len  = seq_len

  def _onehot(self, seq: str) -> torch.Tensor:
    """Convert a single sequence string to one-hot (L, A)."""
    idx = [VOCAB.get(ch, VOCAB['-']) for ch in seq]
    return F.one_hot(torch.tensor(idx), num_classes=len(VOCAB)).float()

  def encode(self, sequence: str, pair_feats: torch.Tensor = None) -> torch.Tensor:
    """
    Encode a single sequence (plus dummy MSA & pair features) into MSA embeddings.
    sequence: str of length <= seq_len
    pair_feats: optional (L,L,C) tensor; if None, uses zeros.
    
    Returns: msa_embeddings of shape (1, msa_size, seq_len, d_msa)
    """
    # Building an MSA batch by repeating the single seq
    onehot = self._onehot(sequence).to(DEVICE)           # (L, A)
    msa    = onehot.unsqueeze(0).repeat(self.msa_size, 1, 1)  # (N, L, A)
    msa    = msa.unsqueeze(0)                           # (1, N, L, A)

    # Pairing features
    if pair_feats is None:
      C = self.model.embed_pair.in_features
      pair_feats = torch.zeros(1, self.seq_len, self.seq_len, C, DEVICE=DEVICE)
    else:
      pair_feats = pair_feats.unsqueeze(0).to(DEVICE)    # (1, L, L, C)

    # Forward pass
    with torch.no_grad():
      msa_emb, pair_emb = self.model(msa, pair_feats)

    return msa_emb  # continuous embeddings

  def decode(self, msa_emb: torch.Tensor) -> str:
    """
    Decode MSA embeddings back to a single sequence by averaging over the MSA dimension
    and taking argmax over the output head.
    
    msa_emb: (1, N, L, d_msa)
    Returns: decoded sequence string of length L
    """
    pooled = msa_emb.mean(dim=1)  # (1, L, d_msa)    1) Average across the MSA sequences
    logits = self.model.msa_out(pooled)  # (1, L, A)     2) Project back to vocab logits
    idx = logits.argmax(dim=-1).squeeze(0).cpu().tolist()  # (L,)   3) Argmax to get indices
    return "".join(ID2AA[i] for i in idx)    # 4) Map back to amino acids

  def __str__(self):
    return f"\t/Biosaic Evoformer tokenizer v1.0.0/\t"