import torch
from torch.nn import functional as F
from .model import DNA_VQVAE, ModelConfig

# --- DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DNA_VOCAB = {"A": 0, "T": 1, "C": 2, "G": 3}
INDEX_TO_DNA = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

class VQTokenizer:
  def __init__(self, model_path):
    self.vocab = DNA_VOCAB
    self.ids_to_dna = INDEX_TO_DNA
    self.DEVICE = DEVICE
    self.model = DNA_VQVAE(ModelConfig).to(DEVICE)
    self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    self.model.eval()

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
    one_hot_seq = self.dna_to_onehot(seq).unsqueeze(0).to(DEVICE)
    _, _, tokens = self.model(one_hot_seq)
    return tokens.sequeeze(0).cpu().numpy()

  def decode(self, tokens: list):
    tokens = torch.tensor(tokens, dtype=torch.long).to(DEVICE)
    z_q = self.model.vq_layer.embeddings(tokens)

    with torch.no_grad():
      logits = self.model.decoder(z_q)
    decoded = self.onehot_to_dna(logits)
    return decoded

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