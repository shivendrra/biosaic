import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INDEX_TO_DNA = {v: k for k, v in DNA_VOCAB.items()}  # Reverse mapping

def dna_to_onehot(seq):
  seq_idx = [DNA_VOCAB[char] for char in seq]  # convert to indices
  return F.one_hot(torch.tensor(seq_idx), num_classes=4).float()  # shape: (L, 4)

class DNADataset(Dataset):
  def __init__(self, file_path):
    with open(file_path, 'r') as f:
      self.sequences = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines

  def __len__(self): return len(self.sequences)
  def __getitem__(self, idx): return self.sequences[idx]

def pad_collate_fn(batch):
  max_len = max(len(seq) for seq in batch)  # finding longest sequence in batch
  
  padded_seqs = []
  for seq in batch:
    padded_seq = seq.ljust(max_len, 'N')  # padding with 'N' (unknown nucleotide)
    padded_seqs.append(padded_seq)

  one_hot_tensors = torch.stack([dna_to_onehot(seq) for seq in padded_seqs])
  return one_hot_tensors  # shape: (batch_size, max_len, 4)

def get_dna_dataloader(file_path, batch_size=16, shuffle=True):
  dataset = DNADataset(file_path)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate_fn)