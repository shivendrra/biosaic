import torch
import torch.nn.functional as F

DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INDEX_TO_DNA = {v: k for k, v in DNA_VOCAB.items()}  # Reverse mapping

def dna_to_onehot(seq):
  seq_idx = [DNA_VOCAB[char] for char in seq]  # convert to indices
  return F.one_hot(torch.tensor(seq_idx), num_classes=4).float()  # shape: (L, 4)

def fix_dna_lines(file_path):
  with open(file_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines & strip whitespace

  merged_sequence = "".join(lines)  # Join all lines into a single sequence
  return merged_sequence