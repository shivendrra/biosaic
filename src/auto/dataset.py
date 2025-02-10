import torch
import torch.nn.functional as F

DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INDEX_TO_DNA = {v: k for k, v in DNA_VOCAB.items()}  # Reverse mapping

class Dataset:
  def __init__(self, path:str):
    self.path = path

  def load_simple(self):
    with open(self.path, "r", encoding="utf-8") as f:
      lines = [line.strip() for line in f.readlines() if line.strip()]  # removing empty lines & strip whitespace
    merged_sequence = "".join(lines)  # joining all lines into a single sequence
    return merged_sequence

  def load_encoded(self, seq=None):
    if seq:
      loaded_sequences = seq
    else:
      loaded_sequences = self.load_simple()
    seq_idx = [DNA_VOCAB[char] for char in loaded_sequences]
    return F.one_hot(torch.tensor(seq_idx), num_classes=4) # shape (L, 4)

  def train_test_split(self, sequence:str=None, ratio:float=0.8):
    split_size = int(0.8 * len(sequence))
    sequence = self.load_encoded(seq=sequence)

    train_data = sequence[:split_size]
    test_data = sequence[split_size:]
    return train_data, test_data