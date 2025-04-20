"""
  @dataset.py
    * contains Dataset class: special class for loading, formatting & creating batches of datasets
     - applicable only for DNA dataset training of VQ-VAE tokenizer
    * Raises:
        FileNotFoundError: invalid file path is given
        ValueError: data length is less than block size
        ValueError: if data is not loaded for performing train-test split
        IndexError: out of range index
    * Returns:
        torch.tensor(): return batches of tokenized DNA datasets"""

import torch
import torch.nn.functional as F
import biosaic
from typing import *
import os

class Dataset:
  """
    Initialize the Dataset
    Args:
      path (str): Path to the DNA data file
      kmer (int): kmer size for the tokenizer & encodings
      ratio (float): Fraction of data to use for testing (default 0.2)
      random_seed (int): random seeding for batching"""
  def __init__(self, path:str, kmer:int, ratio:float=0.25, random_seed:int=1600):
    self.path, self.ratio, self.random_seed  = path, ratio, random_seed
    self.kmer_size = kmer if kmer else 4
    self.tokenizer = biosaic.tokenizer(encoding=biosaic.get_encodings[3])
    self.n_classes = self.tokenizer.vocab_size
    self.train_data, self.val_data = "", ""
    self.load_and_format_data()

  def load_and_format_data(self):
    """
      Loads the file and formats the data:
        * Reads all lines
        * Strips whitespace and removes newline characters
        * Joins all lines into a single continuous string
        * Converts the string to uppercase"""
    if not os.path.isfile(self.path):
      raise FileNotFoundError(f"{self.path} does not exist.")

    with open(self.path, "r", encoding="utf-8") as f:
      raw_lines = f.readlines()

    # Remove empty lines, strip whitespace, and join into one continuous string.
    formatted_data = "".join(line.strip() for line in raw_lines if line.strip())
    self.data = formatted_data[:100000].upper()

  def tokenize(self, seq: str) -> List[str]:
    return self.tokenizer.tokenize(seq)

  def encode_seq(self, seq):
    kmer_encoded = self.tokenizer.encode(seq)
    return kmer_encoded

  def decode_ids(self, ids):
    kmer_decoded = self.tokenizer.decode(ids)
    return kmer_decoded

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

  def train_test_split(self):
    """
      Splits the formatted data into training and testing sets
      Returns:
        A tuple (train_data, test_data) containing the split strings"""
    if not self.data:
      raise ValueError("Data is not loaded. Please check the file content.")

    split_idx = int(len(self.data) * (1 - self.ratio))
    encoded_data = self.tokenizer.encode(self.data)
    self.train_data = self.tokens_to_onehot(encoded_data[:split_idx])
    self.test_data = self.tokens_to_onehot(encoded_data[split_idx:])
    return self.train_data, self.test_data

  def get_batch(self, split:str, batch_size:int, block_size:int, device:str="cpu"):
    """
      Samples a random batch of subsequences from the train or validation data
      Args:
        split (str): "train" or "val"
        batch_size (int): Number of samples in the batch
        block_size (int): Length of each subsequence
        device (str): Device to move the tensors to (e.g. "cpu" or "cuda")
      Returns:
        Tuple of tensors (x, y) where x is the input batch and y is the target batch
        The target is the input sequence shifted by one character"""
    train_data, val_data = self.train_test_split()
    data = train_data if split == "train" else val_data
    if len(data) < block_size + 1:
      raise ValueError("Data length is less than block size.")
    # randomly choose starting indices
    torch.manual_seed(self.random_seed)   ## changing random seeding
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])   # (B, L, n_classes)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # (B, L, n_classes)
    return x.to(device), y.to(device)

  def get_full_data(self):
    """
      Returns the full formatted DNA string"""
    return self.data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    if idx < 0 or idx >= len(self.data):
      raise IndexError("Index out of range.")
    return self.data[idx]