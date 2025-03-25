"""
  @dataset.py
    * contains Dataset class: special class for loading, formatting & creating batches of datasets
     - applicable only for DNA dataset training of VQ-VAE tokenizer (will be modified for model training support)
    * Raises:
        FileNotFoundError: invalid file path is given
        ValueError: data length is less than block size
        ValueError: if data is not loaded for performing train-test split
        IndexError: out of range index
    * Returns:
        torch.tensor(): return batches of tokenized DNA datasets"""

import torch
import torch.nn.functional as F
import os

class Dataset:
  def __init__(self, file_path, ratio:float=0.25):
    """
      Initialize the Dataset
      Args:
        file_path (str): Path to the DNA data file
        test_split (float): Fraction of data to use for testing (default 0.2)"""
    self.file_path = file_path
    self.test_split = ratio
    self.data = ""
    self.train_data = ""
    self.val_data = ""
    self.load_and_format_data()
    self.train_test_split()

  def load_and_format_data(self):
    """
      Loads the file and formats the data:
        - Reads all lines
        - Strips whitespace and removes newline characters
        - Joins all lines into a single continuous string
        - Converts the string to uppercase"""
    if not os.path.isfile(self.file_path):
      raise FileNotFoundError(f"{self.file_path} does not exist.")
      
    with open(self.file_path, "r", encoding="utf-8") as f:
      raw_lines = f.readlines()
      
    # Remove empty lines, strip whitespace, and join into one continuous string.
    formatted_data = "".join(line.strip() for line in raw_lines if line.strip())
    self.data = formatted_data.upper()
  
  @staticmethod
  def dna_to_onehot(seq):
    """
      Converts a DNA sequence (string) into a one-hot encoded tensor
      Only characters 'A', 'C', 'G', 'T' are considered
      Args:
        seq (str): DNA sequence
      Returns:
        Tensor of shape (L, 4) where L is the length of the sequence"""
    DNA_VOCAB = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # convert each character to an index
    indices = [DNA_VOCAB[char] for char in seq if char in DNA_VOCAB]
    onehot = F.one_hot(torch.tensor(indices), num_classes=4).float()
    return onehot

  def get_batch(self, split, batch_size, block_size, device="cpu"):
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
    data = self.train_data if split == "train" else self.val_data
    if len(data) < block_size + 1:
      raise ValueError("Data length is less than block size.")
    # randomly choose starting indices
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([self.dna_to_onehot(data[i:i+block_size]) for i in ix])
    y = torch.stack([self.dna_to_onehot(data[i+1:i+block_size+1]) for i in ix])
    return x.to(device), y.to(device)

  def train_test_split(self):
    """
      Splits the formatted data into training and testing sets
      Returns:
        A tuple (train_data, test_data) containing the split strings"""
    if not self.data:
      raise ValueError("Data is not loaded. Please check the file content.")

    split_idx = int(len(self.data) * (1 - self.test_split))
    self.train_data = self.data[:split_idx]
    self.test_data = self.data[split_idx:]
    return self.train_data, self.test_data

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

# example usage
if __name__ == "__main__":
  file_path = "/content/drive/MyDrive/dna_data.txt"
  dataset = Dataset(file_path, ratio=0.2)
  
  full_data = dataset.get_full_data()
  print("Formatted Data (first 100 characters):")
  print(full_data[:100])
  
  train_data, test_data = dataset.train_test_split()
  print(f"\nTotal Length: {len(full_data)} characters")
  print(f"Train Data Length: {len(train_data)} characters")
  print(f"Test Data Length: {len(test_data)} characters")
  
  # example: get a batch for training
  batch_size = 16
  block_size = 128
  x, y = dataset.get_batch("train", batch_size, block_size, device="cpu")
  print(f"\nBatch shapes: x: {x.shape}, y: {y.shape}")