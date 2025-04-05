with open("data/file.txt", "r", encoding="utf-8") as f:
  sequence = f.read()
  sequence = "".join(line.strip() for line in sequence if line.strip())
  sequence = sequence.upper()
  print("sequence length: ", len(sequence))
  f.close()

from src import DNATokenizer

token = DNATokenizer(encoding="base_5k")

# sequence = "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTG"
# encoded = token.encode(sequence)
# decoded = token.decode(encoded)
tokenized = token.tokenize(sequence)

# print(encoded[:100])
# print(decoded[:400])
# print(decoded == sequence)

# print(token.vocab_size)

import matplotlib.pyplot as plt
from collections import Counter

def plot_frequency(items):
  """
  Computes the frequency of each item in the provided list and displays
  a bar chart using matplotlib.

  Args:
    items (list): List of items for which to compute frequency.
  """
  freq_counter = Counter(items)
  # sort items by frequency in descending order
  sorted_items = sorted(freq_counter.items(), key=lambda x: x[1], reverse=True)
  labels, frequencies = zip(*sorted_items) if sorted_items else ([], [])
  
  plt.figure(figsize=(20, 12))
  plt.bar(labels, frequencies, color='skyblue')
  plt.xlabel('Items')
  plt.ylabel('Frequency')
  plt.title('Frequency of Items')
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.show()

plot_frequency(tokenized)