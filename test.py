import os

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

from src import KMer

CHUNK_SIZE = 1000000

with open("file.txt", "r", encoding="utf-8") as f:
  test_data = f.read()

tokenizer = KMer(kmer=4)
encoded_chunks = []
decoded_chunks = []

for i in range(0, len(test_data), CHUNK_SIZE):
  chunk = test_data[i:i + CHUNK_SIZE]
  print(f"Processing chunk {i // CHUNK_SIZE + 1}/{-(-len(test_data) // CHUNK_SIZE)}...")
  shredded_chunk = tokenizer._shred(chunk)
  encoded_chunk = tokenizer.encode(chunk)
  decoded_chunk = tokenizer.decode(encoded_chunk)
  encoded_chunks.extend(encoded_chunk)
  decoded_chunks.append(decoded_chunk)

decoded_result = ''.join(decoded_chunks)
print(decoded_result)
print("Decoded matches original:", decoded_result == test_data)
del tokenizer