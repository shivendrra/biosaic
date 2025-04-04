from src import KMer, PerChar

# token = KMer(kmer_size=4)
token = PerChar()

with open("data/file.txt", "r", encoding="utf-8") as f:
  sequence = f.read()
  dataset = "".join(line.strip() for line in sequence if line.strip())
  dataset = dataset.upper()
  print("sequence length: ", len(dataset))
  del sequence
  f.close()

# sequence = "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTG"
# token.build_vocab()
# token.save("./model")
# token.load("./model/base_4k.json")
encoded = token.encode(dataset)
decoded = token.decode(encoded)

print(encoded[:100])
print(decoded[:400])
print(decoded == dataset)

# tokenized = token.tokenize(sequence)
# ids = token.chars_to_ids(tokenized)
# chars = token.ids_to_chars(ids)
# print(ids)
# print(chars)
# print(token.verify(tokenized, 'model'))