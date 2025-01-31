from src.kmer import KMerPy

token = KMerPy(kmer_size=4)

# with open("file.txt", "r", encoding="utf-8") as f:
#   sequence = f.read()
#   print("sequence length: ", len(sequence))
#   f.close()

sequence = "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTG"
token.build_vocab()
token.save("./model")
token.load("./model/base_4k.json")
encoded = token.encode(sequence)
decoded = token.decode(encoded)

print(encoded)
print(decoded)
print(decoded == sequence)