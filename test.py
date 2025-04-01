from src import DNATokenizer, pre_model, pre_encoding

print("available models: ", pre_model)
print("available encodings: ", pre_encoding)

token = DNATokenizer(encoding=pre_encoding[2])

# with open("data/file.txt", "r", encoding="utf-8") as f:
#   sequence = f.read()
#   sequence = "".join(line.strip() for line in sequence if line.strip())
#   sequence = sequence.upper()
#   print("sequence length: ", len(sequence))
#   f.close()

sequence = "AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGCTTCTGAACTG"
encoded = token.encode(sequence)
decoded = token.decode(encoded)
tokenized = token.tokenize(sequence)

print(tokenized)
print(encoded[:100])
print(decoded[:400])
print(decoded == sequence)

print(token.vocab_size)