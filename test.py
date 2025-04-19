import biosaic
from biosaic import tokenizer

token = tokenizer(encoding=biosaic.get_encodings[2])
print(token.vocab_size)

sequence = "TCTTACATAGAAAGGAGCGGTATTTGGTATGAATTTATTTGCAACTGACTG"
encoded = token.encode(sequence)
decoded = token.decode(encoded)
tokenized = token.tokenize(sequence)

print(tokenized)
print(encoded[:100])
print(decoded[:300])
print(decoded == sequence)