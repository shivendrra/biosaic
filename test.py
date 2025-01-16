from .src import KMer

tokenizer = KMer(kmer=4)
sequence = "BAACATGTCCTGCATGGCATTAMGTTTGTTGGGGCAGTGCCCGPGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAAE"
print("shreded sequence: ", tokenizer._shred(sequence))
encoded = tokenizer.encode(sequence)
decoded = tokenizer.decode(encoded)

print("Encoded sequence:", encoded)
print("Decoded sequence:", decoded)
print("decoded string matches the original string:", decoded == sequence)

tokenizer.save("./vocab")
del tokenizer