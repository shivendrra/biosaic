import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

from src import  PerChar

# tokenizer = KMer(kmer=4)
tokenizer = PerChar()
sequence = "BAACATGTCCTGCATGGCATTAMGTTTGTTGGGGCAGTGCCCGPGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAAE"
# print("shreded sequence: ", tokenizer._shred(sequence))
encoded = tokenizer.encode(sequence)
decoded = tokenizer.decode(encoded)

print("Encoded sequence:", encoded)
print("Decoded sequence:", decoded)
print("decoded string matches the original string:", decoded == sequence)

# tokenizer.save("./vocab")
del tokenizer