import os
from biosaic import bpe_trainer, split_file, BPE

def main():
  current_directory = os.path.dirname(os.path.abspath(__file__))
  os.chdir(current_directory)
  # with open("data/file2.txt", "r", encoding="utf-8") as f:
  #   dataset = f.readlines()
  #   dataset = "".join(line.strip() for line in dataset if line.strip())
  #   dataset = dataset.upper()
  #   print(f"Loaded training data. Length: {len(dataset)}")

  # trainer = bpe_trainer(kmer_size=4)
  # trainer.train(dataset, vocab_size=500, early_stop=10)
  # trainer.save("dna_vocab", as_json=True)  # saves as json
  # trainer.save("dna_vocab")  # saves as binary
  
  sequence = "TCTTACATAGAAAGGAGCGGTATTTGGTATGAATTTATTTGCAACTGACTGCTTGGAAGTTGGCGTACATCTTTCCACGGAAACTATGAAAATACTGGTCAGCCTCTCAGTCATTTCATAAAATCTTGATTTTGTATTACAACAAATTAGGATATTTTCAGTAGAACTGATTGTAAGGCCAGACTGTTGGAATGTAATTCCTTCCCAAACATCTCTCAGGGGCACTTTCCTGAACGGCTGCTGACAGCAGCATTTGAGGACGGTGGGGCGGAGGACATCCTGGGGGGCCTGGCTTCTTGGGAACTGGAGGCTTTGGCCCTTGTCCCACCCCTGCTCCCCTGAGGAGGGAGGCGTGGGGCCCTGGGCTGGCTGCAAGACGTGGAGTGACTGTGGGTCCCCGTGGCCCCTGACATGCTCCCAGGGAACCCAAGAAAAGACTGAGACCCTGTGGTGCCTCCCGCTTTCCATCCGCATTCCATGGCAGGTGAGTCTGATTATTCGAAGGAGGCTGGAGTGTGGGCGGAGGGCAGCGCCAGGTTTCCCAATCAGATTTGCTCAGGGTCCCTCCAGCAGTCCATGCCGCAGAGGCTGTCCCTTGGGGGCCCACGCATCCTAGCCACGGCCTCCTCACGTCCATGCGGGGATTTGCGCCCTGGAAGGAGCCGCCCGGCTGCCTCTCGCCAACATGCAGCACTTCCCTTCCTTTCCATGGAGCACGGTTCCTGTCCCGGGGGTCCATATTGGCCACTGTGGGAGAGAGTCGGGCAGCTGAATTCCCGCAGGTGGGAATGCCAGGGCCCGAGGATGTTGCCCCTGTCCTGAAGGCTGTCGCCCGATCGCTCTATCCAAGGCTGCCCTGGGGCAGCGTCACCTGGGGGTCCTGCGGGGGCTTCTCAGCACAGCATCCAGCACTGCCACCTAGTGTGTTCCCGTCACGTCTCCTCCCCCCGCCTGCACCAGGCACCAGAGACCCGGATGCCAAGGCCTGTCAGCTTCCTCAATGGGAAACTTTTCTTCAGTGAACAAAGCTCTGTTTTATA"
  token = BPE()
  token.load(model_path="model/dna_1k.model")

  encoded = token.encode(sequence)
  decoded = token.decode(encoded)
  print(encoded, '\n', len(encoded))
  print(decoded, '\n', len(decoded))
  print(decoded == sequence)

if __name__ == "__main__":
  from multiprocessing import freeze_support
  freeze_support()  # optional, but safe on Windows
  # split_file(input_path="data/chunk.txt", output_dir="data/", num_files=3)
  main()