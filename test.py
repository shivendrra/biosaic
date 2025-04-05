import os
from src import bpe_trainer, split_file

def main():
  current_directory = os.path.dirname(os.path.abspath(__file__))
  os.chdir(current_directory)
  with open("data/chunk_01.txt", "r", encoding="utf-8") as f:
    dataset = f.readlines()
    dataset = "".join(line.strip() for line in dataset if line.strip())
    dataset = dataset.upper()
    print(f"Loaded training data. Length: {len(dataset)}")

  trainer = bpe_trainer(kmer_size=4)
  trainer.initialize_vocab()
  trainer.train(dataset, vocab_size=1276, early_stop=25)
  # trainer.save("model/dna_vocab", as_json=True)  # saves as json
  trainer.save("model/dna_vocab", )  # saves as binary

if __name__ == "__main__":
  from multiprocessing import freeze_support
  freeze_support()  # optional, but safe on Windows
  # split_file(input_path="data/chunk.txt", output_dir="data/", num_files=3)
  main()