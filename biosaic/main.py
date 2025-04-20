from .kmer import KMer
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory)

pre_model = ["dna-perchar", "enigma1", "EnBERT", "enigma2"]
pre_encoding = ["base_1k", "base_2k", "base_3k", "base_4k", "base_5k"]
pre_mode = ["kmer", "bpe", "vae"]

main_base_url = "https://github.com/shivendrra/biosaic/blob/main/model/"  # fetches from main branch
dev_base_url = "https://raw.githubusercontent.com/shivendrra/biosaic/dev/model/"  # fetches from dev branch
hugginface_url = "https://huggingface.co/shivendrra/BiosaicTokenizer/resolve/main/kmers/"  # fetches from huggingface librrary

class tokenizer:
  def __init__(self, encoding:str):
    if encoding not in pre_encoding:
      raise ValueError(f"`{encoding}` doesn't exist try using the existing encoding!")
    self.encoding = encoding
    get_encoding_path = dev_base_url + encoding + ".model"
    self.kmer_size = int(encoding.split('_')[1].replace('k',''))
    self._tokenizer = KMer(self.kmer_size)
    self._tokenizer.load(model_path=get_encoding_path)

  def encode(self, sequence):
    return self._tokenizer.encode(sequence)

  def decode(self, ids):
    return self._tokenizer.decode(ids)

  def tokenize(self, sequence):
    return self._tokenizer.tokenize(sequence)

  def detokenize(self, ids):
    return self._tokenizer.detokenize(ids)

  @property
  def vocab(self):
    return self._tokenizer.vocab

  @property
  def vocab_size(self):
    return self._tokenizer.vocab_size

  def __str__(self):
    return f"biosaic.tokenizer <kmer_size={self.kmer_size}, encoding={self.encoding}>"