from .kmer import KMerPy
# from .auto.tokenizer import tokenizer
import os
current_directory = os.path.dirname(os.path.abspath(__file__))
print("curr: ", current_directory)
os.chdir(current_directory)

pre_model = ["dna-perchar", "enigma1", "EnBERT", "enigma2"]
pre_encoding = ["base_1k", "base_2k", "base_3k", "base_4k", "base_5k"]
pre_mode = ["kmer", "bpe", "vae"]

class DNATokenizer:
  def __init__(self, encoding:str, mode:str="kmer", model:str=None):
    if encoding not in pre_encoding:
      raise ValueError(f"`{encoding}` doesn't exist try using the existing encoding!")
    if mode not in pre_mode:
      raise ValueError(f"`{mode}` doesn't exist try using the existing modes!")
    if model is not None and model not in pre_model:
      raise ValueError(f"`{model}` model doesn't exist maybe try choosing some other model!")
    
    self.model, self.encoding, self.mode = model, encoding, mode

    # switch tokenizer based on model selection
    if model in ["dna-perchar", "enigma1"]:
      self.tokenizer = KMerPy(kmer_size=1)  # swapped the C-version of PerChar with normal KMer class with size=1
    else:
      # extract kmer size from encoding string e.g. base_4k -> 4
      try:
        kmer_size = int(encoding.split('_')[1].replace('k',''))
      except (IndexError, ValueError):
        raise ValueError("encoding format invalid, should be like 'base_4k'")
      self.tokenizer = KMerPy(kmer_size=kmer_size)
      vocab_path = os.path.join(current_directory, "vocabs", f"{encoding}.json") # attempt to load vocab from vocabs directory
      if os.path.exists(vocab_path):
        self.tokenizer.load(vocab_path)
      else:
        print("Error loading the vocabs, building vocabs!")
        self.tokenizer.build_vocab()  # build vocab if vocab file not found

  def encode(self, sequence):
    return self.tokenizer.encode(sequence)

  def decode(self, ids):
    return self.tokenizer.decode(ids)

  def tokenize(self, sequence):
    if self.mode == "vae":
      raise TypeError("Function only available for the `kmer` & `bpe` modes!")
    return self.tokenizer.tokenize(sequence)

  def one_hot_encode(self, sequence):
    if self.mode != "vae":
      raise TypeError("Function only available for the `VAE` mode!")
    return self.tokenizer.dna_to_onehot(sequence)

  @property
  def vocab(self):
    return self.tokenizer.vocab

  @property
  def vocab_size(self):
    return self.tokenizer.vocab_size

  def __str__(self):
    return f"biosaic DNATokenizer <mode={self.mode}, encoding={self.encoding}>"