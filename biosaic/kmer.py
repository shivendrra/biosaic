from itertools import product
import json, pickle
import os, io, requests, tempfile, urllib

class KMer:
  def __init__(self, kmer_size:int=4):
    self.kmer_size = kmer_size
    self.base_chars = ['A', 'T', 'G', 'C']  # upper-cased base protiens
    self.ids_to_token, self.vocab = {}, {}

    # Calculate the sum of powers of 5 from 4^0 to 4^5 (i.e., 4^0 + 4^1 + 4^2 + 4^3 + 4^4 + 4^5)
    # The range(6) generates numbers from 0 to 5, and for each i, we compute 4 ** i.
    # subtracting 1 from total to adjust the size
    self.vocab_size = len(self.base_chars) ** kmer_size

  def tokenize(self, sequence):
    sequence = sequence.upper() # ensures sequence entered is upper-cased
    if any(ch not in self.base_chars for ch in sequence):
      raise ValueError("Invalid character in DNA sequence")
    return [sequence[i:i+self.kmer_size] for i in range(len(sequence) - self.kmer_size + 1)]

  def detokenize(self, ids):
    return "".join(ids[i][0] for i in range(len(ids))) + ids[-1][1:]

  def build_vocab(self, continuous=False):
    letters, combos = sorted(self.base_chars), []
    if continuous:
      for L in range(1, self.kmer_size + 1):
        combos.extend(product(letters, repeat=L))
    else:
      combos = list(product(letters, repeat=self.kmer_size))
    self.vocab = {''.join(c): i for i, c in enumerate(combos)}
    self.ids_to_token = {v: k for k, v in self.vocab.items()}
    self.vocab_size = len(self.vocab.items())

  def encode(self, sequence):
    tokenized_data = self.tokenize(sequence)
    return [self.vocab[kmer] for kmer in tokenized_data if kmer in self.vocab]

  def decode(self, ids):
    tokens = self.ids_to_chars(ids)
    return self.detokenize(tokens)

  def ids_to_chars(self, ids: list[int]):
    """returns the list containing chars mapped to ids

    Args:
      ids (List[int]): list containing only output tokens from a model or just ids
    Returns:
      List: list with the respective chars
    """
    assert isinstance(ids, list) and len(ids) > 0, "ids must be a non-empty list"
    assert isinstance(ids[0], int), "only accepts encoded ids"
    return [self.ids_to_token[i] for i in ids]

  def chars_to_ids(self, chars: list[str]):
    """returns the list containing ids mapped to chars

    Args:
      chars (List[str]): list containing tokenized chars for id mapping
    Returns:
      Lits: list with the respective ids
    """
    assert isinstance(chars, list) and len(chars) > 0, "chars must be a non-empty list"
    assert isinstance(chars[0], str), "only accepts tokenized strings"
    return [self.vocab[i] for i in chars]

  def verify(self, ids, file=None):
    """returns a list containing true/false values for respective matching kmers
      also saves them to a file, as needed by user

    Args:
      ids (List[str]): list containing tokenized chars
      file (Optional|None): file path
    Returns:
      dictonary: dictonary containing mapped true/false pairs for verification
    """
    verified = []
    ids = self.ids_to_chars(ids) if isinstance(ids[0], int) else ids
    for i in range(len(ids) - 1):
      match = ids[i][1:] == ids[i + 1][:-1]
      verified.append({"kmer1": ids[i], "kmer2": ids[i + 1], "match": match})
    if file:
      file_path = os.path.join(file, "verify.json")
      with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(verified, f)
    return verified

  def save(self, path, as_json=False):
      os.makedirs(os.path.dirname(path), exist_ok=True)
      data = {
        "kmer_size": self.kmer_size,
        "vocab_size": self.vocab_size,
        "trained_vocab": self.vocab
      }
      if as_json:
        with open(path + ".json", "w", encoding="utf-8") as f:
          json.dump(data, f, indent=2)
      else:
        with open(path + ".model", "wb") as f:
          pickle.dump(data, f)
      print(f"DEBUGG INFO[104] [Saved] Vocabulary saved to {path + ('.json' if as_json else '.model')}")

  def load(self, model_path: str):
    def is_url(path):
      return path.startswith("http://") or path.startswith("https://")

    if is_url(model_path):
      # print(f"DEBUGG INFO[200] Fetching remote model from: {model_path}")
      with tempfile.NamedTemporaryFile(delete=False, suffix=".model" if model_path.endswith(".model") else ".json") as tmp_file:
        urllib.request.urlretrieve(model_path.replace("blob/", ""), tmp_file.name)
        model_path = tmp_file.name

    if model_path.endswith(".json"):
      with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    elif model_path.endswith(".model"):
      with open(model_path, "rb") as f:
        data = pickle.load(f)
    else:
      raise TypeError("Only supports vocab file format `.model` & `.json`")

    self.vocab = data["trained_vocab"]
    self.vocab_size = data.get("vocab_size", None)
    self.kmer_size = data.get("kmer_size", None)
    self.ids_to_token = {v: k for k, v in self.vocab.items()}
    # print(f"DEBUGG INFO[201] Vocab loaded successfully with {self.vocab_size} size")