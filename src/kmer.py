from itertools import product
import json

class KMerPy:
  def __init__(self, kmer_size:int=4):
    self.kmer_size = kmer_size
    self.base_chars = ['A', 'T', 'G', 'C', '\n']  # upper-cased base protiens
    self.ids_to_token, self.vocab = [], {}

    # Calculate the sum of powers of 5 from 5^0 to 5^5 (i.e., 5^0 + 5^1 + 5^2 + 5^3 + 5^4 + 5^5)
    # The range(6) generates numbers from 0 to 5, and for each i, we compute 5 ** i.
    # subtracting 1 from total to adjust the size
    self.vocab_size = 0
    self.vocab_size += sum(len(self.base_chars) ** i for i in range(self.kmer_size)) - 1

  def tokenize(self, sequence):
    sequence = sequence.upper() # ensures sequence entered is upper-cased
    return [sequence[i:i+self.kmer_size] for i in range(len(sequence))]

  def detokenize(self, ids):
    return "".join(ids[i][:1] for i in range(len(ids)))

  def build_vocab(self):
    index = 0
    chars = sorted(self.base_chars)
    for k in range(1, self.kmer_size + 1):
      for combination in product(chars, repeat=k):
        token = ''.join(combination)
        self.vocab[token] = index
        self.ids_to_token.append(token)
        index += 1

  def encode(self, sequence):
    tokenized_data = self.tokenize(sequence)
    encoded_tokens = [self.vocab[kmer] for kmer in tokenized_data]
    return encoded_tokens

  def decode(self, ids):
    chars = self.ids_to_chars(ids)
    detokenized_data = self.detokenize(chars)
    return detokenized_data

  def ids_to_chars(self, ids:list[int]):
    """returns the list containing chars mapped to ids

    Args:
      ids (List[int]): list containing only output tokens from a model or just ids
    Returns:
      List: list with the respective chars
    """
    assert isinstance(ids, list) and len(ids) > 0, "ids must be a non-empty list"
    assert type(ids[0]) == int, "only accepts encoded ids"
    chars = []
    for i in ids:
      chars.append(self.ids_to_token[i])
    return chars

  def chars_to_ids(self, chars:list[str]):
    """returns the list containing ids mapped to chars

    Args:
      chars (List[str]): list containing tokenized chars for id mapping
    Returns:
      Lits: list with the respective ids
    """
    assert isinstance(chars, list) and len(chars) > 0, "ids must be a non-empty list"
    assert type(chars[0]) == str, "only accepts tokenized kmer pairs"
    ids = []
    for i in chars:
      ids.append(self.vocab[i])
    return ids
  
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
      file_path = f"{file}/verify.json"
      with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(verified, f)
    return verified

  def save(self, path:str):
    # saving the vocab model as a json file for easy saving & retrieval
    vocab_file = f"{path}/base_{self.kmer_size}k.json"
    with open(vocab_file, 'w', encoding='utf-8') as f:
      json.dump(self.vocab, f)
    print("saved the vocab!")

  def load(self, path:str):
    assert path.endswith('.json')
    with open(path, 'r') as f:
      vocab = json.load(f)
    print("loaded the vocab!")

    # re-initializing the vocabs & neccessary idexings
    self.vocab = vocab
    self.vocab_size = len(vocab)
    for token, idx in self.vocab.items():
        self.ids_to_token.append(token)