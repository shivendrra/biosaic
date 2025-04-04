import ctypes
from ctypes import c_int, c_char_p, byref, POINTER
from .cbase import libkmer, CKMer

class KMer(CKMer):
  def __init__(self, kmer:int= 4):
    assert isinstance(kmer, int), "KMer value must be a positive integer"
    self._core_tokenizer = libkmer.create_tokenizer(c_int(kmer))
    if not self._core_tokenizer:
      raise MemoryError("Failed to initialize KMer tokenizer")
    libkmer.build_vocab(self._core_tokenizer)
    self.tokenized_size = 0

  def _shred(self, seq: str):
    # basic code that splits the string into chunks according to spcified KMer size
    # leaves the special tokens as a single string, doesn't include them in the chunks
    #     "BACTAGAAAMCTTE" -> ["B", "ACTA", "GAAA", "M", "CTT", "E"]
    if seq is not None:
      kmers_ptr = POINTER(c_char_p)()
      self.tokenized_size = libkmer.tokenize_sequence(self._core_tokenizer, seq.encode("utf-8"), byref(kmers_ptr))
      kmers = [kmers_ptr[i].decode("utf-8") for i in range(self.tokenized_size)]
      return kmers
    else:
      raise ValueError("Sequence can't be NULL or Empty! Must provide some value")
  
  def _build_vocab(self):
    # uses basic permutative logic to builds the vocab, as the key -> value pairs
    #     "AACT" -> 442
    #     "M" -> 3
    #     "ACTT" -> 758
    # fully deterministic & key-value pairs never change since follows basic 
    # string spliting & permutative logic, kinda slow but gets the job done
    libkmer.build_vocab(self._core_tokenizer)

  def encode(self, seq: str):
    # simple encoding functions, simply looks up for the provided strings or "keys"
    # & returns it's value or index from the vocab lookup table
    encoded_size = c_int(0)
    encoded_ptr = libkmer.encode_sequence(self._core_tokenizer, seq.encode("utf-8"), byref(encoded_size))
    encoded = [encoded_ptr[i] for i in range(encoded_size.value)]
    return encoded

  def decode(self, ids: int):
    # just works reverse as of the `encode()`
    encoded_size = len(ids)
    encoded_array = (c_int * encoded_size)(*ids)
    decoded_ptr = libkmer.decode_sequence(self._core_tokenizer, encoded_array, c_int(encoded_size))
    decoded = ctypes.string_at(decoded_ptr).decode("utf-8")
    return decoded

  def save(self, path: str):
    # doesn't really do nothing, just there for debugging, problems related to vocab
    # building or looking up the values manually
    libkmer.save(self._core_tokenizer, path.encode("utf=8"))

  def __del__(self):
    libkmer.free_tokenizer(self._core_tokenizer)

pre_model = ["dna-perchar", "enigma1", "EnBERT", "enigma2"]
pre_encoding = ["base_2k", "base_3k", "base_4k", "base_5k"]

class DNATokenizer:
  def __init__(self, encoding:str, model:str=None):
    if encoding not in pre_encoding:
      raise ValueError(f"`{encoding}` doesn't exist. Try using the existing encoding!")
    if model not in pre_model:
      raise ValueError(f"`{model}` model doesn't exist. maybe try chosing some other model!")
    self.model = model
    self.encoding = encoding
    self.tokenizer = ""
    self.sp_start_seq = "<b>"
    self.sp_enc_seq = "</b>"
    self.sp_pad_seq = "<p>"
    pass

  def encode(self, sequence):
    pass

  def decode(self, ids):
    pass

  def tokenize(self, sequence):
    pass

  @property
  def vocab(self):
    pass

  @property
  def vocab_size(self):
    pass

  def __str__(self):
    return f"Biosaic DNATokenizer <model={self.model}, encoding={self.encoding}>"