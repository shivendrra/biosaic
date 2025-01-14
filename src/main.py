import ctypes, os
from ctypes import c_int, c_char_p, byref, POINTER
from cbase import libkmer, CKMer

current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

class Biosaic(CKMer):
  def __init__(self, kmer:int= 4):
    assert isinstance(kmer, int), "KMer value must be a positive integer"
    self._core_tokenizer = libkmer.create_tokenizer(c_int(kmer))
    libkmer.build_vocab(self._core_tokenizer)

  def _shred(self, seq: str):
    if seq is not None:
      n_kmers = c_int(0)
      kmers_ptr = POINTER(c_char_p)()
      libkmer.tokenize_sequence(self._core_tokenizer, seq.encode("utf-8"), byref(kmers_ptr), byref(n_kmers))
      kmers = [kmers_ptr[i].decode("utf-8") for i in range(n_kmers.value)]
      return kmers
    else:
      raise ValueError("Sequence can't be NULL or Empty! Must provide some value")
  
  def _build_vocab(self):
    libkmer.build_vocab(self._core_tokenizer)

  def encode(self, seq: str):
    encoded_size = c_int(0)
    encoded_ptr = libkmer.encode_sequence(self._core_tokenizer, seq.encode("utf-8"), byref(encoded_size))
    encoded = [encoded_ptr[i] for i in range(encoded_size.value)]
    return encoded

  def decode(self, ids: int):
    encoded_size = len(ids)
    encoded_array = (c_int * encoded_size)(*ids)
    decoded_ptr = libkmer.decode_sequence(self._core_tokenizer, encoded_array, c_int(encoded_size))
    if not decoded_ptr:
      raise RuntimeError("decode_sequence returned a null pointer.")
    decoded = ctypes.string_at(decoded_ptr).decode("utf-8")
    return decoded

  def save(self, path: str):
    libkmer.save(self._core_tokenizer, path.encode("utf=8"))

  def __del__(self):
    libkmer.free_tokenizer(self._core_tokenizer)

if __name__ == "__main__":
    tokenizer = Biosaic(kmer=4)
    sequence = "BAACATGTCCTGCATGGCATTAMGTTTGTTGGGGCAGTGCCCGPGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAAE"
    print("shreded sequence: ", tokenizer._shred(sequence))

    encoded = tokenizer.encode(sequence)
    print("Encoded sequence:", encoded)
    
    decoded = tokenizer.decode(encoded)
    print("Decoded sequence:", decoded)
    
    tokenizer.save("./vocab.model")
    # tokenizer.free()
    del tokenizer