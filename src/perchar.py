import ctypes
from .cbase import libchar, CChar

class PerChar(CChar):
  def __init__(self):
    self._core_tokenizer = libchar.init_tokenizer()
  
  def encode(self, seq: str):
    encoded_size = ctypes.c_size_t(0)
    encoded_ptr = libchar.encode_sequence(self._core_tokenizer, seq.encode("utf-8"), ctypes.byref(encoded_size))
    encoded = [encoded_ptr[i] for i in range(encoded_size.value)]
    return encoded

  def decode(self, ids: list):
    encoded_size = len(ids)
    encoded_array = (ctypes.c_int * encoded_size)(*ids)
    decoded_ptr = libchar.decode_sequence(self._core_tokenizer, encoded_array, ctypes.c_size_t(encoded_size))
    decoded = ctypes.string_at(decoded_ptr).decode("utf-8")
    return decoded

  def __del__(self):
    libchar.free_tokenizer(self._core_tokenizer)