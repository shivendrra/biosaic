import os, ctypes
from ctypes import c_int, c_char_p

libkmer_path = os.path.join(os.path.dirname(__file__), "build/libkmer.so")
libkmer = ctypes.CDLL(libkmer_path)

MAX_TOKEN_SIZE = 10
MAX_VOCAB_SIZE = 10000
MAX_BASE_CHARS = 6
SPECIAL_TOKEN_COUNT = 6

# defining KMer structure (partial, as we're using it via functions)
class CKMer(ctypes.Structure):
  _fields_ = [
    ("chars", ctypes.c_char_p * MAX_BASE_CHARS),
    ("special_tokens", ctypes.c_char_p * SPECIAL_TOKEN_COUNT),
    ("kmers", ctypes.c_int),
    ("vocab_size", ctypes.c_int),
    ("id_to_token", ctypes.c_char_p),
    ("token_to_id", ctypes.c_int)
  ]

# function prototypes
libkmer.create_tokenizer.argtypes = [ctypes.c_int]
libkmer.create_tokenizer.restype = ctypes.POINTER(CKMer)
# libkmer.tokenize_sequence.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)), ctypes.POINTER(ctypes.c_int)]
libkmer.tokenize_sequence.argtypes = [
  ctypes.POINTER(CKMer),  # Pointer to CKMer structure
  ctypes.c_char_p,        # Input sequence (C string)
  ctypes.POINTER(ctypes.POINTER(c_char_p)),  # Pointer to an array of C strings
  ctypes.POINTER(c_int)   # Pointer to integer for k-mer count
]
libkmer.tokenize_sequence.restype = None
libkmer.build_vocab.argtypes = [ctypes.POINTER(CKMer)]
libkmer.build_vocab.restype = None
libkmer.encode_sequence.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
libkmer.encode_sequence.restype = ctypes.POINTER(ctypes.c_int)
libkmer.decode_sequence.argtypes = [ctypes.POINTER(CKMer), ctypes.POINTER(c_int), c_int]
libkmer.decode_sequence.restype = c_char_p
libkmer.save.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p]
libkmer.save.restype = None
libkmer.free_tokenizer.argtypes = [ctypes.POINTER(CKMer)]
libkmer.free_tokenizer.restype = None

# character level tokenizer starts from here
# interface to ``perchar.c``

libchar_path = os.path.join(os.path.dirname(__file__), "build/libchar.so")
libchar = ctypes.CDLL(libchar_path)

MAX_CHARS = 256
MAX_STRINGS_SIZE = 1000

class CChar(ctypes.Structure):
  _fields_ = [
    ("chars", ctypes.c_char_p * MAX_CHARS),
    ("vocab_size", ctypes.c_size_t),
    ("str_to_idx", ctypes.c_int * MAX_CHARS),
    ("idx_to_str", ctypes.c_char_p * MAX_CHARS)
  ]

libchar.init_tokenizer.argtypes = None
libchar.init_tokenizer.restype = ctypes.POINTER(CChar)
libchar.encode_sequence.argtypes = [ctypes.POINTER(CChar), ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
libchar.encode_sequence.restype = ctypes.POINTER(c_int)
libchar.decode_sequence.argtypes = [ctypes.POINTER(CChar), ctypes.POINTER(c_int), ctypes.c_size_t]
libchar.decode_sequence.restype = c_char_p
libchar.free_tokenizer.argtypes = [ctypes.POINTER(CChar)]
libchar.free_tokenizer.restype = None