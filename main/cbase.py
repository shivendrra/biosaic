import os, ctypes

libkmer_path = os.path.join(os.path.dirname(__file__), "./../build/libkmerkmer.so")
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
libkmer.tokenizer_sequence.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p, ctypes.c_char, ctypes.c_int]
libkmer.tokenizer_sequence.restype = None
libkmer.build_vocab.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p, ctypes.c_int]
libkmer.build_vocab.restype = None
libkmer.encode_sequence.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p, ctypes.c_int]
libkmer.encode_sequence.restype = ctypes.c_int
libkmer.decode_sequence.argtypes = [ctypes.POINTER(CKMer), ctypes.c_int, ctypes.c_int]
libkmer.decode_sequence.restype = ctypes.c_char_p
libkmer.save.argtypes = [ctypes.POINTER(CKMer), ctypes.c_char_p]
libkmer.save.restype = None
libkmer.free_tokenizer.argtypes = [ctypes.POINTER(CKMer)]
libkmer.free_tokenizer.restype = None