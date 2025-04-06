from itertools import product
import multiprocessing, timeit
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from array import array
import os, json, pickle

AMINO_ACIDS = [
  'A','R','N','D','C','Q','E','G','H','I',
  'L','K','M','F','P','S','T','W','Y','V','-'
]

VOCAB = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
ID2AA = {i: aa for aa, i in VOCAB.items()}
DNA_VOCAB = {"A": 0, "T": 1, "C": 2, "G": 3}
INDEX_TO_DNA = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

def _count_pairs_chunk(ids_chunk):
  return Counter(zip(ids_chunk, ids_chunk[1:]))

def get_stats(ids):
  n_procs = max(1, multiprocessing.cpu_count() - 2)
  chunk = max(1, len(ids) // n_procs)
  futures, total = [], Counter()
  with ProcessPoolExecutor(max_workers=n_procs) as exe:
    for i in range(0, len(ids), chunk):
      sub = ids[i : i + chunk + 1]
      futures.append(exe.submit(_count_pairs_chunk, sub))
    for fut in as_completed(futures):
      total.update(fut.result())
  return total

def _kmer_chunk(start, end, seq, k):
  return [seq[i:i+k] for i in range(start, end - k + 1)]

def get_kmers(seq, kmer_size=4):
  seq = seq.upper()
  n_procs = max(1, multiprocessing.cpu_count() - 2)
  chunk_size = max(1, len(seq) // n_procs)
  futures, kmers = [], []
  with ProcessPoolExecutor(max_workers=n_procs) as exe:
    for i in range(0, len(seq) - kmer_size + 1, chunk_size):
      start = i
      end = min(i + chunk_size + kmer_size - 1, len(seq))
      futures.append(exe.submit(_kmer_chunk, start, end, seq, kmer_size))
    for fut in as_completed(futures):
      kmers.extend(fut.result())

  # adding leftover token if any (1 to k-1 characters at the end)
  rem = len(seq) % kmer_size
  if rem:
    leftover = seq[-rem:]
    kmers.append(leftover)
  return kmers

def merge(ids, pair, idx):
  out = array('I')
  a, b, i, n = pair[0], pair[1], 0, len(ids)
  while i < n:
    if i+1 < n and ids[i] == a and ids[i+1] == b:
      out.append(idx)
      i += 2
    else:
      out.append(ids[i])
      i += 1
  return out

def batch_merge(ids, merge_map):
  out, i, n = array('I'), 0, len(ids)
  while i < n:
    if i + 1 < n:
      pair = (ids[i], ids[i+1])
      new_id = merge_map.get(pair)
      if new_id is not None:
        out.append(new_id)
        i += 2
        continue
    out.append(ids[i])
    i += 1
  return out

class bpe_trainer:
  def __init__(self, kmer_size, continuous=False):
    self.init_vocab_size = len(DNA_VOCAB)
    self.kmer_size = kmer_size
    self.base_vocab = {}
    self.vocab = {}
    self.initialize_vocab(continuous)

  def __str__(self):
    return f"\t/Biosaic BPE trainer v0.1/\t"

  def initialize_vocab(self, continuous=False):
    print("DEBUGG INFO[101] Intializing Vocabs")
    letters = sorted(DNA_VOCAB.keys())
    combos = []
    if continuous:
      for L in range(1, self.kmer_size + 1):
        combos.extend(product(letters, repeat=L))
    else:
      combos = list(product(letters, repeat=self.kmer_size))
    self.base_vocab = {''.join(c): i for i, c in enumerate(combos)}
    self.init_vocab_size = len(self.base_vocab)
    self.vocab = {v: k for k, v in self.base_vocab.items()}
    print(f"DEBUGG INFO[201] Vocabs Initialized successfully for K_Mer size of {self.kmer_size}")

  def _base_encode(self, tokens):
    arr = array('I')
    for t in tokens:
      arr.append(self.base_vocab.get(t, 0))
    return arr

  def train(self, seq, vocab_size, early_stop=10):
    print(f"DEBUGG INFO[106] Starting the training with target_vocab: {vocab_size}, early_stopping: {early_stop}")
    num_merges = vocab_size - self.init_vocab_size
    tokens = get_kmers(seq, self.kmer_size)
    print(f"DEBUGG INFO[107] Converted sequence into K_mers of length: {self.kmer_size}, total tokens: {len(tokens)}")
    ids = self._base_encode(tokens)
    merge_count, invalidated_pairs = 0, set()

    while merge_count < num_merges:
      t0 = timeit.default_timer()
      stats = get_stats(ids)
      for pair in invalidated_pairs:
        if pair in stats:
          del stats[pair]
      t1 = timeit.default_timer()
      print(f"DEBUGG INFO[102] [Stats] {len(stats)} pairs in {(t1 - t0)*1000:.1f}ms")

      top = [p for p in stats.most_common(early_stop) if p[0] not in invalidated_pairs]
      if not top:
        print("DEBUGG WARN[301] No more pairs to merge.")
        break

      merge_map = {}
      for pair, freq in top:
        if merge_count >= num_merges:
          break
        new_id = self.init_vocab_size + merge_count
        token_str = f"{self.vocab.get(pair[0], pair[0])}{self.vocab.get(pair[1], pair[1])}"
        self.vocab[new_id] = token_str
        merge_map[pair] = new_id
        merge_count += 1

      t0 = timeit.default_timer()
      try:
        ids = batch_merge(ids, merge_map)
      except KeyError as e:
        print(f"DEBUGG ERR[999] Merge failed for pair: {e}. Adding to invalidated cache.")
        invalidated_pairs.add(e)
        continue
      t1 = timeit.default_timer()
      val = (t1 - t0) * 1000 if (t1 - t0) < 1 else (t1 - t0)
      unit = "ms" if (t1 - t0) < 1 else "s"
      print(f"DEBUGG INFO[102] [Batch merge] applied {len(merge_map)} merges in {val:.1f}{unit}")

      for idx, (pair, freq) in enumerate(top, start=merge_count - len(merge_map) + 1):
        if pair not in merge_map:
          print(f"DEBUGG WARN[302] Skipping invalid pair {pair}")
          continue
        mid = merge_map[pair]
        print(f"DEBUGG INFO[103] Merging {idx}/{num_merges}: ({pair} -> id {mid}), freq: {freq}")
    self.vocab = {v: k for k, v in self.vocab.items()}

  def save(self, path, as_json=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
      "kmer_size": self.kmer_size,
      "init_vocab_size": self.init_vocab_size,
      "merged_vocab": self.vocab
    }
    if as_json:
      with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    else:
      with open(path + ".model", "wb") as f:
        pickle.dump(data, f)
    print(f"DEBUGG INFO[104] [Saved] Vocabulary saved to {path + ('.json' if as_json else '.model')}")

class BPE:
  def __init__(self, encodings:str=None):
    self.vocab = {}
    self.inv_vocab = {}
    self.kmer_size = None
    if encodings:
      self.load(encodings)

  def __str__(self):
    return f"\t/Biosaic BPE tokenizer v0.1/\t"

  def load(self, model_path: str):
    if model_path.endswith(".json"):
      with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    elif model_path.endswith(".model"):
      with open(model_path, "rb") as f:
        data = pickle.load(f)
    else:
      raise TypeError("Only supports vocab file format `.model` & `.json`")

    self.vocab = data["merged_vocab"]
    self.kmer_size = data.get("kmer_size", None)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    print(f"DEBUGG INFO[201] Vocab loaded successfully with {len(self.vocab)} tokens")

  def encode(self, seq: list[str]) -> array:
    tokens = get_kmers(seq, self.kmer_size)
    ids = array('I')
    for tok in tokens:
      if tok not in self.vocab:
        raise KeyError(f"Token '{tok}' not in vocabulary")
      ids.append(self.vocab[tok])
    return list(ids)

  def decode(self, ids: list[int] | array) -> str:
    tokens = [self.inv_vocab[i] for i in ids]
    if not tokens:
      return ""
    result = tokens[0]
    k = self.kmer_size
    for i in range(1, len(tokens)):
      prev = tokens[i - 1]
      curr = tokens[i]

      # find max possible overlap between previous and current
      overlap = 0
      for j in range(1, k):
        if prev[-j:] == curr[:j]:
          overlap = j
      result += curr[overlap:]
    return result

  def token_to_id(self, token: str) -> int:
    return self.vocab[token]

  def id_to_token(self, idx: int) -> str:
    return self.inv_vocab[idx]