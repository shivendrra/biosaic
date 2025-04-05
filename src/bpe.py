from itertools import product
import multiprocessing, timeit
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from array import array
import os, json, pickle

## key-value pairs of Amino Acids & DNA
AMINO_ACIDS = [
  'A','R','N','D','C','Q','E','G','H','I',
  'L','K','M','F','P','S','T','W','Y','V','-'  # 21st for gap/pad
]

VOCAB = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
ID2AA = {i: aa for aa, i in VOCAB.items()}

DNA_VOCAB = {"A": 0, "T": 1, "C": 2, "G": 3}
INDEX_TO_DNA = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

def _count_pairs_chunk(ids_chunk):
  # using Counter in C for speed
  return Counter(zip(ids_chunk, ids_chunk[1:]))

def get_stats(ids):
  # counting adjacent pairs using multiple processes + Counter
  n_procs = max(1, multiprocessing.cpu_count() - 2)
  chunk = max(1, len(ids) // n_procs)
  futures = []
  total = Counter()

  with ProcessPoolExecutor(max_workers=n_procs) as exe:
    for i in range(0, len(ids), chunk):
      # overlapping by 1 to catch pairs at boundaries
      sub = ids[i : i + chunk + 1]
      futures.append(exe.submit(_count_pairs_chunk, sub))

    for fut in as_completed(futures):
      total.update(fut.result())
  return total

def _kmer_chunk(seq, k):
  return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def get_kmers(seq, kmer_size=4):
  # parallel k-mer tokenization using processes
  seq = seq.upper()
  n_procs = max(1, multiprocessing.cpu_count() - 2)
  chunk = max(1, len(seq) // n_procs)
  futures = []
  kmers = []

  with ProcessPoolExecutor(max_workers=n_procs) as exe:
    for i in range(0, len(seq), chunk):
      # extending by kmer_size-1 to avoid cutting kmers
      sub = seq[i : i + chunk + kmer_size - 1]
      futures.append(exe.submit(_kmer_chunk, sub, kmer_size))

    for fut in as_completed(futures):
      kmers.extend(fut.result())
  return kmers

def merge(ids, pair, idx):
  # Single-pass merge;
  # consider batching merges for more speed
  out = array('I')
  a, b, i, n = pair, 0, len(ids)
  while i < n:
    if i+1 < n and ids[i] == a and ids[i+1] == b:
      out.append(idx)
      i += 2
    else:
      out.append(ids[i])
      i += 1
  return out

def batch_merge(ids, merge_map):
  # Single pass: if (ids[i], ids[i+1]) in merge_map,
  # emit merge_map[(a,b)] and skip 2; else emit ids[i] and skip 1.
  out = array('I')
  i, n = 0, len(ids)
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
  def __init__(self, kmer_size):
    self.init_vocab_size = len(DNA_VOCAB)
    self.kmer_size = kmer_size
    self.base_vocab = {}
    self.merged_vocab = {}  # token_id -> actual token (string)

  def initialize_vocab(self, continuous=False):
    letters = sorted(DNA_VOCAB.keys())
    combos = []
    if continuous:
      for L in range(1, self.kmer_size+1):
        combos += product(letters, repeat=L)
    else:
      combos = product(letters, repeat=self.kmer_size)

    self.base_vocab = {''.join(c): i for i, c in enumerate(combos)}
    self.init_vocab_size = len(self.base_vocab)
    self.merged_vocab = {v: k for k, v in self.base_vocab.items()}

  def _base_encode(self, tokens):
    arr = array('I')
    for t in tokens:
      arr.append(self.base_vocab.get(t, 0))
    return arr

  def train(self, seq, vocab_size, early_stop=10, verbose=True):
    num_merges = vocab_size - self.init_vocab_size
    tokens = get_kmers(seq, self.kmer_size)
    ids = self._base_encode(tokens)
    merge_count = 0

    while merge_count < num_merges:
      # )1 computing stats
      t0 = timeit.default_timer()
      stats = get_stats(ids)
      t1 = timeit.default_timer()
      if verbose:
        dt = t1 - t0
        print(f"[Stats] {len(stats)} pairs in {dt*1000:.1f}ms")

      # 2) picking up top pairs
      top = stats.most_common(early_stop)
      if not top:
        if verbose:
          print("No more pairs to merge.")
        break

      # build merge_map for batch
      merge_map = {}
      for pair, freq in top:
        if merge_count >= num_merges:
          break
        new_id = self.init_vocab_size + merge_count
        token_str = f"{self.merged_vocab.get(pair[0], pair[0])}_{self.merged_vocab.get(pair[1], pair[1])}"
        self.merged_vocab[new_id] = token_str
        merge_map[pair] = new_id
        merge_count += 1

      # 3) applying all merges in one pass
      t0 = timeit.default_timer()
      ids = batch_merge(ids, merge_map)
      t1 = timeit.default_timer()
      dt = t1 - t0
      unit = "ms" if dt < 1 else "s"
      val = dt*1000 if dt < 1 else dt
      if verbose:
        print(f"[Batch merge] applied {len(merge_map)} merges in {val:.1f}{unit}")

      # 4) logging each merge detail
      for idx, (pair, freq) in enumerate(top, start=merge_count-len(merge_map)+1):
        mid = merge_map[pair]
        if verbose:
          print(f"  → Merge {idx}/{num_merges}: {pair} → id {mid}, freq {freq}")
  
  def save(self, path, as_json=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
      "kmer_size": self.kmer_size,
      "init_vocab_size": self.init_vocab_size,
      "base_vocab": self.base_vocab,
      "merged_vocab": self.merged_vocab
    }
    if as_json:
      with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    else:
      with open(path + ".model", "wb") as f:
        pickle.dump(data, f)
    print(f"[Saved] Vocabulary saved to {path + ('.json' if as_json else '.model')}")

  def load(self, path):
    if path.endswith(".json"):
      with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    else:
      with open(path, "rb") as f:
        data = pickle.load(f)

    self.kmer_size = data["kmer_size"]
    self.init_vocab_size = data["init_vocab_size"]
    self.base_vocab = data["base_vocab"]
    self.merged_vocab = data["merged_vocab"]
    print(f"[Loaded] Vocabulary loaded from {path}")