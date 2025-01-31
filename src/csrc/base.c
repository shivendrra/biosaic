#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "inc/tqdm.h"
#include "base.h"

void init_tokenizer(BaseTokenizer* tokenizer, int kmer_size) {
  if (!tokenizer) {
    fprintf(stderr, "Tokenizer instance is NULL!\n");
    exit(EXIT_FAILURE);
  }
  tokenizer->kmer_size = kmer_size;
  tokenizer->vocab_size = 0;
  // vocab_size is basically ``summation from i=0 to n=chars_size len(self->base_chars)^kmer_size``, since we're trying to create each
  // possible token -> idx pair till the declared KMer size
  // so if kmer = 4:
  //        vocab_size = 5 + 25 + 125 + 625 = 780
  int vocab_size = 0;
  for (int i = 0; i < kmer_size; i++) {
    vocab_size += pow(strlen(BASE_CHARS), i);
  }
  tokenizer->vocab_size += strlen(SPECIAL_CHARS);
  tokenizer->entries = malloc(vocab_size * sizeof(KmerEntry));
}

void build_vocab(BaseTokenizer* tokenizer) {
  if (!tokenizer || !tokenizer->entries) {
    fprintf(stderr, "Tokenizer is not initialized properly!\n");
    exit(EXIT_FAILURE);
  }

  int index = 0;

  for (int i = 0; i < strlen(SPECIAL_CHARS); i++) {
    tokenizer->entries[index].idx = index;
    tokenizer->entries[index].val = (char*)malloc(2); // allocating space for token + '\0'
    if (!tokenizer->entries[index].val) {
      fprintf(stderr, "Memory allocation for special token failed at index %d\n", index);
      exit(EXIT_FAILURE);
    }
    tokenizer->entries[index].val[0] = SPECIAL_CHARS[i];
    tokenizer->entries[index].val[1] = '\0';
    index++;
  }
  tqdm bar; // initialized tqdm bar
  init_tqdm(&bar, "Building the vocab: ", false, "KMers", true, tokenizer->vocab_size - strlen(SPECIAL_CHARS) , 1);

  for (int k = 1; k <= tokenizer->kmer_size; k++) {
    int* indices = (int*)malloc(k * sizeof(int));
    char* combination = (char*)malloc((k + 1) * sizeof(char));
    if (!indices || !combination) {
      fprintf(stderr, "Memory allocation failed for k-mer generation!\n");
      exit(EXIT_FAILURE);
    }

    for (int i = 0; i < k; i++) indices[i] = 0;
    combination[k] = '\0';

    while (1) {
      for (int i = 0; i < k; i++) {
        combination[i] = BASE_CHARS[indices[i]];
      }

      tokenizer->entries[index].idx = index;
      tokenizer->entries[index].val = (char*)malloc((k + 1) * sizeof(char));
      if (!tokenizer->entries[index].val) {
        fprintf(stderr, "Memory allocation failed for token at index %d\n", index);
        exit(EXIT_FAILURE);
      }
      memcpy(tokenizer->entries[index].val, combination, k + 1);
      index++;
      update_tqdm(&bar, 1, index == tokenizer->vocab_size);
      fflush(stdout);

      int i;
      for (i = k - 1; i >= 0; i--) {
        if (indices[i] < strlen(BASE_CHARS) - 1) {
          indices[i]++;
          break;
        }
        indices[i] = 0;
      }
      if (i < 0) break;
    }

    free(indices);
    free(combination);
  }
  close_tqdm(&bar);
}

void free_tokenizer(BaseTokenizer* tokenizer) {
  if (!tokenizer) {
    fprintf(stderr, "No instance of BaseTokenizer class found!");
    exit(EXIT_FAILURE);
  }
  free(tokenizer->entries->idx);
  free(tokenizer->entries->val);
  free(tokenizer->entries);
  free(tokenizer);
}