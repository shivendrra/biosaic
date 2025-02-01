#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "kmer.h"

void test_tokenization() {
  KMer* tokenizer = create_tokenizer(4);
  char* seq = "ATGCATG";
  char** kmers;
  int n_kmers;

  tokenize_sequence(tokenizer, seq, &kmers, &n_kmers);

  assert(n_kmers == 2); // Check correct k-mer count
  assert(strcmp(kmers[0], "A") == 0);
  assert(strcmp(kmers[1], "TGCA") == 0);

  printf("test_tokenization passed.\n");

  for (int i = 0; i < n_kmers; i++) free(kmers[i]);
  free(kmers);
  free_tokenizer(tokenizer);
}

void test_encoding_decoding() {
  KMer* tokenizer = create_tokenizer(4);
  char* seq = "ATGC";
  int* encoded;
  int n_kmers;

  encoded = encode_sequence(tokenizer, seq, &n_kmers);
  assert(n_kmers > 0); // Ensure something was encoded

  char* decoded = decode_sequence(tokenizer, encoded, n_kmers);
  assert(strcmp(decoded, seq) == 0); // Ensure decoding matches original

  printf("test_encoding_decoding passed.\n");

  free(encoded);
  free(decoded);
  free_tokenizer(tokenizer);
}

void test_invalid_token_handling() {
  KMer* tokenizer = create_tokenizer(4);
  char* seq = "ATGXM";
  int* encoded;
  int n_kmers;

  encoded = encode_sequence(tokenizer, seq, &n_kmers);

  for (int i = 0; i < n_kmers; i++) {
    if (encoded[i] == -1) {
      printf("test_invalid_token_handling passed.\n");
      free(encoded);
      free_tokenizer(tokenizer);
      return;
    }
  }

  assert(0 && "Expected invalid token handling.");
}

int main() {
  test_tokenization();
  test_encoding_decoding();
  test_invalid_token_handling();
  return 0;
}