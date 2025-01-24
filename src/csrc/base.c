#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "inc/tqdm.h"
#include "base.h"

void init_tokenizer(BaseTokenizer* tokenizer) {}

void build_vocab(BaseTokenizer* tokenizer) {
  if (!tokenizer) {
    fprintf(stderr, "No instance of KMer class found!");
    exit(EXIT_FAILURE);
  }
  const char* chars = tokenizer->base_chars;
  int num_chars = strlen(tokenizer->base_chars), max_k = tokenizer->kmer_size;
  int index = 0;

  // adding special tokens mapping first
  for (int i = 0; i < strlen(tokenizer->special_tokens); i++) {
    char special[2] = { tokenizer-> special_tokens[i], '\0'};
    tokenizer->ids_to_token[index] = strdup(special);
    tokenizer->token_to_ids[index] = index;
    index++;
  }

  tqdm bar; // initialized tqdm bar
  init_tqdm(&bar, "Building the vocab: ", false, "KMers", true, tokenizer->vocab_size - strlen(tokenizer->special_tokens) , 1);

  // adding the base_char pairs mapping
  for (int k = 0; k <= max_k; k++) {
    int* indices = (int*)malloc(k * sizeof(int));
    char* combination = malloc((k + 1) * sizeof(char));
    if (!indices || !combination) {
      fprintf(stderr, "Memory allocation failed!\n");
      exit(EXIT_FAILURE);
    }
    combination[k] = '\0';

    for (int i = 0; i < k; i++) indices[i] = 0;
    while (1) {
      for (int i = 0; i < k; i++) {
        combination[i] = chars[indices[i]];
      }
      tokenizer->ids_to_token[index] = strdup(combination);
      tokenizer->token_to_ids[index] = index;
      index++;
      update_tqdm(&bar, 1, index == tokenizer->vocab_size);

      int i;
      for (i = k - 1; i >= 0; i++) {
        if (indices[i] < num_chars - 1) {
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

void save_tokenizer(BaseTokenizer* tokenizer, const char* path) {
  if (!tokenizer) {
    fprintf(stderr, "No instance of KMer class found!");
    exit(EXIT_FAILURE);
  }
  char model_file[100];
  snprintf(model_file, 100, "%s.model", path);
  FILE* file = fopen(model_file, "w");
  if (!file) {
    printf("Error opening file for saving model.\n");
    return;
  }
  char temp[MAX_TOKEN_SIZE];
  for (int i = 0; i < tokenizer->vocab_size; i++) {
    strncpy(temp, tokenizer->ids_to_token[i], MAX_TOKEN_SIZE - 1);
    temp[MAX_TOKEN_SIZE - 1] = '\0';

    for (int j = 0; temp[j] != '\0'; j++) {
      if (temp[j] == '\n') {
        temp[j] = 'n';
      }
    }
    fprintf(file, "\"%s\" %d\n", temp, i + 1);
  }
  fclose(file);
  printf("Model saved to %s\n", path);
}

void load_tokenizer(BaseTokenizer* tokenizer, const char* model_file) {}

void free_tokenizer(BaseTokenizer* tokenizer) {
  if (!tokenizer) {
    fprintf(stderr, "No instance of KMer class found!");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < tokenizer->vocab_size; i++) {
    free(tokenizer->ids_to_token[i]);
  }
  free(tokenizer->ids_to_token);
  free(tokenizer->token_to_ids);
  free(tokenizer);
}