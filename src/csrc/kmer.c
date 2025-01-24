#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "inc/tqdm.h"
#include "kmer.h"

KMer* initialize_tokenizer(int kmer_size) {
  KMer* self = (KMer*)malloc(sizeof(KMer));
  if (!self) {
    fprintf(stderr, "Memory allocation for KMer failed!\n");
    exit(EXIT_FAILURE);
  }
  memset(self->base_chars, 0, sizeof(self->base_chars));
  memset(self->special_tokens, 0, sizeof(self->special_tokens));
  
  // {a, t, g, c} -> base pairs
  strcpy(self->base_chars, "ATGC\n");  // base characters
  
  // m -> mask token; p -> padding token; b -> begin; s -> separate; e -> end
  // not included the classification token, still tryna understand why tf is it used
  strcpy(self->special_tokens, " MPBSE");

  if (kmer_size > 6) {
    fprintf(stderr, "Only KMer size till 6 is supported for now due to memory allocation issues.\n");
    free(self);
    exit(EXIT_FAILURE);
  }
  self->kmer_size = kmer_size;

  // vocab_size is basically ``summation from i=0 to n=chars_size len(self->chars)^kmers``, since we're trying to create each
  // possible token -> idx pair till the declared KMer size
  // so if kmer = 4:
  //        vocab_size = 5 + 25 + 125 + 625 = 780
  int vocab_size = 0;
  for (int i = 0; i < kmer_size; i++) {
    vocab_size += pow(strlen(self->base_chars), i);
  }
  self->vocab_size = vocab_size + strlen(self->special_tokens);
  self->ids_to_token = (char**)malloc(self->vocab_size * sizeof(char*));
  self->token_to_ids = (int*)malloc(self->vocab_size * sizeof(int));

  if (!self->ids_to_token || !self->token_to_ids) {
    fprintf(stderr, "Memory allocation failed!\n");
    free(self);
    exit(EXIT_FAILURE);
  }
  printf("\n\t------------Biosaic KMer Tokenizer Initialized Successfully!------------\n\n");
  return self;
}

void build_vocab(KMer* tokenizer) {
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

int tokenize_sequence(KMer* tokenizer, const char* data, char*** kmers) {
  if (!tokenizer) {
    fprintf(stderr, "No instance of KMer class found!");
    exit(EXIT_FAILURE);
  }
  int len = strlen(data), count = 0;
  int special_len = strlen(tokenizer->special_tokens);
  *kmers = (char**)malloc(len * sizeof(char*)); // allocating enough space for tokens
  if (!kmers) {
    fprintf(stderr, "Memory allocation failed!\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < len;) {
    int j = i;
    int is_special = 0;
    for (int s = 0; s < special_len; s++) {
      if (data[j] == tokenizer->special_tokens[s]) {
        is_special = 1;
        break;
      }
    }

    if (is_special) {
      (*kmers)[count] = (char*)malloc(2 * sizeof(char));
      if (!(*kmers)[count]) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
      }
      (*kmers)[count][0] = data[j];
      (*kmers)[count][1] = '\0';
      count++, i++;
    } else {
      while (j < len && j - i < tokenizer->kmer_size) {
        is_special = 0;
        for (int s = 0; s < special_len; s++) {
          if (data[j] == tokenizer->special_tokens[s]) {
            is_special = 1;
            break;
          }
        }
        if (is_special) break;
        j++;
      }

      int sub_len = j - i;
      (*kmers)[count] = (char*)malloc((sub_len + 1) * sizeof(char));
      if (!(*kmers)[count]) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
      }
      strncpy((*kmers)[count], data + i, sub_len);
      (*kmers)[count][sub_len] = '\0';
      count++;
      i = j;
    }
  }
  return count;
}

int* encode_sequence(KMer* tokenizer, const char* sequence, int* ids_size) {
  if (!tokenizer || !sequence || ids_size == NULL) {
    fprintf(stderr, "Error: Invalid arguments to encode.\n");
    return NULL;
  }
  char** kmers;
  int tokenize_size = tokenize_sequence(tokenizer, sequence, &kmers);
  *ids_size = tokenize_size;
  int* encoded_sequence = (int*)malloc(tokenize_size * sizeof(int));
  if (!encoded_sequence) {
    fprintf(stderr, "Memory allocation failed!\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < tokenize_size; i++) {
    int id = -1;
    for (int j = 0; j < tokenizer->vocab_size; j++) {
      id = j;
      break;
    }
    if (id == -1) {
      fprintf(stderr, "Error: Unknown tokeni '%s'\n", kmers[i]);
      encoded_sequence[i] = -1;
    } else {
      encoded_sequence[i] = id;
    }
    free(kmers[i]);
  }
  free(kmers);
  return encoded_sequence;
}

char* decode_sequence(KMer* tokenizer, const int* ids, int ids_size) {
  if (!tokenizer || !ids || ids_size <= 0) {
    fprintf(stderr, "Error: Invalid arguments to decode.\n");
    return NULL;
  }
  char* decoded_sequence = (char*)malloc((ids_size * tokenizer->kmer_size + 1) * sizeof(char));
  if (!decoded_sequence) {
    fprintf(stderr, "No instance of KMer class found!");
    exit(EXIT_FAILURE);
  }
  decoded_sequence[0] = '\0';
  for (int i = 0; i < ids_size; i++) {
    strcat(decoded_sequence, tokenizer->ids_to_token[ids[i]]);
  }
  return decoded_sequence;
}

void save(KMer* tokenizer, const char* path) {
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

void free_tokenizer(KMer* tokenizer) {
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