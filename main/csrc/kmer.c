#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "kmer.h"
#include "inc/tqdm.h"

KMer* create_tokenizer(int kmers) {
  KMer* self = (KMer*)malloc(sizeof(KMer));
  memset(self->chars, 0, sizeof(self->chars));
  memset(self->special_tokens, 0, sizeof(self->special_tokens));

  // {a, t, g, c, \n} -> base pairs
  strcpy(self->chars, "ATGC\n");  // base characters
  
  // m -> mask token; p -> padding token; b -> begin; s -> separate; e -> end
  // not included the classification token, still tryna understand why tf is it used
  strcpy(self->special_tokens, "MPBSE");
  if (kmers > 6) {
    fprintf(stderr, "KMer size till 6 is supported for now due memory allocation issues\n");
    exit(1);
  }
  self->kmers = kmers;

  // vocab_size is basically ``summation from i=0 to n=chars_size len(self->chars)^kmers``, since we're trying to create each
  // possible token -> idx pair till the declared KMer size
  // so if kmer = 4:
  //        vocab_size = 5 + 25 + 125 + 625 = 780
  int vocab_size = 0;
  for (int i = 1; i <= kmers; i++) {
    vocab_size += pow(strlen(self->chars), i);
  }
  self->vocab_size = vocab_size + strlen(self->special_tokens);
  self->id_to_token = (char**)malloc((vocab_size + 1) * sizeof(char*));
  self->token_to_id = (int*)malloc((vocab_size + 1) * sizeof(int));
  return self;
}

void tokenize_sequence(KMer* tokenizer, const char* data, char*** kmers, int* n_kmers) {
  int len = strlen(data);
  int special_len = strlen(tokenizer->special_tokens);
  int count = 0;

  *kmers = (char**)malloc((len + 1) * sizeof(char*));  // allocating enough space for tokens

  for (int i = 0; i < len;) {
    int j = i;
    int is_special = 0;

    // checking for special token at the current position
    for (int s = 0; s < special_len; s++) {
      if (data[j] == tokenizer->special_tokens[s]) {
        is_special = 1;
        break;
      }
    }

    if (is_special) {  // handle special token as a separate substring
      (*kmers)[count] = (char*)malloc(2 * sizeof(char));
      (*kmers)[count][0] = data[j];
      (*kmers)[count][1] = '\0';
      count++;
      i++;  // move to the next character
    } else {  // extract sub-token until a special token is found or k-mer length is reached
      while (j < len && j - i < tokenizer->kmers) {
        is_special = 0;
        for (int s = 0; s < special_len; s++) {
          if (data[j] == tokenizer->special_tokens[s]) {
            is_special = 1;
            break;
          }
        }
        if (is_special) break;  // stop if a special token is found
        j++;
      }

      int sub_len = j - i;
      (*kmers)[count] = (char*)malloc((sub_len + 1) * sizeof(char));
      strncpy((*kmers)[count], data + i, sub_len);
      (*kmers)[count][sub_len] = '\0';
      count++;
      i = j;  // moving to the next position
    }
  }

  *n_kmers = count;
}

void build_vocab(KMer* tokenizer) {
  const char *chars = tokenizer->chars;
  int num_chars = strlen(chars), max_k = tokenizer->kmers;
  int index = 0;

  // adding special tokens first
  for (int i = 0; i < strlen(tokenizer->special_tokens); i++) {
    char special[2] = { tokenizer->special_tokens[i], '\0' };
    tokenizer->id_to_token[index] = strdup(special);
    tokenizer->token_to_id[index] = index;
    index++;
  }

  tqdm bar;
  init_tqdm(&bar, "Building the vocab: ", false, "pairs", true, tokenizer->vocab_size - strlen(tokenizer->special_tokens), 1);

  for (int k = 1; k <= max_k; k++) {
    int *indices = (int *)malloc(k * sizeof(int));
    if (!indices) {
      fprintf(stderr, "Memory allocation failed\n");
      return;
    }
    char *combination = (char *)malloc((k + 1) * sizeof(char));
    combination[k] = '\0';

    for (int i = 0; i < k; i++) {
      indices[i] = 0;
    }

    while (1) {
      for (int i = 0; i < k; i++) {
        combination[i] = chars[indices[i]];
      }
      tokenizer->id_to_token[index] = strdup(combination);
      tokenizer->token_to_id[index] = index;
      index++;
      update_tqdm(&bar, 1, index == tokenizer->vocab_size);
      fflush(stdout);

      int i;
      for (i = k - 1; i >= 0; i--) {
        if (indices[i] < num_chars - 1) {
          indices[i]++;
          break;
        }
        indices[i] = 0;
      }
      if (i < 0) {
        break;
      }
    }
    free(indices);
    free(combination);
  }
  close_tqdm(&bar);
}

int* encode_sequence(KMer* tokenizer, const char* seq, int* encoded_size) {
  char** kmers;
  int n_kmers;
  tokenize_sequence(tokenizer, seq, &kmers, &n_kmers);
  *encoded_size = n_kmers;
  int* encoded_seq = (int*)malloc(n_kmers * sizeof(int));

  for (int i = 0; i < n_kmers; i++) {
    int id = -1;
    for (int j = 0; j < tokenizer->vocab_size; j++) {
      if (strcmp(kmers[i], tokenizer->id_to_token[j]) == 0) {
        id = j;
        break;
      }
    }
    if (id == -1) {
      fprintf(stderr, "Error: Unknown token '%s'\n", kmers[i]);
      encoded_seq[i] = -1;
    } else {
      encoded_seq[i] = id;
    }
    free(kmers[i]);
  }
  free(kmers);
  return encoded_seq;
}

char *decode_sequence(KMer* tokenizer, const int* encoded_sequence, int encoded_size) {
  int k = tokenizer->kmers;
  char* decoded_sequence = (char *)malloc((encoded_size * k + 1) * sizeof(char));
  decoded_sequence[0] = '\0';
  for (int i = 0; i < encoded_size; i++) {
    strcat(decoded_sequence, tokenizer->id_to_token[encoded_sequence[i]]);
  }
  return decoded_sequence;
}

void save(KMer* tokenizer, const char* path) {
  char model_file[100];
  snprintf(model_file, 100, "%s.model", path);
  FILE* file = fopen(model_file, "w");
  if (!file) {
    printf("Error opening file for saving model.\n");
    return;
  }
  char temp[MAX_TOKEN_SIZE];
  for (int i = 0; i < tokenizer->vocab_size; i++) {
    strncpy(temp, tokenizer->id_to_token[i], MAX_TOKEN_SIZE - 1);
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
  for (int i = 0; i < tokenizer->vocab_size; i++) {
    free(tokenizer->id_to_token[i]);
  }
  free(tokenizer->id_to_token);
  free(tokenizer->token_to_id);
  free(tokenizer);
}