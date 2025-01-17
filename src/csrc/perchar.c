#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "perchar.h"

PerChar* init_tokenizer() {
  PerChar* self = (PerChar*)malloc(sizeof(PerChar));
  if (!self) {
    fprintf(stderr, "Failed to initialize the tokenizer\n");
    exit(1);
  }
  strcpy(self->chars, "\nATGCMPBSE");  // base characters
  // {a, t, g, c} -> base pairs
  // m -> mask token; p -> padding token; b -> begin; s -> separate; e -> end
  // not included the classification token, i'm still tryna understand why tf is it used
  // classifaction token: https://aditya007.medium.com/understanding-the-cls-token-in-bert-a-comprehensive-guide-a62b3b94a941
  self->vocab_size = strlen(self->chars);
  for (int i = 0; i < self->vocab_size; i++) {
    self->str_to_idx[(int)self->chars[i]] = i;
    self->id_to_str[i] = self->chars[i];
  }
  return self;
}

int* encode_sequence(PerChar *tokenizer, const char *string, size_t* encoded_size) {
  size_t len = strlen(string);
  int* encoded = (int*)malloc(len * sizeof(int));
  *encoded_size = len;
  for (int i = 0; i < *encoded_size; i++) {
    if(tokenizer->str_to_idx[(int)string[i]] >= 0) {
      encoded[i] = tokenizer->str_to_idx[(int)string[i]];
    } else {
      // this logic handles the excpetion
      // when there's a new character that's not in the vocab; it's added to a new sepcial_idx
      // special_idx = vocab_size + i
      int special_index = tokenizer->vocab_size;
      tokenizer->str_to_idx[(int)string[i]] = special_index;
      tokenizer->id_to_str[i] = string[i];
      tokenizer->vocab_size++;
      encoded[i] = special_index;
    }
  }
  return encoded;
}

char* decode_sequence(PerChar* tokenizer, const int* encoded, size_t encoded_size) {
  char* decoded = (char*)malloc((encoded_size + 1) * sizeof(char));
  for (int i = 0; i < encoded_size; i++) {
    decoded[i] = tokenizer->id_to_str[encoded[i]];
  }
  decoded[encoded_size] = '\0'; // ensures proper line termination
  return decoded;
}

void free_tokenizer(PerChar* tokenizer) {
  free(tokenizer);
}