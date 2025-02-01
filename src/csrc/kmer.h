/*
  kmer.h
  * main ``KMer`` class codes in this file
  * tokenizes the dna data based on the vocab & respective kmer size
  * compile it as:
    ** '.so': gcc -shared -fPIC -o libkmer.so kmer.c / for linux
    ** '.dll': gcc -shared -o libkmer.dll kmer.c / for windows
*/

#ifndef __KMER__H__
#define __KMER__H__

#define  BASE_VOCAB_SIZE  6
#define  MAX_SPECIAL_TOKENS  6
#define  MAX_MERGES  10000
#define  MAX_TOKEN_SIZE  6

typedef struct {
  char base_chars[BASE_VOCAB_SIZE];
  char special_tokens[MAX_SPECIAL_TOKENS];
  int kmer_size;
  int vocab_size;
  char** ids_to_token;
  int* token_to_ids;
} KMer;

KMer* create_tokenizer(int kmer_size);
int tokenize_sequence(KMer* tokenizer, const char* data, char*** kmers);
void build_vocab(KMer* tokenizer);
int* encode_sequence(KMer* tokenizer, const char* sequence, int* ids_size);
char* decode_sequence(KMer* tokenizer, const int* ids, int ids_size);
void save(KMer* tokenizer, const char* path);
void free_tokenizer(KMer* tokenizer);

#endif