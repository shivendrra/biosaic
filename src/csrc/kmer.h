/*
  kmer.h
  - main ``KMer`` class codes in this file
  - tokenizes the dna data based on the vocab & respective kmer size
  - compile it as:
    -- '.so': g++ -shared -fPIC -o libkmer.so kmer.c / for linux
    -- '.dll': g++ -shared -o libkmer.dll kmer.c / for windows
*/

#ifndef __KMER__H__
#define __KMER__H__

#define  MAX_BASE_CHARS  6
#define  SPECIAL_TOKEN_COUNT  6
#define  MAX_TOKEN_SIZE  6
#define  MAX_VOCAB_SIZE  19530  // supporting till kmer size of 6

typedef struct {
  char chars[MAX_BASE_CHARS];
  char special_tokens[SPECIAL_TOKEN_COUNT];
  int kmers;
  int vocab_size;
  char** id_to_token;
  int* token_to_id;
} KMer;

KMer* create_tokenizer(int kmers);
void tokenize_sequence(KMer* tokenizer, const char* data, char*** kmers, int* n_kmers);
void build_vocab(KMer* tokenizer);
int* encode_sequence(KMer* tokenizer, const char* seq, int* encoded_size);
char* decode_sequence(KMer* tokenizer, const int* encoded_seq, int encoded_size);
void save(KMer* tokenizer, const char* path);
void free_tokenizer(KMer* tokenizer);

#endif  //!__KMER__H__