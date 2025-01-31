/*
  @base.h
  * base class for KMer tokenization processes
    ** has necessary building blocks for further classes
  * to be compiled with the respective file containing the main logic
*/

#ifndef __BASE__H__
#define __BASE__H__

#define  BASE_VOCAB_SIZE  6
#define  MAX_SPECIAL_TOKENS  6
#define  MAX_MERGES  10000
#define  MAX_TOKEN_SIZE  6

#define  BASE_CHARS  "ATCG\n"
#define  SPECIAL_CHARS  "MPSE "

typedef struct {
  int idx;
  char* val;
} KmerEntry;

typedef struct {
  int kmer_size;
  int vocab_size;
  KmerEntry* entries;
} BaseTokenizer;

void init_tokenizer(BaseTokenizer* tokenizer, int kmer_size);
void build_vocab(BaseTokenizer* tokenizer);
void save_tokenizer(BaseTokenizer* tokenizer, const char* file_path);
void load_tokenizer(BaseTokenizer* tokenizer, const char* model_file);
void free_tokenizer(BaseTokenizer* tokenizer);

#endif  //!__BASE__H__