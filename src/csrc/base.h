#ifndef __BASE__H__
#define __BASE__H__

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
} BaseTokenizer;

void init_tokenizer(BaseTokenizer* tokenizer);
void build_vocab(BaseTokenizer* tokenizer);
void save_tokenizer(const BaseTokenizer* tokenizer, const char* file_path);
void load_tokenizer(BaseTokenizer* tokenizer, const char* model_file);
void free_tokenizer(BaseTokenizer* tokenizer);

#endif  //!__BASE__H__