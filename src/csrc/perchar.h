/*
  perchar.h
  - per-character tokenizer, tokenizes on the based set of vocab
  - set size of vocab, no need of loading or saving vocab
  - compile it as:
    -- '.so': g++ -shared -fPIC -o libchar.so perchar.c / for linux
    -- '.dll': g++ -shared -o libchar.dll perchar.c / for windows
*/

#ifndef __PERCHAR__H__
#define __PERCHAR__H__

#define MAX_CHARS 256
#define MAX_STRING_SIZE 1000

typedef struct {
  char chars[MAX_CHARS];
  size_t vocab_size;
  int str_to_idx[MAX_CHARS];
  char id_to_str[MAX_CHARS];
} PerChar;

PerChar* init_tokenizer();
int* encode_sequence(PerChar* tokenizer, const char* string, size_t* encoded_length);
char* decode_sequence(PerChar* tokenizer, const int* encoded, size_t len);
void free_tokenizer(PerChar* tokenizer);

#endif  //!__PERCHAR__H__