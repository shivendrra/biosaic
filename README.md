# Biosaic Tokenizer

## Overview
The Biosaic KMer Tokenizer is a sequence-processing library designed for bioinformatics applications. It tokenizes DNA sequences into k-mers (subsequences of length `k`) and encodes these tokens into unique IDs for further processing. Additionally, the library provides decoding, vocabulary management, and serialization functionality.

## Features
- **Tokenization**: Converts input sequences into k-mers.
- **Encoding**: Maps k-mers to unique integer IDs.
- **Decoding**: Reconstructs sequences from encoded IDs.
- **Vocabulary Building**: Generates a vocabulary of all possible k-mers and special tokens.
- **Serialization**: Saves and loads tokenizers for reuse.
- **Support for Special Tokens**: Includes special tokens like masking, padding, start, end, and separator.

## Repository Structure
```
biosaic/
├── src/
│   └── build/
│   │   └── libkmer.so        # complied .so file
│   └── csrc/
│   │   └── inc/
│   │   │   └── tqdm.c        # python's tqdm implemented in c
│   │   └── kmer.c            # core C implementation of the KMer tokenizer
│   │   └── kmer.h            # header file for the C KMer implementation
│   │   └── perchar.c         # core C implementation of the PerChar tokenizer
│   │   └── perchar.h         # header file for the C PerChar implementation
│   ├── cbase.py              # python-C interface for the C library
│   ├── main.py               # main script to test the tokenizer functionality
│   ├── kmer.py               # pure-python implementation of similar kinda code logic
├── README.md                 # Project documentation
├── test.py
```

## Prerequisites
### System Requirements
- **Operating System**: Linux, macOS, or Windows with support for GCC or Clang.
- **Python**: Version 3.7 or higher.
- **C Compiler**: GCC or Clang for compiling the C library.

### Dependencies
- **Python Modules**:
  - `ctypes`: For interfacing Python with the C library.
  - `os`: For file and path handling.
- **C Libraries**:
  - Standard C libraries for memory allocation and string handling.


## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shivendrra/biosaic.git
   cd biosaic
   ```

2. **Compile the C Library**:
   Navigate to the `src` directory and compile the C code:
   ```bash
   cd src
   gcc -shared -o build/libkmer.so -fPIC kmer.c
   ```

3. **Run the Python Script**:
   Test the tokenizer using the `main.py` script:
   ```bash
   python src/main.py
   ```

## Usage
Create an instance of the tokenizer with a specified k-mer size, & split them into tokens, encode & decode them fastly:
```python
from .src import KMer

tokenizer = KMer(kmer=4)
sequence = "BAACATGTCCTGCATGGCATTAMGTTTGTTGGGGCAGTGCCCGPGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAAE"
print("shreded sequence: ", tokenizer._shred(sequence))
encoded = tokenizer.encode(sequence)
decoded = tokenizer.decode(encoded)

print("Encoded sequence:", encoded)
print("Decoded sequence:", decoded)
print("decoded string matches the original string:", decoded == sequence)

tokenizer.save("./vocab")
del tokenizer
```

## Debugging
1. Use `valgrind` (Linux) or similar tools to detect memory issues:
   ```bash
   valgrind --leak-check=full ./build/libkmer.so
   ```
2. Enable verbose logging in the Python script to trace issues.

## Contributing
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Create a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.