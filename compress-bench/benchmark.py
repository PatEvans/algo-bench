"""
Module for generating prompts specific to the LLM C Compression benchmark.
"""
import random
from typing import Optional, List, Tuple # Corrected imports

# --- Prompt Generation ---

def generate_prompt_examples(num_examples: int = 2) -> List[Tuple[bytes, bytes, bytes]]:
    """Generates small input/compressed/output examples for the compression prompt."""
    examples = []
    # Example 1: Simple text
    original1 = b"hello hello world"
    # Example compressed data (could be anything, just needs to decompress correctly)
    # This is illustrative; a real LLM might produce different compressed forms.
    compressed1 = b"h\x01\x01w\x01" # Simplistic example representation
    examples.append((original1, compressed1, original1))

    # Example 2: Repeating data
    original2 = b"AAAAABBBBB"
    compressed2 = b"A\x05B\x05" # Simplistic example representation
    examples.append((original2, compressed2, original2))

    # Add more complex examples if needed, ensuring compressed data is plausible
    # and can be decompressed back to original by a hypothetical function.

    return examples[:num_examples] # Return exactly num_examples

def create_compression_prompt(examples: Optional[list[tuple[bytes, bytes, bytes]]] = None) -> str:
    """
    Creates a prompt to ask an LLM for lossless compression and decompression functions.
    """
    base_prompt = """
Generate C code implementing a lossless data compression algorithm.
Start your response *directly* with the necessary `#include` directives or the `typedef` statement. Do not include any introductory text, language identifiers (like 'c' or '```c'), or markdown formatting before the actual code begins.

The code MUST include the following three functions with the exact signatures specified:

1.  `typedef struct { unsigned char* data; size_t size; } Buffer;`
    (You MUST include this struct definition.)

2.  `Buffer compress(const unsigned char* input_data, size_t input_size);`
    - Takes a pointer `input_data` to the raw input bytes and its `input_size`.
    - Compresses the data.
    - Allocates memory for the compressed data using `malloc`.
    - Returns a `Buffer` struct containing the pointer to the allocated compressed data and its size.
    - If compression fails or input is invalid, it should return a Buffer with data=NULL and size=0.

3.  `Buffer decompress(const unsigned char* compressed_data, size_t compressed_size);`
    - Takes a pointer `compressed_data` to the compressed bytes and its `compressed_size`.
    - Decompresses the data.
    - Allocates memory for the original data using `malloc`.
    - Returns a `Buffer` struct containing the pointer to the allocated original data and its size.
    - If decompression fails or input is invalid, it should return a Buffer with data=NULL and size=0.

4.  `void free_buffer(Buffer buffer);`
    - Takes a `Buffer` struct (as returned by `compress` or `decompress`).
    - Frees the memory pointed to by `buffer.data` using `free`.
    - This function is crucial for memory management by the caller.

Constraints and Requirements:
- The code MUST be self-contained standard C (C99 or C11 recommended).
- Include necessary headers (like `<stdlib.h>`, `<string.h>`, `<stddef.h>`).
- DO NOT use external compression libraries (like zlib, zstd, etc.). Implement the algorithm directly.
- The functions MUST handle arbitrary byte sequences, not just text.
- The functions MUST be thread-safe if they use global state (prefer avoiding global state).
- The goal is low latency and good compression ratio. Aim for a balance.
- DO NOT include a `main` function. Provide only the struct definition and the three required functions.
- Ensure `malloc` return values are checked for NULL.
"""
    # Examples are less critical for C prompt structure but can be kept for conceptual illustration
    if examples:
        example_str = "\n\nConceptual Examples (Illustrative - actual C implementation details will differ):\n"
        for original, compressed, expected_decompressed in examples:
            example_str += f"Original Data (bytes): {repr(original)}\n"
            example_str += f"Conceptually Compressed To (bytes): {repr(compressed)}\n"
            example_str += f"Decompressed Back To (bytes): {repr(expected_decompressed)}\n\n"
        return base_prompt + example_str.strip()
    else:
        return base_prompt

# Evaluation logic removed, now handled by framework.benchmark_runner
# and framework.docker_exec_wrapper.py

# run_single_benchmark function removed, now handled by framework.app_base

# __main__ block removed as evaluation is triggered via Flask app
