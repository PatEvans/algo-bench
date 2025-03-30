#include <stdlib.h>
#include <string.h>
#include <stddef.h> // For size_t

// Required struct definition
typedef struct {
    unsigned char* data;
    size_t size;
} Buffer;

// Baseline C compress: Allocates memory and copies input data (no compression)
// This serves as a baseline for C execution overhead via ctypes.
Buffer compress(const unsigned char* input_data, size_t input_size) {
    Buffer output_buffer;
    output_buffer.data = NULL;
    output_buffer.size = 0;

    if (input_data == NULL && input_size > 0) {
        return output_buffer; // Invalid input
    }
    if (input_size == 0) {
         // Handle zero-size input: return empty buffer allocated
         output_buffer.data = (unsigned char*)malloc(0); // Allocate 0 bytes is implementation-defined but often works
         output_buffer.size = 0;
         // No need to check malloc(0) for NULL usually, but safe practice varies
         return output_buffer;
    }


    // Allocate memory for the "compressed" data (just a copy)
    output_buffer.data = (unsigned char*)malloc(input_size);
    if (output_buffer.data == NULL) {
        // Allocation failed
        return output_buffer; // Return empty buffer
    }

    // Copy input data to output buffer
    memcpy(output_buffer.data, input_data, input_size);
    output_buffer.size = input_size;

    return output_buffer;
}

// Baseline C decompress: Allocates memory and copies input data (no decompression)
Buffer decompress(const unsigned char* compressed_data, size_t compressed_size) {
    Buffer output_buffer;
    output_buffer.data = NULL;
    output_buffer.size = 0;

     if (compressed_data == NULL && compressed_size > 0) {
        return output_buffer; // Invalid input
    }
     if (compressed_size == 0) {
         // Handle zero-size input: return empty buffer allocated
         output_buffer.data = (unsigned char*)malloc(0);
         output_buffer.size = 0;
         return output_buffer;
    }

    // Allocate memory for the "decompressed" data (just a copy)
    output_buffer.data = (unsigned char*)malloc(compressed_size);
    if (output_buffer.data == NULL) {
        // Allocation failed
        return output_buffer; // Return empty buffer
    }

    // Copy compressed data to output buffer
    memcpy(output_buffer.data, compressed_data, compressed_size);
    output_buffer.size = compressed_size;

    return output_buffer;
}

// Function to free the allocated buffer data
void free_buffer(Buffer buffer) {
    if (buffer.data != NULL) {
        free(buffer.data);
        // Optional: Set to NULL after free to prevent double-free issues if struct is reused
        // buffer.data = NULL;
        // buffer.size = 0;
    }
}
