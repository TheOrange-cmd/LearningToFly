// model_runner.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEBUG_MODE

// Forward declaration of the entry function from the generated code
void entry(const float tensor_input[1][3][240][520], float tensor_output[1][3][5][1]);

int main(int argc, char** argv) {
    #ifdef DEBUG_MODE
        printf("[DEBUG] Model runner starting.\n");
        fflush(stdout);
    #endif

    // Check command line arguments
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    
    const char* input_filename = argv[1];
    const char* output_filename = argv[2];
    
    #ifdef DEBUG_MODE
        printf("[DEBUG] Input file: %s\n", input_filename);
        printf("[DEBUG] Output file: %s\n", output_filename);
        fflush(stdout);
    #endif

    // Check if the input file exists before trying to open it
    #ifdef DEBUG_MODE
        FILE *check_file = fopen(input_filename, "rb");
        if (check_file == NULL) {
            printf("[ERROR] Input file does not exist or cannot be opened: %s\n", input_filename);
            fflush(stdout);
            return 1;
        }
        fclose(check_file);
        printf("[DEBUG] Input file exists and can be opened\n");
        fflush(stdout);
    #endif

    // Open input file
    FILE* input_file = fopen(input_filename, "rb");
    if (!input_file) {
        printf("[ERROR] Could not open input file %s\n", input_filename);
        fflush(stdout);
        return 1;
    }
    
    #ifdef DEBUG_MODE
        printf("[DEBUG] Input file opened successfully\n");
        
        // Get file size
        fseek(input_file, 0, SEEK_END);
        long file_size = ftell(input_file);
        fseek(input_file, 0, SEEK_SET);
        printf("[DEBUG] Input file size: %ld bytes\n", file_size);
        
        // Expected size in bytes
        size_t expected_size_bytes = 1 * 3 * 240 * 520 * sizeof(float);
        printf("[DEBUG] Expected input size: %zu bytes (%zu floats)\n", 
               expected_size_bytes, expected_size_bytes / sizeof(float));
               
        if (file_size != expected_size_bytes) {
            printf("[WARNING] File size mismatch! Expected %zu bytes, got %ld bytes\n", 
                   expected_size_bytes, file_size);
        }
        fflush(stdout);
    #endif
    
    // Allocate memory for input tensor - using static array
    #ifdef DEBUG_MODE
        printf("[DEBUG] Allocating memory for input tensor (1x3x240x520 = %d floats)\n", 
               1 * 3 * 240 * 520);
        fflush(stdout);
    #endif
    
    float tensor_input[1][3][240][520];
    
    #ifdef DEBUG_MODE
        printf("[DEBUG] Memory allocated for input tensor\n");
        fflush(stdout);
    #endif
    
    // Read input data
    size_t input_size = 1 * 3 * 240 * 520;
    
    #ifdef DEBUG_MODE
        printf("[DEBUG] Reading %zu floats (%zu bytes) from input file\n", 
               input_size, input_size * sizeof(float));
        fflush(stdout);
    #endif
    
    size_t read_count = fread(tensor_input, sizeof(float), input_size, input_file);
    fclose(input_file);
    
    #ifdef DEBUG_MODE
        printf("[DEBUG] Read %zu floats from input file\n", read_count);
        fflush(stdout);
    #endif
    
    if (read_count != input_size) {
        printf("[ERROR] Expected to read %zu elements, but got %zu\n", input_size, read_count);
        fflush(stdout);
        return 1;
    }
    
    #ifdef DEBUG_MODE
        printf("[DEBUG] Input data loaded successfully\n");
        printf("[DEBUG] First few values of input tensor: %f, %f, %f, %f\n", 
               tensor_input[0][0][0][0], tensor_input[0][0][0][1], 
               tensor_input[0][0][0][2], tensor_input[0][0][0][3]);
        fflush(stdout);
    #endif
    
    // Allocate memory for output tensor
    #ifdef DEBUG_MODE
        printf("[DEBUG] Allocating memory for output tensor (1x3x5x1 = %d floats)\n", 
               1 * 3 * 5 * 1);
        fflush(stdout);
    #endif
    
    float tensor_output[1][3][5][1];
    
    // Zero initialize output tensor to be safe
    #ifdef DEBUG_MODE
        printf("[DEBUG] Zero initializing output tensor\n");
        fflush(stdout);
    #endif
    
    memset(tensor_output, 0, sizeof(tensor_output));
    
    // Run the model inference
    #ifdef DEBUG_MODE
        printf("[DEBUG] Starting model inference...\n");
        fflush(stdout);
    #endif
    
    entry(tensor_input, tensor_output);
    
    #ifdef DEBUG_MODE
        printf("[DEBUG] Model inference completed\n");
        printf("[DEBUG] First few values of output tensor: %f, %f, %f, %f\n", 
               tensor_output[0][0][0][0], tensor_output[0][0][1][0], 
               tensor_output[0][0][2][0], tensor_output[0][0][3][0]);
        fflush(stdout);
    #endif
    
    // Write output to file
    #ifdef DEBUG_MODE
        printf("[DEBUG] Opening output file for writing: %s\n", output_filename);
        fflush(stdout);
    #endif
    
    FILE* output_file = fopen(output_filename, "wb");
    if (!output_file) {
        printf("[ERROR] Could not open output file %s\n", output_filename);
        fflush(stdout);
        return 1;
    }
    
    size_t output_size = 1 * 3 * 5 * 1;
    
    #ifdef DEBUG_MODE
        printf("[DEBUG] Writing %zu floats (%zu bytes) to output file\n", 
               output_size, output_size * sizeof(float));
        fflush(stdout);
    #endif
    
    size_t write_count = fwrite(tensor_output, sizeof(float), output_size, output_file);
    fclose(output_file);
    
    #ifdef DEBUG_MODE
        printf("[DEBUG] Wrote %zu floats to output file\n", write_count);
        fflush(stdout);
    #endif
    
    if (write_count != output_size) {
        printf("[ERROR] Expected to write %zu elements, but wrote %zu\n", 
               output_size, write_count);
        fflush(stdout);
        return 1;
    }
    
    printf("[SUCCESS] Inference completed successfully.\n");
    fflush(stdout);
    return 0;
}