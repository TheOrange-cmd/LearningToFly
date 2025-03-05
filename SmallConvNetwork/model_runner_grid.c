// model_runner_grid.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Forward declaration of the entry function from onnx2c-generated code
void entry(const float tensor_input[1][3][240][520], float tensor_output[1][3][5][1]);

int main(int argc, char** argv) {
    // Check command line arguments
    if (argc != 3) {
        printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    
    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    // Open input file
    FILE* input_file = fopen(input_filename, "rb");
    if (!input_file) {
        printf("Error: Could not open input file %s\n", input_filename);
        return 1;
    }
    
    // Allocate memory for input and output tensors
    float (*heap_input)[3][240][520] = malloc(sizeof(float[1][3][240][520]));
    if (!heap_input) {
        printf("Error: Failed to allocate memory for input tensor\n");
        fclose(input_file);
        return 1;
    }
    
    float (*heap_output)[3][5][1] = malloc(sizeof(float[1][3][5][1]));
    if (!heap_output) {
        printf("Error: Failed to allocate memory for output tensor\n");
        free(heap_input);
        fclose(input_file);
        return 1;
    }
    
    // Read input data
    size_t input_elements = 1 * 3 * 240 * 520;
    size_t read_count = fread(heap_input, sizeof(float), input_elements, input_file);
    fclose(input_file);
    
    if (read_count != input_elements) {
        printf("Error: Expected to read %zu elements, but got %zu\n", input_elements, read_count);
        free(heap_input);
        free(heap_output);
        return 1;
    }
    
    // Zero initialize output tensor
    memset(heap_output, 0, sizeof(float[1][3][5][1]));
    
    // Run the model inference
    entry(*heap_input, *heap_output);
    
    // Write output to file
    FILE* output_file = fopen(output_filename, "wb");
    if (!output_file) {
        printf("Error: Could not open output file %s\n", output_filename);
        free(heap_input);
        free(heap_output);
        return 1;
    }
    
    size_t output_elements = 1 * 3 * 5 * 1;
    size_t write_count = fwrite(heap_output, sizeof(float), output_elements, output_file);
    fclose(output_file);
    
    if (write_count != output_elements) {
        printf("Error: Expected to write %zu elements, but wrote %zu\n", output_elements, write_count);
    }
    
    // Print some output values for debugging
    printf("Model output summary:\n");
    for (int danger_level = 0; danger_level < 3; danger_level++) {
        printf("Danger level %d:\n", danger_level);
        for (int col = 0; col < 5; col++) {
            printf("  Column %d: %.4f\n", col, (*heap_output)[danger_level][col][0]);
        }
    }
    
    // Free allocated memory
    free(heap_input);
    free(heap_output);
    
    return 0;
}