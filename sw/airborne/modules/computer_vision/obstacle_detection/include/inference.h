// inference.h
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "model.h"
#include <math.h>

// Structure to hold obstacle detection outputs
struct model_output_t {
    float values[MODEL_OUTPUT_ROW_SIZE][MODEL_OUTPUT_COL_SIZE]; 
};

// Structure to hold border detection output
struct border_output_t {
    float value;  // Single output value (already sigmoided in the model)
};

// Define a union type to hold either output format
typedef union {
    struct {
        float values[MODEL_OUTPUT_ROW_SIZE][MODEL_OUTPUT_COL_SIZE]; // [1][3]
    } obstacle;
    struct {
        float value;  // Single output value
    } border;
} model_data_t;

// Unified output structure
typedef struct {
    uint8_t type;  // Type identifier: 0 for obstacle, 1 for border
    model_data_t data;
} unified_model_output_t;

// Initialize both inference systems
bool init_inference(void);

// Run inference for obstacle detection
bool run_obstacle_inference(const float* rgb_data, 
                          int width,
                          int height,
                          struct model_output_t* output);

// Run inference for border detection
// bool run_border_inference(const float* rgb_data, 
//                          int width,
//                          int height,
//                          struct border_output_t* output);

void clip_tensor_values(float* tensor, size_t size, float min_val, float max_val);
void normalize_input_tensor(float* tensor, int width, int height, int channels);
void normalize_intermediate_tensor(float* tensor, int size);
bool run_border_inference(const float* rgb_data, int width, int height, struct border_output_t* output);

// Cleanup inference resources
void cleanup_inference(void);

#endif // INFERENCE_H