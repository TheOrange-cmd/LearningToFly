// inference.h
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdbool.h>
#include <stddef.h>  // for size_t
#include <stdint.h>
#include "model.h"
#include <math.h>  // for sqrtf and expf

// Structure to hold obstacle detection outputs
struct model_output_t {
    float values[MODEL_OUTPUT_ROW_SIZE][MODEL_OUTPUT_COL_SIZE]; 
};

// Structure to hold border detection output
struct border_output_t {
    float value;  // Single output value for border detection
};

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