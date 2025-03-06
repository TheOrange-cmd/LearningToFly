#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdbool.h>
#include <stdint.h>

// Structure to hold model outputs
struct model_output_t {
    float values[5];  // [columns]
};

// Initialize the inference system
bool init_inference(void);

// Run inference on RGB input data
// Input should be normalized float values (0-1) in RGB format
// Returns true on success, false on failure
bool run_inference(const float* rgb_data, 
                  int width,
                  int height,
                  struct model_output_t* output);

// Cleanup inference resources
void cleanup_inference(void);

#endif // INFERENCE_H