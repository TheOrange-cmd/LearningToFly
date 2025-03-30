// inference.h
#ifndef INFERENCE_H
#define INFERENCE_H

#include "model.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Structure to hold obstacle detection outputs
struct model_output_t {
  float values[FRONT_MODEL_OUTPUT_ROW_SIZE][FRONT_MODEL_OUTPUT_COL_SIZE];
};

// Structure to hold border detection output
struct border_output_t {
  float value;
};

// Define a union type to hold either output format
typedef union {
  struct {
    float values[FRONT_MODEL_OUTPUT_ROW_SIZE][FRONT_MODEL_OUTPUT_COL_SIZE];
  } obstacle;
  struct {
    float value; // Single output value
  } border;
} model_data_t;

// Unified output structure
typedef struct {
  uint8_t type; // Type identifier: 0 for obstacle, 1 for border
  model_data_t data;
} unified_model_output_t;

// Initialize both inference systems
bool init_inference(void);

// Run inference for obstacle detection
bool run_obstacle_inference(const float *rgb_data, int width, int height,
                            struct model_output_t *output);
bool run_border_inference(const float *rgb_data, int width, int height,
                          struct border_output_t *output);

// Cleanup inference resources
void cleanup_inference(void);

#endif // INFERENCE_H