/**
 *
 * Copyright (C) 2025 Daniel Rugge <d.j.rugge@student.tudelft.nl>
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */
/**
 * @file
 * sw/airborne/modules/computer_vision/obstacle_detection/src/inference.c
 *
 * @brief Queue functions for vision processing.
 *
 * @note Developement assisted by Claude Sonnet 3.5
 */

// inference.c
#include "inference.h"
#include "border_model_wrapper.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "debug_print.h"
DEFINE_DEBUG_PRINT("INFERENCE")

// Get access to the tensor unions from the generated model
extern union tensor_union_0 tu0;
extern union tensor_union_1 tu1;
extern union tensor_union_2 tu2;

// Forward declarations for both models
extern void
entry_obstacle(const float tensor_input[1][3][FRONT_MODEL_INPUT_SIZE]
                                       [FRONT_MODEL_INPUT_SIZE],
               float tensor_output[FRONT_MODEL_OUTPUT_ROW_SIZE]
                                  [FRONT_MODEL_OUTPUT_COL_SIZE]);

extern void entry_border(const float tensor_input[1][3][BOTTOM_MODEL_INPUT_SIZE]
                                                 [BOTTOM_MODEL_INPUT_SIZE],
                         float tensor_output[1][1]);

typedef float obstacle_input_tensor_t[1][3][FRONT_MODEL_INPUT_SIZE]
                                     [FRONT_MODEL_INPUT_SIZE];
typedef float obstacle_output_tensor_t[FRONT_MODEL_OUTPUT_ROW_SIZE]
                                      [FRONT_MODEL_OUTPUT_COL_SIZE];
typedef float border_input_tensor_t[1][3][BOTTOM_MODEL_INPUT_SIZE]
                                   [BOTTOM_MODEL_INPUT_SIZE];
typedef float border_output_tensor_t[1][1];

static obstacle_input_tensor_t *obstacle_input = NULL;
static obstacle_output_tensor_t *obstacle_output = NULL;
static border_input_tensor_t *border_input = NULL;
static border_output_tensor_t *border_output = NULL;

/**
 * @brief Initialize neural network inference system
 *
 * Allocates memory for input/output tensors of both obstacle and border models.
 * Must be called before any inference operations.
 *
 * @return true All tensor allocations succeeded
 * @return false Failed to allocate any tensor (cleans up partial allocations)
 */
bool init_inference(void) {
  // Allocate tensors for obstacle detection
  obstacle_input =
      (obstacle_input_tensor_t *)malloc(sizeof(obstacle_input_tensor_t));
  obstacle_output =
      (obstacle_output_tensor_t *)malloc(sizeof(obstacle_output_tensor_t));

  // Allocate tensors for border detection
  border_input = (border_input_tensor_t *)malloc(sizeof(border_input_tensor_t));
  border_output =
      (border_output_tensor_t *)malloc(sizeof(border_output_tensor_t));

  if (!obstacle_input || !obstacle_output || !border_input || !border_output) {
    debug_print("Failed to allocate tensors\n");
    cleanup_inference();
    return false;
  }

  return true;
}

/**
 * @brief Run obstacle detection model inference
 *
 * Processes YUV image data through the obstacle detection neural network.
 * Validates input dimensions match model requirements (FRONT_MODEL_INPUT_SIZE).
 * Copies YUV channels to input tensor and executes model.
 *
 * @param yuv_data Pointer to YUV422 planar data (Y followed by U then V)
 * @param width Input image width (must match FRONT_MODEL_INPUT_SIZE)
 * @param height Input image height (must match FRONT_MODEL_INPUT_SIZE)
 * @param output Structure to store model outputs (3 danger values)
 * @return true Inference succeeded and output populated
 * @return false Invalid inputs or allocation failure
 */
bool run_obstacle_inference(const float *yuv_data, int width, int height,
                            struct model_output_t *output) {
  if (!obstacle_input || !obstacle_output || !yuv_data || !output) {
    return false;
  }

  if (width != FRONT_MODEL_INPUT_SIZE || height != FRONT_MODEL_INPUT_SIZE) {
    debug_print("Invalid obstacle input dimensions: %dx%d (expected "
                "%dx%d)\n",
                width, height, FRONT_MODEL_INPUT_SIZE, FRONT_MODEL_INPUT_SIZE);
    return false;
  }

  // Get pointers to the YUV channels
  const float *y_data = yuv_data;
  const float *u_data = yuv_data + (width * height);
  const float *v_data = yuv_data + (2 * width * height);

  // Copy to input tensor
  for (int c = 0; c < 3; c++) {
    const float *channel_data;
    switch (c) {
    case 0:
      channel_data = y_data;
      break;
    case 1:
      channel_data = u_data;
      break;
    case 2:
      channel_data = v_data;
      break;
    }

    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int src_idx = h * width + w;
        (*obstacle_input)[0][c][h][w] = channel_data[src_idx];
      }
    }
  }

  entry_obstacle(*obstacle_input, *obstacle_output);

  // Copy results
  float *flat_output = (float *)obstacle_output;
  output->values[0][0] = flat_output[0];
  output->values[0][1] = flat_output[1];
  output->values[0][2] = flat_output[2];

  return true;
}

/**
 * @brief Run border detection model inference
 *
 * Processes YUV image data through the border detection neural network.
 * Centers U/V channels around zero (original range 0-1 -> -0.5 to 0.5).
 * Clamps output to [0.0, 1.0] range and handles NaN cases.
 *
 * @param yuv_data Pointer to YUV422 planar data (Y followed by U then V)
 * @param width Input image width (must match BOTTOM_MODEL_INPUT_SIZE)
 * @param height Input image height (must match BOTTOM_MODEL_INPUT_SIZE)
 * @param output Structure to store single border confidence value
 * @return true Inference succeeded and output populated
 * @return false Invalid inputs or allocation failure
 */
bool run_border_inference(const float *yuv_data, int width, int height,
                          struct border_output_t *output) {

  if (!border_input || !border_output || !yuv_data || !output) {
    debug_print("Null pointer check failed!");
    return false;
  }

  // debug prints for input ranges
  float y_min = INFINITY, y_max = -INFINITY;
  float u_min = INFINITY, u_max = -INFINITY;
  float v_min = INFINITY, v_max = -INFINITY;

  for (int i = 0; i < width * height; i++) {
    y_min = fminf(y_min, yuv_data[i]);
    y_max = fmaxf(y_max, yuv_data[i]);
    u_min = fminf(u_min, yuv_data[i + width * height]);
    u_max = fmaxf(u_max, yuv_data[i + width * height]);
    v_min = fminf(v_min, yuv_data[i + 2 * width * height]);
    v_max = fmaxf(v_max, yuv_data[i + 2 * width * height]);
  }

  // debug_print("Input ranges - Y: [%.3f, %.3f], U: [%.3f, %.3f], V: [%.3f,
  // %.3f]",
  //     y_min, y_max, u_min, u_max, v_min, v_max);

  const float *y_data = yuv_data;                    // Y channel (luminance)
  const float *u_data = yuv_data + (width * height); // U channel (Cb)
  const float *v_data = yuv_data + (2 * width * height); // V channel (Cr)

  // Process input data
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      int src_idx = h * width + w;

      // Verify U and V are properly centered around 0
      float u_val = u_data[src_idx];
      float v_val = v_data[src_idx];

      (*border_input)[0][0][h][w] = y_data[src_idx];
      (*border_input)[0][1][h][w] = u_val - 0.5f; // Explicitly center U
      (*border_input)[0][2][h][w] = v_val - 0.5f; // Explicitly center V
    }
  }

  // Add debug prints for intermediate values
  // debug_print("First few input values after copying:");
  // debug_print("Y: %.3f %.3f %.3f",
  //     (*border_input)[0][0][0][0],
  //     (*border_input)[0][0][0][1],
  //     (*border_input)[0][0][0][2]);
  // debug_print("U: %.3f %.3f %.3f",
  //     (*border_input)[0][1][0][0],
  //     (*border_input)[0][1][0][1],
  //     (*border_input)[0][1][0][2]);
  // debug_print("V: %.3f %.3f %.3f",
  //     (*border_input)[0][2][0][0],
  //     (*border_input)[0][2][0][1],
  //     (*border_input)[0][2][0][2]);

  entry_border(*border_input, *border_output);

  float model_output = (*border_output)[0][0];
  if (isnan(model_output)) {
    debug_print("WARNING: Model output is NaN");
    model_output = 0.0f;
  } else if (model_output > 1.0f) {
    debug_print("WARNING: Model output > 1.0: %.3f", model_output);
    model_output = 1.0f;
  } else if (model_output < 0.0f) {
    debug_print("WARNING: Model output < 0.0: %.3f", model_output);
    model_output = 0.0f;
  }
  output->value = model_output;
  // debug_print("Model output (inside function): %.6f", output->value);
  // debug_print("Output pointer address: %p", (void*)output);
  // debug_print("Output value address: %p", (void*)&(output->value));

  return true;
}

/**
 * @brief Cleanup inference system resources
 *
 * Releases all allocated tensor memory and nullifies pointers.
 * Should be called during module shutdown or after initialization failure.
 */
void cleanup_inference(void) {
  free(obstacle_input);
  free(obstacle_output);
  free(border_input);
  free(border_output);
  obstacle_input = NULL;
  obstacle_output = NULL;
  border_input = NULL;
  border_output = NULL;
}