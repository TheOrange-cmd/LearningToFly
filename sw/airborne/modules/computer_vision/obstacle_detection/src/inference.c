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
extern void entry_obstacle(
    const float tensor_input[MODEL_INPUT_BATCH][MODEL_INPUT_CHANNELS]
                            [MODEL_INPUT_HEIGHT][MODEL_INPUT_WIDTH],
    float tensor_output[1][1][MODEL_OUTPUT_ROW_SIZE][MODEL_OUTPUT_COL_SIZE]);

extern void entry_border(const float tensor_input[1][3][30][30],
                         float tensor_output[1][1]);

typedef float obstacle_input_tensor_t[MODEL_INPUT_BATCH][MODEL_INPUT_CHANNELS]
                                     [MODEL_INPUT_HEIGHT][MODEL_INPUT_WIDTH];
typedef float obstacle_output_tensor_t[MODEL_OUTPUT_ROW_SIZE]
                                      [MODEL_OUTPUT_COL_SIZE];
typedef float border_input_tensor_t[1][3][30][30];
typedef float border_output_tensor_t[1][1];

static obstacle_input_tensor_t *obstacle_input = NULL;
static obstacle_output_tensor_t *obstacle_output = NULL;
static border_input_tensor_t *border_input = NULL;
static border_output_tensor_t *border_output = NULL;

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
    printf("[Inference] Failed to allocate tensors\n");
    cleanup_inference();
    return false;
  }

  return true;
}

bool run_obstacle_inference(const float *yuv_data, int width, int height,
                            struct model_output_t *output) {
  if (!obstacle_input || !obstacle_output || !yuv_data || !output) {
    return false;
  }

  if (width != MODEL_INPUT_WIDTH || height != MODEL_INPUT_HEIGHT) {
    printf("[Inference] Invalid obstacle input dimensions: %dx%d (expected "
           "%dx%d)\n",
           width, height, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
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