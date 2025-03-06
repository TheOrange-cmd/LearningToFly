// inference.c
#include "inference.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Forward declaration with correct dimensions
extern void entry(const float tensor_input[1][3][520][240], float tensor_output[1][1][5][1]);

typedef float input_tensor_t[1][3][520][240];
typedef float output_tensor_t[1][1][5][1];

static input_tensor_t* input_tensor = NULL;
static output_tensor_t* output_tensor = NULL;

bool init_inference(void) {
    input_tensor = (input_tensor_t*)malloc(sizeof(input_tensor_t));
    if (!input_tensor) {
        printf("[Inference] Failed to allocate input tensor\n");
        return false;
    }

    output_tensor = (output_tensor_t*)malloc(sizeof(output_tensor_t));
    if (!output_tensor) {
        printf("[Inference] Failed to allocate output tensor\n");
        free(input_tensor);
        input_tensor = NULL;
        return false;
    }

    return true;
}

bool run_inference(const float* rgb_data, 
                  int width,
                  int height,
                  struct model_output_t* output) {
    if (!input_tensor || !output_tensor || !rgb_data || !output) {
        return false;
    }

    if (width != 240 || height != 520) {
        printf("[Inference] Invalid input dimensions: %dx%d (expected 240x520)\n", 
               width, height);
        return false;
    }

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int rgb_idx = (h * width + w) * 3;
            (*input_tensor)[0][0][h][w] = rgb_data[rgb_idx];     // R
            (*input_tensor)[0][1][h][w] = rgb_data[rgb_idx + 1]; // G
            (*input_tensor)[0][2][h][w] = rgb_data[rgb_idx + 2]; // B
        }
    }

    memset(output_tensor, 0, sizeof(output_tensor_t));
    entry(*input_tensor, *output_tensor);

    // Copy results to output structure
    for (int i = 0; i < 5; i++) {
        output->values[i] = (*output_tensor)[0][0][i][0];
    }

    return true;
}

void cleanup_inference(void) {
    if (input_tensor) {
        free(input_tensor);
        input_tensor = NULL;
    }
    if (output_tensor) {
        free(output_tensor);
        output_tensor = NULL;
    }
}