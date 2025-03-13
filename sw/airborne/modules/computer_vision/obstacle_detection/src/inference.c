// inference.c
#include "inference.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "model.h"
#include "border_model_wrapper.h"

// Get access to the tensor unions from the generated model
extern union tensor_union_0 tu0;
extern union tensor_union_1 tu1;
extern union tensor_union_2 tu2;


// Forward declarations for both models
extern void entry_obstacle(const float 
    tensor_input[MODEL_INPUT_BATCH][MODEL_INPUT_CHANNELS][MODEL_INPUT_HEIGHT][MODEL_INPUT_WIDTH], 
    float tensor_output[1][1][MODEL_OUTPUT_ROW_SIZE][MODEL_OUTPUT_COL_SIZE]);

extern void entry_border(const float 
    tensor_input[1][4][240][240], 
    float tensor_output[1][1]);
    
typedef float obstacle_input_tensor_t[MODEL_INPUT_BATCH][MODEL_INPUT_CHANNELS][MODEL_INPUT_HEIGHT][MODEL_INPUT_WIDTH];
typedef float obstacle_output_tensor_t[1][1][MODEL_OUTPUT_ROW_SIZE][MODEL_OUTPUT_COL_SIZE];
typedef float border_input_tensor_t[1][4][240][240];
typedef float border_output_tensor_t[1][1];

static obstacle_input_tensor_t* obstacle_input = NULL;
static obstacle_output_tensor_t* obstacle_output = NULL;
static border_input_tensor_t* border_input = NULL;
static border_output_tensor_t* border_output = NULL;

void clip_tensor_values(float* tensor, size_t size, float min_val, float max_val) {
    for(size_t i = 0; i < size; i++) {
        if(tensor[i] < min_val) tensor[i] = min_val;
        if(tensor[i] > max_val) tensor[i] = max_val;
    }
}

void normalize_input_tensor(float* tensor, int width, int height, int channels) {
    // First pass - get mean and std
    float sum = 0.0f;
    float sq_sum = 0.0f;
    int size = width * height * channels;
    
    for(int i = 0; i < size; i++) {
        sum += tensor[i];
        sq_sum += tensor[i] * tensor[i];
    }
    
    float mean = sum / size;
    float std = sqrtf(sq_sum/size - mean*mean);
    
    // Second pass - normalize
    for(int i = 0; i < size; i++) {
        tensor[i] = (tensor[i] - mean) / (std + 1e-5f);
        // Clip to reasonable range
        if(tensor[i] < -3.0f) tensor[i] = -3.0f;
        if(tensor[i] > 3.0f) tensor[i] = 3.0f;
    }
}

void normalize_intermediate_tensor(float* tensor, int size) {
    float sum = 0.0f;
    float sq_sum = 0.0f;
    
    for(int i = 0; i < size; i++) {
        sum += tensor[i];
        sq_sum += tensor[i] * tensor[i];
    }
    
    float mean = sum / size;
    float std = sqrtf(sq_sum/size - mean*mean);
    
    for(int i = 0; i < size; i++) {
        tensor[i] = (tensor[i] - mean) / (std + 1e-5f);
        // Clip to reasonable range
        if(tensor[i] < -3.0f) tensor[i] = -3.0f;
        if(tensor[i] > 3.0f) tensor[i] = 3.0f;
    }
}

bool init_inference(void) {
    // Allocate tensors for obstacle detection
    obstacle_input = (obstacle_input_tensor_t*)malloc(sizeof(obstacle_input_tensor_t));
    obstacle_output = (obstacle_output_tensor_t*)malloc(sizeof(obstacle_output_tensor_t));
    
    // Allocate tensors for border detection
    border_input = (border_input_tensor_t*)malloc(sizeof(border_input_tensor_t));
    border_output = (border_output_tensor_t*)malloc(sizeof(border_output_tensor_t));
    
    if (!obstacle_input || !obstacle_output || !border_input || !border_output) {
        printf("[Inference] Failed to allocate tensors\n");
        cleanup_inference();
        return false;
    }

    return true;
}

bool run_obstacle_inference(const float* rgb_data, 
    int width,
    int height,
    struct model_output_t* output) {
    if (!obstacle_input || !obstacle_output || !rgb_data || !output) {
        return false;
    }

    if (width != MODEL_INPUT_WIDTH || height != MODEL_INPUT_HEIGHT) {
        printf("[Inference] Invalid obstacle input dimensions: %dx%d (expected %dx%d)\n", 
        width, height, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);
        return false;
    }

    memcpy(*obstacle_input, rgb_data, sizeof(obstacle_input_tensor_t));
    memset(obstacle_output, 0, sizeof(obstacle_output_tensor_t));

    entry_obstacle(*obstacle_input, *obstacle_output);

    // Copy results to output structure 
    for (int i = 0; i < MODEL_OUTPUT_ROW_SIZE; i++) {
        for (int j = 0; j < MODEL_OUTPUT_COL_SIZE; j++) {
            output->values[i][j] = (*obstacle_output)[0][0][i][j];
        }
    }

    return true;
}

bool run_border_inference(const float* rgb_data, 
    int width,
    int height,
    struct border_output_t* output) {
    if (!border_input || !border_output || !rgb_data || !output) {
        return false;
    }

    // Calculate input statistics before normalization
    float sum = 0.0f;
    float sum_sq = 0.0f;
    float min_val = rgb_data[0];
    float max_val = rgb_data[0];
    
    for(int i = 0; i < width * height * 3; i++) {
        float val = rgb_data[i];
        sum += val;
        sum_sq += val * val;
        if(val < min_val) min_val = val;
        if(val > max_val) max_val = val;
    }
    
    float mean_raw = sum / (width * height * 3);
    float std_raw = sqrtf((sum_sq / (width * height * 3)) - (mean_raw * mean_raw));

    // printf("[Inference] Input stats - Mean: %.3f, Std: %.3f, Min: %.3f, Max: %.3f\n",
    //        mean_raw, std_raw, min_val, max_val);

    // Clear state
    memset(border_input, 0, sizeof(*border_input));
    memset(border_output, 0, sizeof(*border_output));
    clear_border_tensors();
    
    // Process input
    float* input_ptr = (float*)(*border_input);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};
    
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            for(int c = 0; c < 3; c++) {
                int src_idx = (y * width + x) * 3 + c;
                float val = rgb_data[src_idx] / 255.0f;
                val = (val - mean[c]) / (std[c] + 1e-6f);
                if(val < -3.0f) val = -3.0f;
                if(val > 3.0f) val = 3.0f;
                input_ptr[src_idx] = val;
            }
        }
    }

    // Calculate normalized input statistics
    sum = 0.0f;
    sum_sq = 0.0f;
    min_val = input_ptr[0];
    max_val = input_ptr[0];
    
    for(int i = 0; i < width * height * 3; i++) {
        float val = input_ptr[i];
        sum += val;
        sum_sq += val * val;
        if(val < min_val) min_val = val;
        if(val > max_val) max_val = val;
    }
    
    float mean_norm = sum / (width * height * 3);
    float std_norm = sqrtf((sum_sq / (width * height * 3)) - (mean_norm * mean_norm));

    // printf("[Inference] Normalized stats - Mean: %.3f, Std: %.3f, Min: %.3f, Max: %.3f\n",
    //        mean_norm, std_norm, min_val, max_val);

    // Run inference
    entry_border(*border_input, *border_output);

    // Process output
    float raw_output = (*border_output)[0][0];
    if(raw_output < -10.0f) raw_output = -10.0f;
    if(raw_output > 10.0f) raw_output = 10.0f;
    
    output->value = 1.0f / (1.0f + expf(-raw_output));

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