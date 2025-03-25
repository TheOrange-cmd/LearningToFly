// obstacle_model_wrapper.h
#ifndef BORDER_MODEL_WRAPPER_H
#define BORDER_MODEL_WRAPPER_H

#include "model.h"

// First declare the original entry function signature
void entry(
    const float tensor_input_1[1][3][FRONT_MODEL_INPUT_SIZE]
                              [FRONT_MODEL_INPUT_SIZE],
    float tensor_42[FRONT_MODEL_OUTPUT_ROW_SIZE][FRONT_MODEL_OUTPUT_COL_SIZE]);

// Then declare our wrapper function that will call it
void entry_obstacle(
    const float tensor_input_1[1][3][FRONT_MODEL_INPUT_SIZE]
                              [FRONT_MODEL_INPUT_SIZE],
    float tensor_42[FRONT_MODEL_OUTPUT_ROW_SIZE][FRONT_MODEL_OUTPUT_COL_SIZE]);
#endif