// border_model_wrapper.h
#ifndef BORDER_MODEL_WRAPPER_H
#define BORDER_MODEL_WRAPPER_H

// Only declare what other code needs to know about
void entry_border(const float tensor_input[1][4][240][240], float tensor_output[1][1]);
void clear_border_tensors(void);

void print_tensor_debug_info(const float* input_tensor, const float* output_tensor);

#endif