// border_model_wrapper.h
#ifndef BORDER_MODEL_WRAPPER_H
#define BORDER_MODEL_WRAPPER_H

// First declare the original entry function signature
void entry(const float tensor_input_1[1][3][240][240], float tensor_42[1][3]);

// Then declare our wrapper function that will call it
void entry_border(const float tensor_input_1[1][3][240][240], float tensor_42[1][3]);

void clear_border_tensors(void);
void print_tensor_debug_info(const float* input_tensor, const float* output_tensor);

#endif