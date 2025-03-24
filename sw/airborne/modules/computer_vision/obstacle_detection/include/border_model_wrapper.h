// border_model_wrapper.h
#ifndef BORDER_MODEL_WRAPPER_H
#define BORDER_MODEL_WRAPPER_H

// First declare the original entry function signature
void entry(const float tensor_input[1][3][30][30], float tensor_output[1][1]);

// Then declare our wrapper function that will call it
void entry_border(const float tensor_input[1][3][30][30],
                  float tensor_output[1][1]);

// void clear_border_tensors(void);
// void print_tensor_debug_info(const float* input_tensor, const float*
// output_tensor);

#endif