// border_model_wrapper.c
#include <stdio.h>
#include "border_model_wrapper.h"

#define entry entry_border
#include "models/border_detector.c"
#undef entry

void clear_border_tensors(void) {
    memset(&tu0, 0, sizeof(tu0));
    memset(&tu1, 0, sizeof(tu1));
    memset(&tu2, 0, sizeof(tu2));
}

void print_tensor_debug_info(const float* input_tensor, const float* output_tensor) {
    printf("Input tensor first few values:\n");
    for(int i = 0; i < 5; i++) {
        printf("%f ", input_tensor[i]);
    }
    printf("\n");

    printf("First few values from tu0:\n");
    for(int i = 0; i < 5; i++) {
        printf("%f ", tu0.tensor__Slice_output_0[0][0][0][i]);
    }
    printf("\n");

    if (output_tensor) {
        printf("Output tensor value: %f\n", output_tensor[0]);
    }
}