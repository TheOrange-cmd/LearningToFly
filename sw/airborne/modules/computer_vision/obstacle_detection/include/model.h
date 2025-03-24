// Model specifications
#ifndef MODEL_SPECS_H
#define MODEL_SPECS_H

// Input specifications (batch size, channels, height, width)
#define MODEL_INPUT_BATCH 1
#define MODEL_INPUT_CHANNELS 3
#define MODEL_INPUT_HEIGHT 60
#define MODEL_INPUT_WIDTH 60
#define DOWNSCALE_FACTOR 4

// Expected input range
#define MODEL_INPUT_MIN 0.0f
#define MODEL_INPUT_MAX 255.0f

// Output specifications
#define MODEL_OUTPUT_ROW_SIZE 1
#define MODEL_OUTPUT_COL_SIZE 3

#endif // MODEL_SPECS_H