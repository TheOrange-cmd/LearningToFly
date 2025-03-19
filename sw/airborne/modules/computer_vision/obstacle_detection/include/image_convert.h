#ifndef IMAGE_CONVERT_H
#define IMAGE_CONVERT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h> 
#include <math.h>    

// Initialize the YUV to RGB conversion lookup tables
bool init_yuv_conversion(void);

// Get required buffer size for RGB conversion
size_t get_rgb_buffer_size(int width, int height);

// Convert functions that take pre-allocated buffer
bool convert_uyvy_to_rgb_front(const uint8_t* uyvy_data, 
    int width, 
    int height, 
    float* rgb_buffer,
    size_t buffer_size);

bool convert_uyvy_to_rgb_bottom(const uint8_t* uyvy_data, 
    int width, 
    int height, 
    float* rgb_buffer,
    size_t buffer_size);

bool convert_uyvy_to_yuv_bottom(const uint8_t* uyvy_data, 
    int width, 
    int height,
    float* yuv_buffer,
    size_t buffer_size, int downscale_factor);

#endif // IMAGE_CONVERT_H