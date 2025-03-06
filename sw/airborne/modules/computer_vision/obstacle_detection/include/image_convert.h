#ifndef IMAGE_CONVERT_H
#define IMAGE_CONVERT_H

#include <stdint.h>
#include <stdbool.h>

// Initialize the YUV to RGB conversion lookup tables
bool init_yuv_conversion(void);

// Convert UYVY format to RGB
// Input: UYVY data, width, height
// Output: RGB float array (allocated inside function)
// Returns: true on success, false on failure
bool convert_uyvy_to_rgb(const uint8_t* uyvy_data,
                        int width,
                        int height,
                        float** rgb_output);

#endif // IMAGE_CONVERT_H