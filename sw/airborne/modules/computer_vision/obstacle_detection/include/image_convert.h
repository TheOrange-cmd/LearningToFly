#ifndef IMAGE_CONVERT_H
#define IMAGE_CONVERT_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool convert_uyvy_to_yuv_downscale(const uint8_t *uyvy_data, int width,
                                   int height, float *yuv_buffer,
                                   size_t buffer_size, int downscale_factor);

bool convert_uyvy_to_yuv_crop_with_scale(const uint8_t *uyvy_data,
                                         int orig_width, int orig_height,
                                         int crop_width, int crop_height,
                                         float *yuv_buffer, size_t buffer_size,
                                         int scale_factor);

#endif // IMAGE_CONVERT_H