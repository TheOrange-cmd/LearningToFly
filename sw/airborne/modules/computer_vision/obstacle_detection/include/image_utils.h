#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <stdbool.h>
#include <stdint.h>

void ensure_directory_exists(const char *path);
void save_yuv_image(const uint8_t *data, int width, int height,
                    const char *filename);
void save_rgb_image(const float *r_data, const float *g_data,
                    const float *b_data, int width, int height,
                    const char *filename);

#endif // IMAGE_UTILS_H