#include "image_convert.h"
#include <stdlib.h>
#include <string.h>

// Lookup tables for YUV to RGB conversion
static float yuv_to_r[256][256]; // y, v -> r
static float yuv_to_g[256][256][256]; // y, u, v -> g
static float yuv_to_b[256][256]; // y, u -> b
static bool tables_initialized = false;

bool init_yuv_conversion(void) {
    if (tables_initialized) {
        return true;
    }

    // Initialize conversion tables
    for (int y = 0; y < 256; y++) {
        for (int u = 0; u < 256; u++) {
            float uf = u - 128;
            for (int v = 0; v < 256; v++) {
                float vf = v - 128;
                
                // Standard YUV to RGB conversion
                float r = (y + 1.402f * vf);
                float g = (y - 0.344f * uf - 0.714f * vf);
                float b = (y + 1.772f * uf);

                // Clamp and normalize to 0-1
                yuv_to_r[y][v] = r < 0 ? 0 : (r > 255 ? 1 : r / 255.0f);
                yuv_to_g[y][u][v] = g < 0 ? 0 : (g > 255 ? 1 : g / 255.0f);
                yuv_to_b[y][u] = b < 0 ? 0 : (b > 255 ? 1 : b / 255.0f);
            }
        }
    }

    tables_initialized = true;
    return true;
}

static inline void convert_uyvy_to_rgb_4pixels(
    const uint8_t* uyvy,
    float* rgb,
    int rgb_offset
) {
    // First pair
    uint8_t u1 = uyvy[0];
    uint8_t y1 = uyvy[1];
    uint8_t v1 = uyvy[2];
    uint8_t y2 = uyvy[3];

    // Second pair
    uint8_t u2 = uyvy[4];
    uint8_t y3 = uyvy[5];
    uint8_t v2 = uyvy[6];
    uint8_t y4 = uyvy[7];

    // Convert first pair
    rgb[rgb_offset] = yuv_to_r[y1][v1];
    rgb[rgb_offset + 1] = yuv_to_g[y1][u1][v1];
    rgb[rgb_offset + 2] = yuv_to_b[y1][u1];

    rgb[rgb_offset + 3] = yuv_to_r[y2][v1];
    rgb[rgb_offset + 4] = yuv_to_g[y2][u1][v1];
    rgb[rgb_offset + 5] = yuv_to_b[y2][u1];

    // Convert second pair
    rgb[rgb_offset + 6] = yuv_to_r[y3][v2];
    rgb[rgb_offset + 7] = yuv_to_g[y3][u2][v2];
    rgb[rgb_offset + 8] = yuv_to_b[y3][u2];

    rgb[rgb_offset + 9] = yuv_to_r[y4][v2];
    rgb[rgb_offset + 10] = yuv_to_g[y4][u2][v2];
    rgb[rgb_offset + 11] = yuv_to_b[y4][u2];
}

static inline void convert_uyvy_to_rgb_2pixels(
    const uint8_t* uyvy,
    float* rgb,
    int rgb_offset
) {
    uint8_t u = uyvy[0];
    uint8_t y1 = uyvy[1];
    uint8_t v = uyvy[2];
    uint8_t y2 = uyvy[3];

    rgb[rgb_offset] = yuv_to_r[y1][v];
    rgb[rgb_offset + 1] = yuv_to_g[y1][u][v];
    rgb[rgb_offset + 2] = yuv_to_b[y1][u];

    rgb[rgb_offset + 3] = yuv_to_r[y2][v];
    rgb[rgb_offset + 4] = yuv_to_g[y2][u][v];
    rgb[rgb_offset + 5] = yuv_to_b[y2][u];
}

bool convert_uyvy_to_rgb(const uint8_t* uyvy_data, 
                        int width, 
                        int height, 
                        float** rgb_output) {
    if (!tables_initialized || !uyvy_data || width <= 0 || height <= 0) {
        return false;
    }

    // Allocate RGB output buffer
    size_t rgb_size = width * height * 3 * sizeof(float);
    float* rgb = malloc(rgb_size);
    if (!rgb) {
        return false;
    }

    int total_pixels = width * height;
    int i = 0;
    
    // Process 4 pixels at a time
    for (; i < total_pixels - 3; i += 4) {
        convert_uyvy_to_rgb_4pixels(
            &uyvy_data[i * 2],
            rgb,
            i * 3
        );
    }

    // Handle remaining pixels in pairs
    for (; i < total_pixels - 1; i += 2) {
        convert_uyvy_to_rgb_2pixels(
            &uyvy_data[i * 2],
            rgb,
            i * 3
        );
    }

    *rgb_output = rgb;
    return true;
}