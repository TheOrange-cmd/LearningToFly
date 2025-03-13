#include "image_convert.h"
#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define FRONT_CAMERA_WIDTH 240
#define FRONT_CAMERA_HEIGHT 520
#define BOTTOM_CAMERA_WIDTH 240
#define BOTTOM_CAMERA_HEIGHT 240

// Lookup tables for YUV to RGB conversion
static float yuv_to_r[256][256]; // y, v -> r
static float yuv_to_g[256][256][256]; // y, u, v -> g
static float yuv_to_b[256][256]; // y, u -> b
static bool tables_initialized = false;
static int frame_counter = 0;

// Forward declarations
static bool convert_uyvy_to_rgb_common(const uint8_t* uyvy_data, 
    int width, 
    int height,
    float* rgb_buffer,
    size_t buffer_size);

static inline void store_to_rgb_channels(
    float r, float g, float b,
    float* r_channel, float* g_channel, float* b_channel,
    int idx
) {
    r_channel[idx] = r;
    g_channel[idx] = g;
    b_channel[idx] = b;
}

static inline void convert_uyvy_to_rgb_4pixels(
    const uint8_t* uyvy,
    float* r_channel, float* g_channel, float* b_channel,
    int idx
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

    // Store each pixel in channel-wise format
    store_to_rgb_channels(
        yuv_to_r[y1][v1], yuv_to_g[y1][u1][v1], yuv_to_b[y1][u1],
        r_channel, g_channel, b_channel, idx
    );
    store_to_rgb_channels(
        yuv_to_r[y2][v1], yuv_to_g[y2][u1][v1], yuv_to_b[y2][u1],
        r_channel, g_channel, b_channel, idx + 1
    );
    store_to_rgb_channels(
        yuv_to_r[y3][v2], yuv_to_g[y3][u2][v2], yuv_to_b[y3][u2],
        r_channel, g_channel, b_channel, idx + 2
    );
    store_to_rgb_channels(
        yuv_to_r[y4][v2], yuv_to_g[y4][u2][v2], yuv_to_b[y4][u2],
        r_channel, g_channel, b_channel, idx + 3
    );
}

static inline void convert_uyvy_to_rgb_2pixels(
    const uint8_t* uyvy,
    float* r_channel, float* g_channel, float* b_channel,
    int idx
) {
    uint8_t u = uyvy[0];
    uint8_t y1 = uyvy[1];
    uint8_t v = uyvy[2];
    uint8_t y2 = uyvy[3];

    store_to_rgb_channels(
        yuv_to_r[y1][v], yuv_to_g[y1][u][v], yuv_to_b[y1][u],
        r_channel, g_channel, b_channel, idx
    );
    store_to_rgb_channels(
        yuv_to_r[y2][v], yuv_to_g[y2][u][v], yuv_to_b[y2][u],
        r_channel, g_channel, b_channel, idx + 1
    );
}

size_t get_rgb_buffer_size(int width, int height) {
    // RGB data is stored in planar format (RRR...GGG...BBB)
    return MODEL_INPUT_CHANNELS * width * height * sizeof(float);
}

bool init_yuv_conversion(void) {
    if (tables_initialized) {
        return true;
    }

    printf("[Convert] Initializing YUV conversion tables\n");

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

                // Clamp to 0-255 and scale to MODEL_INPUT_MAX
                yuv_to_r[y][v] = r < 0 ? 0 : (r > 255 ? MODEL_INPUT_MAX : (r / 255.0f) * MODEL_INPUT_MAX);
                yuv_to_g[y][u][v] = g < 0 ? 0 : (g > 255 ? MODEL_INPUT_MAX : (g / 255.0f) * MODEL_INPUT_MAX);
                yuv_to_b[y][u] = b < 0 ? 0 : (b > 255 ? MODEL_INPUT_MAX : (b / 255.0f) * MODEL_INPUT_MAX);
            }
        }
    }

    // Debug check some values
    printf("[Convert] Sample YUV->RGB conversions:\n");
    printf("Y=128, U=128, V=128 -> R=%.2f G=%.2f B=%.2f\n",
        yuv_to_r[128][128],
        yuv_to_g[128][128][128],
        yuv_to_b[128][128]);
    printf("Y=255, U=128, V=128 -> R=%.2f G=%.2f B=%.2f\n",
        yuv_to_r[255][128],
        yuv_to_g[255][128][128],
        yuv_to_b[255][128]);

    tables_initialized = true;
    return true;
}

static bool convert_uyvy_to_rgb_common(const uint8_t* uyvy_data, 
    int width, 
    int height,
    float* rgb_buffer,
    size_t buffer_size) {
    
    if (!tables_initialized || !uyvy_data || !rgb_buffer || width <= 0 || height <= 0) {
        return false;
    }

    // Verify buffer size
    size_t required_size = get_rgb_buffer_size(width, height);
    if (buffer_size < required_size) {
        printf("[Convert] Buffer too small: got %zu bytes, need %zu bytes\n", 
               buffer_size, required_size);
        return false;
    }

    frame_counter++;
    
    // Debug first few bytes of input data
    printf("[Convert] First 16 bytes of UYVY data: ");
    for(int i = 0; i < 16; i++) {
        printf("%d ", uyvy_data[i]);
    }
    printf("\n");

    float* r_channel = rgb_buffer;
    float* g_channel = rgb_buffer + (height * width);
    float* b_channel = rgb_buffer + (2 * height * width);

    int total_pixels = width * height;
    int i = 0;

    for (; i < total_pixels - 3; i += 4) {
        convert_uyvy_to_rgb_4pixels(
            &uyvy_data[i * 2],
            r_channel, g_channel, b_channel,
            i
        );
    }

    for (; i < total_pixels - 1; i += 2) {
        convert_uyvy_to_rgb_2pixels(
            &uyvy_data[i * 2],
            r_channel, g_channel, b_channel,
            i
        );
    }

    // Debug output
    printf("[Convert] First few RGB values:\n");
    for(int i = 0; i < 4; i++) {
        printf("Pixel %d: R=%.2f G=%.2f B=%.2f\n",
            i,
            r_channel[i],
            g_channel[i],
            b_channel[i]
        );
    }

    // Calculate and print channel ranges
    float r_min = r_channel[0], r_max = r_channel[0];
    float g_min = g_channel[0], g_max = g_channel[0];
    float b_min = b_channel[0], b_max = b_channel[0];
    for(int i = 0; i < width * height; i++) {
        r_min = fminf(r_min, r_channel[i]);
        r_max = fmaxf(r_max, r_channel[i]);
        g_min = fminf(g_min, g_channel[i]);
        g_max = fmaxf(g_max, g_channel[i]);
        b_min = fminf(b_min, b_channel[i]);
        b_max = fmaxf(b_max, b_channel[i]);
    }
    printf("[Convert] Channel ranges - R: %.2f to %.2f, G: %.2f to %.2f, B: %.2f to %.2f\n",
        r_min, r_max, g_min, g_max, b_min, b_max);

    return true;
}

bool convert_uyvy_to_rgb_front(const uint8_t* uyvy_data, 
    int width, 
    int height,
    float* rgb_buffer,
    size_t buffer_size) {
    
    if (width != FRONT_CAMERA_WIDTH || height != FRONT_CAMERA_HEIGHT) {
        printf("[Convert] Invalid front camera dimensions: %dx%d (expected %dx%d)\n",
            width, height, FRONT_CAMERA_WIDTH, FRONT_CAMERA_HEIGHT);
        return false;
    }

    return convert_uyvy_to_rgb_common(uyvy_data, width, height, rgb_buffer, buffer_size);
}

bool convert_uyvy_to_rgb_bottom(const uint8_t* uyvy_data, 
    int width, 
    int height,
    float* rgb_buffer,
    size_t buffer_size) {
    
    if (width != BOTTOM_CAMERA_WIDTH || height != BOTTOM_CAMERA_HEIGHT) {
        printf("[Convert] Invalid bottom camera dimensions: %dx%d (expected %dx%d)\n",
            width, height, BOTTOM_CAMERA_WIDTH, BOTTOM_CAMERA_HEIGHT);
        return false;
    }

    return convert_uyvy_to_rgb_common(uyvy_data, width, height, rgb_buffer, buffer_size);
}