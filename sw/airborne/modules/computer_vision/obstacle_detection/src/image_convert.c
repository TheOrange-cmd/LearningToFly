#include "image_convert.h"
#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "debug_print.h"
DEFINE_DEBUG_PRINT("IMG_CONVERT")

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

    // BT.601 full range coefficients
    const float kr = 0.299f;
    const float kg = 0.587f;
    const float kb = 0.114f;

    // Initialize conversion tables
    for (int y = 0; y < 256; y++) {
        float yf = y / 255.0f;  // Normalize to [0,1]
        for (int u = 0; u < 256; u++) {
            float uf = (u / 255.0f - 0.5f);  // Center at 0
            for (int v = 0; v < 256; v++) {
                float vf = (v / 255.0f - 0.5f);  // Center at 0
                
                // BT.601 full range conversion
                float r = yf + (2.0f * (1.0f - kr)) * vf;
                float g = yf - (2.0f * (1.0f - kb) * kb/kg) * uf - (2.0f * (1.0f - kr) * kr/kg) * vf;
                float b = yf + (2.0f * (1.0f - kb)) * uf;
    
                // Clamp to [0,1]
                yuv_to_r[y][v] = r < 0 ? 0 : (r > 1 ? 1 : r);
                yuv_to_g[y][u][v] = g < 0 ? 0 : (g > 1 ? 1 : g);
                yuv_to_b[y][u] = b < 0 ? 0 : (b > 1 ? 1 : b);
            }
        }
    }
    

    // Debug check some values
    // printf("[Convert] Sample YUV->RGB conversions:\n");
    // printf("Y=128, U=128, V=128 -> R=%.2f G=%.2f B=%.2f\n",
    //     yuv_to_r[128][128],
    //     yuv_to_g[128][128][128],
    //     yuv_to_b[128][128]);
    // printf("Y=255, U=128, V=128 -> R=%.2f G=%.2f B=%.2f\n",
    //     yuv_to_r[255][128],
    //     yuv_to_g[255][128][128],
    //     yuv_to_b[255][128]);

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
    // printf("[Convert] First 16 bytes of UYVY data: ");
    // for(int i = 0; i < 16; i++) {
    //     printf("%d ", uyvy_data[i]);
    // }
    // printf("\n");

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
    // printf("[Convert] Converting front camera frame to RGB\n");
    return convert_uyvy_to_rgb_common(uyvy_data, width, height, rgb_buffer, buffer_size);
}

bool convert_uyvy_to_rgb_bottom(const uint8_t* uyvy_data, 
    int width, 
    int height,
    float* rgb_buffer,
    size_t buffer_size) {
    
    if (width != BOTTOM_CAMERA_WIDTH || height != BOTTOM_CAMERA_HEIGHT) {
        debug_print("Invalid dimensions: got %dx%d, expected %dx%d",
            width, height, BOTTOM_CAMERA_WIDTH, BOTTOM_CAMERA_HEIGHT);
        return false;
    }

    size_t required_size = (size_t)width * height * 3 * sizeof(float);
    if (buffer_size != required_size) {
        debug_print("Invalid buffer size: got %zu, need %zu", buffer_size, required_size);
        return false;
    }

    if (!uyvy_data || !rgb_buffer) {
        debug_print("Null pointers provided");
        return false;
    }

    return convert_uyvy_to_rgb_common(uyvy_data, width, height, rgb_buffer, buffer_size);
}

// Highly optimized version for 2x downscaling
bool convert_uyvy_to_yuv_downscale2(const uint8_t* uyvy_data,
    int width,
    int height,
    float* yuv_buffer,
    size_t buffer_size) {
    // Fixed 2x downscaling: original (width x height) -> (width/2 x height/2)
    int out_width = width / 2;
    int out_height = height / 2;

    // Check dimensions are even
    if (width % 2 != 0 || height % 2 != 0) {
        debug_print("Input dimensions must be divisible by 2");
        return false;
    }

    // Check buffer size
    size_t required_size = out_width * out_height * 3 * sizeof(float);
    if (buffer_size != required_size) {
        debug_print("Invalid buffer size: got %zu, need %zu", buffer_size, required_size);
        return false;
    }

    if (!uyvy_data || !yuv_buffer) {
        debug_print("Null pointers provided");
        return false;
    }

    float* y_channel = yuv_buffer;
    float* u_channel = yuv_buffer + (out_width * out_height);
    float* v_channel = yuv_buffer + (2 * out_width * out_height);

    const float inv_255 = 1.0f / 255.0f;
    const int stride = width * 2;  // Bytes per row

    // Process 2x2 blocks - hardcoded for maximum efficiency
    for (int out_y = 0; out_y < out_height; out_y++) {
        int in_y = out_y * 2;
        int out_idx_base = out_y * out_width;
        const uint8_t* row1 = uyvy_data + in_y * stride;
        const uint8_t* row2 = row1 + stride;

        for (int out_x = 0; out_x < out_width; out_x++) {
            int in_x = out_x * 2;
            int out_idx = out_idx_base + out_x;
            int in_idx = in_x * 2;

            // Get 4 bytes from top row (2 pixels)
            uint8_t u1 = row1[in_idx];
            uint8_t y1 = row1[in_idx + 1];
            uint8_t v1 = row1[in_idx + 2];
            uint8_t y2 = row1[in_idx + 3];

            // Get 4 bytes from bottom row (2 pixels)
            uint8_t u2 = row2[in_idx];
            uint8_t y3 = row2[in_idx + 1];
            uint8_t v2 = row2[in_idx + 2];
            uint8_t y4 = row2[in_idx + 3];

            // Average 4 Y values at once
            y_channel[out_idx] = (y1 + y2 + y3 + y4) * 0.25f * inv_255;

            // Average U and V values
            u_channel[out_idx] = (u1 + u2) * 0.5f * inv_255 - 0.5f;
            v_channel[out_idx] = (v1 + v2) * 0.5f * inv_255 - 0.5f;
        }
    }
    return true;
}

// Highly optimized version for 4x downscaling
bool convert_uyvy_to_yuv_downscale4(const uint8_t* uyvy_data,
     int width,
     int height,
     float* yuv_buffer,
     size_t buffer_size) {
    // Fixed 4x downscaling: original (width x height) -> (width/4 x height/4)
    int out_width = width / 4;
    int out_height = height / 4;

    // Check dimensions are divisible by 4
    if (width % 4 != 0 || height % 4 != 0) {
        debug_print("Input dimensions must be divisible by 4");
        return false;
    }

    // Check buffer size
    size_t required_size = out_width * out_height * 3 * sizeof(float);
    if (buffer_size != required_size) {
        debug_print("Invalid buffer size: got %zu, need %zu", buffer_size, required_size);
        return false;
    }

    if (!uyvy_data || !yuv_buffer) {
        debug_print("Null pointers provided");
        return false;
    }

    float* y_channel = yuv_buffer;
    float* u_channel = yuv_buffer + (out_width * out_height);
    float* v_channel = yuv_buffer + (2 * out_width * out_height);

    const float inv_255 = 1.0f / 255.0f;
    const float one_sixteenth = 1.0f / 16.0f;
    const float one_fourth = 0.25f;
    const int stride = width * 2;  // Bytes per row in UYVY format

    // For 4x4 block, we have 8 UYVY groups (16 pixels)
    for (int out_y = 0; out_y < out_height; out_y++) {
        int in_y_base = out_y * 4;

        for (int out_x = 0; out_x < out_width; out_x++) {
            int in_x_base = out_x * 4;
            int out_idx = out_y * out_width + out_x;

            // Accumulators
            float y_sum = 0.0f;
            float u_sum = 0.0f;
            float v_sum = 0.0f;

            // Process 4 rows
            for (int y_offset = 0; y_offset < 4; y_offset++) {
                int in_y = in_y_base + y_offset;
                const uint8_t* row = uyvy_data + in_y * stride;

                // Process 2 UYVY groups per row (4 pixels)
                for (int x_group = 0; x_group < 2; x_group++) {
                    int in_idx = (in_x_base + x_group * 2) * 2;

                    // Extract values for 2 pixels in this group
                    uint8_t u = row[in_idx];
                    uint8_t y1 = row[in_idx + 1];
                    uint8_t v = row[in_idx + 2];
                    uint8_t y2 = row[in_idx + 3];

                    // Accumulate values
                    y_sum += y1 + y2;
                    u_sum += u;
                    v_sum += v;
                }
            }

        // Store averaged values
        y_channel[out_idx] = y_sum * one_sixteenth * inv_255;    // Average of 16 Y values
        u_channel[out_idx] = u_sum * one_fourth * inv_255 - 0.5f; // Average of 8 U values
        v_channel[out_idx] = v_sum * one_fourth * inv_255 - 0.5f; // Average of 8 V values
        }
    }

    return true;
}

// Highly optimized version for 8x downscaling
bool convert_uyvy_to_yuv_downscale8(const uint8_t* uyvy_data,
    int width,
    int height,
    float* yuv_buffer,
    size_t buffer_size) {
   // Fixed 8x downscaling: original (width x height) -> (width/8 x height/8)
   int out_width = width / 8;
   int out_height = height / 8;

   // Check dimensions are divisible by 8
   if (width % 8 != 0 || height % 8 != 0) {
       debug_print("Input dimensions must be divisible by 8");
       return false;
   }

   // Check buffer size
   size_t required_size = out_width * out_height * 3 * sizeof(float);
   if (buffer_size != required_size) {
       debug_print("Invalid buffer size: got %zu, need %zu", buffer_size, required_size);
       return false;
   }

   if (!uyvy_data || !yuv_buffer) {
       debug_print("Null pointers provided");
       return false;
   }

   float* y_channel = yuv_buffer;
   float* u_channel = yuv_buffer + (out_width * out_height);
   float* v_channel = yuv_buffer + (2 * out_width * out_height);

   const float inv_255 = 1.0f / 255.0f;
   const float one_sixtyfourth = 1.0f / 64.0f;
   const float one_sixteenth = 1.0f / 16.0f;
   const int stride = width * 2;  // Bytes per row in UYVY format

   // For 8x8 block, we have 32 UYVY groups (64 pixels)
   for (int out_y = 0; out_y < out_height; out_y++) {
       int in_y_base = out_y * 8;

       for (int out_x = 0; out_x < out_width; out_x++) {
           int in_x_base = out_x * 8;
           int out_idx = out_y * out_width + out_x;

           // Accumulators
           float y_sum = 0.0f;
           float u_sum = 0.0f;
           float v_sum = 0.0f;

           // Process 8 rows
           for (int y_offset = 0; y_offset < 8; y_offset++) {
               int in_y = in_y_base + y_offset;
               const uint8_t* row = uyvy_data + in_y * stride;

               // Process 4 UYVY groups per row (8 pixels)
               for (int x_group = 0; x_group < 4; x_group++) {
                   int in_idx = (in_x_base + x_group * 2) * 2;

                   // Extract values for 2 pixels in this group
                   uint8_t u = row[in_idx];
                   uint8_t y1 = row[in_idx + 1];
                   uint8_t v = row[in_idx + 2];
                   uint8_t y2 = row[in_idx + 3];

                   // Accumulate values
                   y_sum += y1 + y2;
                   u_sum += u;
                   v_sum += v;
               }
           }

           // Store averaged values
           y_channel[out_idx] = y_sum * one_sixtyfourth * inv_255;    // Average of 64 Y values
           u_channel[out_idx] = u_sum * one_sixteenth * inv_255 - 0.5f; // Average of 32 U values
           v_channel[out_idx] = v_sum * one_sixteenth * inv_255 - 0.5f; // Average of 32 V values
       }
   }

   return true;
}

// Wrapper function that calls the appropriate specialized function
bool convert_uyvy_to_yuv_bottom(const uint8_t* uyvy_data,
    int width,
    int height,
    float* yuv_buffer,
    size_t buffer_size,
    int downscale_factor) {
    switch (downscale_factor) {
    case 2:
        return convert_uyvy_to_yuv_downscale2(uyvy_data, width, height, yuv_buffer, buffer_size);
    case 4:
        return convert_uyvy_to_yuv_downscale4(uyvy_data, width, height, yuv_buffer, buffer_size);
    case 8:
        return convert_uyvy_to_yuv_downscale8(uyvy_data, width, height, yuv_buffer, buffer_size);
    default:
        debug_print("Unsupported downscale factor: %d", downscale_factor);
        return false;
    }
}