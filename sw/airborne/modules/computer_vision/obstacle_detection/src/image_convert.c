#include "image_convert.h"
#include "model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "debug_print.h"
DEFINE_DEBUG_PRINT("IMG_CONVERT")

bool convert_uyvy_to_yuv_crop(const uint8_t* uyvy_data,
    int orig_width,
    int orig_height,
    int crop_width,
    int crop_height,
    float* yuv_buffer,
    size_t buffer_size) {
    
    // Check buffer size
    size_t required_size = crop_width * crop_height * 3 * sizeof(float);
    if (buffer_size != required_size) {
        debug_print("Invalid buffer size: got %zu, need %zu", buffer_size, required_size);
        return false;
    }

    if (!uyvy_data || !yuv_buffer) {
        debug_print("Null pointers provided");
        return false;
    }

    float* y_channel = yuv_buffer;
    float* u_channel = yuv_buffer + (crop_width * crop_height);
    float* v_channel = yuv_buffer + (2 * crop_width * crop_height);

    const float inv_255 = 1.0f / 255.0f;
    const int stride = orig_width * 2;
    
    int y_offset = (orig_height - crop_height) / 2;
    int x_offset = (orig_width - crop_width) / 2;
    const uint8_t* input = uyvy_data + (y_offset * stride) + (x_offset * 2);

    for (int y = 0; y < crop_height; y++) {
        const uint8_t* row = input + y * stride;
        const int row_offset = y * crop_width;

        for (int x = 0; x < crop_width; x += 2) {
            int out_idx = row_offset + x;
            int in_idx = x * 2;

            uint8_t u = row[in_idx];
            uint8_t y1 = row[in_idx + 1];
            uint8_t v = row[in_idx + 2];
            uint8_t y2 = row[in_idx + 3];

            float y_val1 = y1 * inv_255;
            float y_val2 = y2 * inv_255;
            y_channel[out_idx] = y_val1;
            y_channel[out_idx + 1] = y_val2;

            float u_val = u * inv_255 - 0.5f;
            float v_val = v * inv_255 - 0.5f;
            u_channel[out_idx] = u_val;
            u_channel[out_idx + 1] = u_val;
            v_channel[out_idx] = v_val;
            v_channel[out_idx + 1] = v_val;
        }
    }

    return true;
}

bool convert_uyvy_to_yuv(const uint8_t* uyvy_data,
    int width,
    int height,
    float* yuv_buffer,
    size_t buffer_size) {
    // Check buffer size
    size_t required_size = width * height * 3 * sizeof(float);
    if (buffer_size != required_size) {
        debug_print("Invalid buffer size: got %zu, need %zu", buffer_size, required_size);
        return false;
    }

    if (!uyvy_data || !yuv_buffer) {
        debug_print("Null pointers provided");
        return false;
    }

    float* y_channel = yuv_buffer;
    float* u_channel = yuv_buffer + (width * height);
    float* v_channel = yuv_buffer + (2 * width * height);

    const float inv_255 = 1.0f / 255.0f;
    const int stride = width * 2;  // Bytes per row in UYVY format

    for (int y = 0; y < height; y++) {
        const uint8_t* row = uyvy_data + y * stride;

        for (int x = 0; x < width; x++) {
            int in_idx = x * 2;

            // Extract U, Y, V values for each pixel
            uint8_t u = row[in_idx];
            uint8_t y_val = row[in_idx + 1];
            uint8_t v = row[in_idx + 2];

            // Convert to float and normalize
            y_channel[y * width + x] = y_val * inv_255;
            u_channel[y * width + x] = u * inv_255 - 0.5f;
            v_channel[y * width + x] = v * inv_255 - 0.5f;
        }
    }

    return true;
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
bool convert_uyvy_to_yuv_downscale(const uint8_t* uyvy_data,
    int width,
    int height,
    float* yuv_buffer,
    size_t buffer_size,
    int downscale_factor) {
    switch (downscale_factor) {
    case 1:
        return convert_uyvy_to_yuv(uyvy_data, width, height, yuv_buffer, buffer_size);
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