/**
 *
 * Copyright (C) 2025 Daniel Rugge <d.j.rugge@student.tudelft.nl>
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */
/**
 * @file
 * sw/airborne/modules/computer_vision/obstacle_detection/src/image_utils.c
 * @brief Functions used for image preprocessing for model inference,
 * prioritizing speed over avoiding code duplication.
 *
 * Each function is specialized for a specific combination of UYVY-YUV
 * conversion, cropping, and downscaling that is expected by the neural
 * networks. Multiple downscaling factors are included to allow testing the
 * effect of different resolutions on inference speed after testing accuracy in
 * the training code to find the best trade-off between speed and accuracy.
 *
 * @note Developement assisted by Claude Sonnet 3.5
 */

#include "image_convert.h"
#include "debug_print.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
DEFINE_DEBUG_PRINT("IMG_CONVERT")

// Forward declarations
bool convert_uyvy_to_yuv_crop(const uint8_t *uyvy_data, int orig_width,
                              int orig_height, int crop_width, int crop_height,
                              float *yuv_buffer, size_t buffer_size);

bool convert_uyvy_to_yuv_crop_downscale2(const uint8_t *uyvy_data,
                                         int orig_width, int orig_height,
                                         int crop_width, int crop_height,
                                         float *yuv_buffer, size_t buffer_size);

bool convert_uyvy_to_yuv_crop_downscale4(const uint8_t *uyvy_data,
                                         int orig_width, int orig_height,
                                         int crop_width, int crop_height,
                                         float *yuv_buffer, size_t buffer_size);

bool convert_uyvy_to_yuv_crop_with_scale(const uint8_t *uyvy_data,
                                         int orig_width, int orig_height,
                                         int crop_width, int crop_height,
                                         float *yuv_buffer, size_t buffer_size,
                                         int scale_factor);

bool convert_uyvy_to_yuv_downscale2(const uint8_t *uyvy_data, int orig_width,
                                    int orig_height, float *yuv_buffer,
                                    size_t buffer_size);

bool convert_uyvy_to_yuv_downscale4(const uint8_t *uyvy_data, int orig_width,
                                    int orig_height, float *yuv_buffer,
                                    size_t buffer_size);

bool convert_uyvy_to_yuv_downscale8(const uint8_t *uyvy_data, int orig_width,
                                    int orig_height, float *yuv_buffer,
                                    size_t buffer_size);

bool convert_uyvy_to_yuv(const uint8_t *uyvy_data, int width, int height,
                         float *yuv_buffer, size_t buffer_size);

/**
 * @brief Convert UYVY to YUV with center cropping (no downscaling)
 *
 * Specialized for front camera input. Converts UYVY422 to planar YUV420 format,
 * cropping to specified dimensions from image center. Validates buffer size.
 *
 * @param uyvy_data Input UYVY422 image data
 * @param orig_width Original image width
 * @param orig_height Original image height
 * @param crop_width Width of cropped region
 * @param crop_height Height of cropped region
 * @param yuv_buffer Output buffer for planar YUV data (Y + U + V)
 * @param buffer_size Size of output buffer (must be
 * 3*crop_width*crop_height*sizeof(float))
 * @return true Conversion succeeded
 * @return false Invalid parameters or buffer size mismatch
 */
bool convert_uyvy_to_yuv_crop(const uint8_t *uyvy_data, int orig_width,
                              int orig_height, int crop_width, int crop_height,
                              float *yuv_buffer, size_t buffer_size) {

  // Check buffer size
  size_t required_size = crop_width * crop_height * 3 * sizeof(float);
  if (buffer_size != required_size) {
    debug_print("Invalid buffer size: got %zu, need %zu", buffer_size,
                required_size);
    return false;
  }

  if (!uyvy_data || !yuv_buffer) {
    debug_print("Null pointers provided");
    return false;
  }

  float *y_channel = yuv_buffer;
  float *u_channel = yuv_buffer + (crop_width * crop_height);
  float *v_channel = yuv_buffer + (2 * crop_width * crop_height);

  const float inv_255 = 1.0f / 255.0f;
  const int stride = orig_width * 2;

  int y_offset = (orig_height - crop_height) / 2;
  int x_offset = (orig_width - crop_width) / 2;
  const uint8_t *input = uyvy_data + (y_offset * stride) + (x_offset * 2);

  for (int y = 0; y < crop_height; y++) {
    const uint8_t *row = input + y * stride;
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

/**
 * @brief Convert UYVY to YUV with 2x downscaling and center cropping
 *
 * Processes 2x2 blocks with averaging. Cropped dimensions must be divisible
 * by 2.
 *
 * @param uyvy_data Input UYVY422 image data
 * @param orig_width Original image width
 * @param orig_height Original image height
 * @param crop_width Width of cropped region (must be even)
 * @param crop_height Height of cropped region (must be even)
 * @param yuv_buffer Output buffer (3*(crop/2)^2 floats)
 * @param buffer_size Required buffer size
 * @return true Conversion succeeded
 * @return false Invalid parameters or dimension mismatch
 */
bool convert_uyvy_to_yuv_crop_downscale2(const uint8_t *uyvy_data,
                                         int orig_width, int orig_height,
                                         int crop_width, int crop_height,
                                         float *yuv_buffer,
                                         size_t buffer_size) {

  // Fixed downscaling factor of 2
  const int scale = 2;
  const int out_width = crop_width / scale;
  const int out_height = crop_height / scale;

  // Check if crop dimensions are divisible by scale factor
  if (crop_width % scale != 0 || crop_height % scale != 0) {
    debug_print("Crop dimensions must be divisible by %d", scale);
    return false;
  }

  // Check buffer size
  size_t required_size = out_width * out_height * 3 * sizeof(float);
  if (buffer_size != required_size) {
    debug_print("Invalid buffer size: got %zu, need %zu", buffer_size,
                required_size);
    return false;
  }

  if (!uyvy_data || !yuv_buffer) {
    debug_print("Null pointers provided");
    return false;
  }

  float *y_channel = yuv_buffer;
  float *u_channel = yuv_buffer + (out_width * out_height);
  float *v_channel = yuv_buffer + (2 * out_width * out_height);

  const float inv_255 = 1.0f / 255.0f;
  const float one_fourth = 0.25f;    // 1/4 for averaging 4 pixels
  const int stride = orig_width * 2; // Bytes per row in UYVY format

  // Calculate crop offsets
  int y_offset = (orig_height - crop_height) / 2;
  int x_offset = (orig_width - crop_width) / 2;
  const uint8_t *crop_start = uyvy_data + (y_offset * stride) + (x_offset * 2);

  // Process each 2x2 block in the cropped area
  for (int out_y = 0; out_y < out_height; out_y++) {
    int in_y_base = out_y * scale;

    for (int out_x = 0; out_x < out_width; out_x++) {
      int in_x_base = out_x * scale;
      int out_idx = out_y * out_width + out_x;

      // Accumulators
      float y_sum = 0.0f;
      float u_sum = 0.0f;
      float v_sum = 0.0f;

      // Process 2 rows
      for (int y_offset = 0; y_offset < scale; y_offset++) {
        const uint8_t *row = crop_start + (in_y_base + y_offset) * stride;

        // Process 1 UYVY group per row (2 pixels)
        int in_idx = in_x_base * 2;

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

      // Store averaged values
      y_channel[out_idx] =
          y_sum * one_fourth * inv_255; // Average of 4 Y values
      u_channel[out_idx] =
          u_sum * 0.5f * inv_255 - 0.5f; // Average of 2 U values
      v_channel[out_idx] =
          v_sum * 0.5f * inv_255 - 0.5f; // Average of 2 V values
    }
  }

  return true;
}

/**
 * @brief Convert UYVY to YUV with 4x downscaling and center cropping
 *
 * Processes 4x4 blocks with Y channel averaging and UV subsampling.
 * Cropped dimensions must be divisible by 4.
 *
 * @param uyvy_data Input UYVY422 image data
 * @param orig_width Original image width
 * @param orig_height Original image height
 * @param crop_width Width of cropped region (must be divisible by 4)
 * @param crop_height Height of cropped region (must be divisible by 4)
 * @param yuv_buffer Output buffer (3*(crop/4)^2 floats)
 * @param buffer_size Required buffer size
 * @return true Conversion succeeded
 * @return false Invalid parameters or dimension mismatch
 */
bool convert_uyvy_to_yuv_crop_downscale4(const uint8_t *uyvy_data,
                                         int orig_width, int orig_height,
                                         int crop_width, int crop_height,
                                         float *yuv_buffer,
                                         size_t buffer_size) {

  const int scale = 4;
  const int out_width = crop_width / scale;
  const int out_height = crop_height / scale;

  // Check if crop dimensions are divisible by scale factor
  if (crop_width % scale != 0 || crop_height % scale != 0) {
    debug_print("Crop dimensions must be divisible by %d", scale);
    return false;
  }

  // Check buffer size
  size_t required_size = out_width * out_height * 3 * sizeof(float);
  if (buffer_size != required_size) {
    debug_print("Invalid buffer size: got %zu, need %zu", buffer_size,
                required_size);
    return false;
  }

  if (!uyvy_data || !yuv_buffer) {
    debug_print("Null pointers provided");
    return false;
  }

  float *y_channel = yuv_buffer;
  float *u_channel = yuv_buffer + (out_width * out_height);
  float *v_channel = yuv_buffer + (2 * out_width * out_height);

  const float inv_255 = 1.0f / 255.0f;
  const int stride = orig_width * 2;

  int y_offset = (orig_height - crop_height) / 2;
  int x_offset = (orig_width - crop_width) / 2;
  const uint8_t *crop_start = uyvy_data + (y_offset * stride) + (x_offset * 2);

  // Process blocks with proper averaging
  for (int out_y = 0; out_y < out_height; out_y++) {
    for (int out_x = 0; out_x < out_width; out_x++) {
      int out_idx = out_y * out_width + out_x;
      float y_sum = 0.0f;
      float u_sum = 0.0f;
      float v_sum = 0.0f;
      int y_count = 0;
      int uv_count = 0;

      // Process 4x4 block
      for (int dy = 0; dy < scale; dy++) {
        const uint8_t *row = crop_start + ((out_y * scale + dy) * stride);

        for (int dx = 0; dx < scale; dx++) {
          int x = out_x * scale + dx;
          int pixel_idx = x * 2;

          // For Y channel
          y_sum += row[pixel_idx + 1];
          y_count++;

          // For U/V channels (sampled at half rate)
          if ((dx % 2 == 0) && (dy % 2 == 0)) {
            u_sum += row[pixel_idx];
            v_sum += row[pixel_idx + 2];
            uv_count++;
          }
        }
      }

      // Normalize exactly as in Python
      y_channel[out_idx] = (y_sum / y_count) * inv_255;
      u_channel[out_idx] = (u_sum / uv_count) * inv_255 - 0.5f;
      v_channel[out_idx] = (v_sum / uv_count) * inv_255 - 0.5f;
    }
  }

  return true;
}

/**
 * @brief Select cropped conversion based on scale factor
 *
 * Dispatches to appropriate conversion function for scale factors 1, 2, or 4.
 *
 * @param scale_factor Downscaling factor (1, 2, or 4)
 * @return true Supported scale factor and conversion succeeded
 * @return false Unsupported scale factor or conversion error
 */
bool convert_uyvy_to_yuv_crop_with_scale(const uint8_t *uyvy_data,
                                         int orig_width, int orig_height,
                                         int crop_width, int crop_height,
                                         float *yuv_buffer, size_t buffer_size,
                                         int scale_factor) {

  switch (scale_factor) {
  case 1:
    return convert_uyvy_to_yuv_crop(uyvy_data, orig_width, orig_height,
                                    crop_width, crop_height, yuv_buffer,
                                    buffer_size);
  case 2:
    return convert_uyvy_to_yuv_crop_downscale2(
        uyvy_data, orig_width, orig_height, crop_width, crop_height, yuv_buffer,
        buffer_size);
  case 4:
    return convert_uyvy_to_yuv_crop_downscale4(
        uyvy_data, orig_width, orig_height, crop_width, crop_height, yuv_buffer,
        buffer_size);
  default:
    debug_print(
        "Unsupported scale factor: %d. Supported values are 1, 2, and 4.",
        scale_factor);
    return false;
  }
}

/**
 * @brief Convert full UYVY image to YUV without downscaling
 *
 * Direct conversion preserving original resolution. Output is planar YUV floats
 * normalized to [0,1] for Y and [-0.5,0.5] for UV.
 *
 * @param width Input image width
 * @param height Input image height
 * @param buffer_size Must be 3*width*height*sizeof(float)
 * @return true Conversion succeeded
 * @return false Buffer size mismatch or null pointers
 */
bool convert_uyvy_to_yuv(const uint8_t *uyvy_data, int width, int height,
                         float *yuv_buffer, size_t buffer_size) {
  // Check buffer size
  size_t required_size = width * height * 3 * sizeof(float);
  if (buffer_size != required_size) {
    debug_print("Invalid buffer size: got %zu, need %zu", buffer_size,
                required_size);
    return false;
  }

  if (!uyvy_data || !yuv_buffer) {
    debug_print("Null pointers provided");
    return false;
  }

  float *y_channel = yuv_buffer;
  float *u_channel = yuv_buffer + (width * height);
  float *v_channel = yuv_buffer + (2 * width * height);

  const float inv_255 = 1.0f / 255.0f;
  const int stride = width * 2; // Bytes per row in UYVY format

  for (int y = 0; y < height; y++) {
    const uint8_t *row = uyvy_data + y * stride;

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

/**
 * @brief Downscale UYVY image by factor of 2 and convert to YUV
 *
 * Processes 2x2 blocks with 8 Y samples and 4 UV samples per output pixel.
 *
 * @param width Must be divisible by 2
 * @param height Must be divisible by 2
 * @return true Conversion succeeded
 * @return false Odd dimensions or buffer mismatch
 */
bool convert_uyvy_to_yuv_downscale2(const uint8_t *uyvy_data, int width,
                                    int height, float *yuv_buffer,
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
    debug_print("Invalid buffer size: got %zu, need %zu", buffer_size,
                required_size);
    return false;
  }

  if (!uyvy_data || !yuv_buffer) {
    debug_print("Null pointers provided");
    return false;
  }

  float *y_channel = yuv_buffer;
  float *u_channel = yuv_buffer + (out_width * out_height);
  float *v_channel = yuv_buffer + (2 * out_width * out_height);

  const float inv_255 = 1.0f / 255.0f;
  const int stride = width * 2; // Bytes per row

  // Process 2x2 blocks - hardcoded for maximum efficiency
  for (int out_y = 0; out_y < out_height; out_y++) {
    int in_y = out_y * 2;
    int out_idx_base = out_y * out_width;
    const uint8_t *row1 = uyvy_data + in_y * stride;
    const uint8_t *row2 = row1 + stride;

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
/**
 * @brief Downscale UYVY image by factor of 4 and convert to YUV.
 *
 * Processes 4x4 blocks with 16 Y samples and 8 UV samples per output pixel.
 *
 * @param width Must be divisible by 4
 * @param height Must be divisible by 4
 * @return true Conversion succeeded
 * @return false Invalid dimensions or buffer size
 */
bool convert_uyvy_to_yuv_downscale4(const uint8_t *uyvy_data, int width,
                                    int height, float *yuv_buffer,
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
    debug_print("Invalid buffer size: got %zu, need %zu", buffer_size,
                required_size);
    return false;
  }

  if (!uyvy_data || !yuv_buffer) {
    debug_print("Null pointers provided");
    return false;
  }

  float *y_channel = yuv_buffer;
  float *u_channel = yuv_buffer + (out_width * out_height);
  float *v_channel = yuv_buffer + (2 * out_width * out_height);

  const float inv_255 = 1.0f / 255.0f;
  const float one_sixteenth = 1.0f / 16.0f;
  const float one_fourth = 0.25f;
  const int stride = width * 2; // Bytes per row in UYVY format

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
        const uint8_t *row = uyvy_data + in_y * stride;

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
      y_channel[out_idx] =
          y_sum * one_sixteenth * inv_255; // Average of 16 Y values
      u_channel[out_idx] =
          u_sum * one_fourth * inv_255 - 0.5f; // Average of 8 U values
      v_channel[out_idx] =
          v_sum * one_fourth * inv_255 - 0.5f; // Average of 8 V values
    }
  }

  return true;
}

/**
 * @brief Downscale UYVY image by factor of 8
 *
 * Processes 8x8 blocks with 64 Y samples and 32 UV samples per output pixel.
 *
 * @param width Must be divisible by 8
 * @param height Must be divisible by 8
 * @return true Conversion succeeded
 * @return false Invalid dimensions or buffer size
 */
bool convert_uyvy_to_yuv_downscale8(const uint8_t *uyvy_data, int width,
                                    int height, float *yuv_buffer,
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
    debug_print("Invalid buffer size: got %zu, need %zu", buffer_size,
                required_size);
    return false;
  }

  if (!uyvy_data || !yuv_buffer) {
    debug_print("Null pointers provided");
    return false;
  }

  float *y_channel = yuv_buffer;
  float *u_channel = yuv_buffer + (out_width * out_height);
  float *v_channel = yuv_buffer + (2 * out_width * out_height);

  const float inv_255 = 1.0f / 255.0f;
  const float one_sixtyfourth = 1.0f / 64.0f;
  const float one_sixteenth = 1.0f / 16.0f;
  const int stride = width * 2; // Bytes per row in UYVY format

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
        const uint8_t *row = uyvy_data + in_y * stride;

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
      y_channel[out_idx] =
          y_sum * one_sixtyfourth * inv_255; // Average of 64 Y values
      u_channel[out_idx] =
          u_sum * one_sixteenth * inv_255 - 0.5f; // Average of 32 U values
      v_channel[out_idx] =
          v_sum * one_sixteenth * inv_255 - 0.5f; // Average of 32 V values
    }
  }

  return true;
}

/**
 * @brief Select downscaling conversion wrapper
 *
 * Routes to conversion functions based on downscale factor (1, 2, 4, 8)
 *
 * @param downscale_factor Supported factors: 1, 2, 4, 8
 * @return true Successful dispatch and conversion
 * @return false Unsupported factor or conversion error
 */
bool convert_uyvy_to_yuv_downscale(const uint8_t *uyvy_data, int width,
                                   int height, float *yuv_buffer,
                                   size_t buffer_size, int downscale_factor) {
  switch (downscale_factor) {
  case 1:
    return convert_uyvy_to_yuv(uyvy_data, width, height, yuv_buffer,
                               buffer_size);
  case 2:
    return convert_uyvy_to_yuv_downscale2(uyvy_data, width, height, yuv_buffer,
                                          buffer_size);
  case 4:
    return convert_uyvy_to_yuv_downscale4(uyvy_data, width, height, yuv_buffer,
                                          buffer_size);
  case 8:
    return convert_uyvy_to_yuv_downscale8(uyvy_data, width, height, yuv_buffer,
                                          buffer_size);
  default:
    debug_print("Unsupported downscale factor: %d", downscale_factor);
    return false;
  }
}