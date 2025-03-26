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
 * @brief Functions to save images to disk. Note that saving images is slow,
 * only use for debugging purposes or for collecting training data.
 *
 * @note Developement assisted by Claude Sonnet 3.5
 */

#include "image_utils.h"
#include "debug_print.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

DEFINE_DEBUG_PRINT("IMG_UTILS")

/**
 * @brief Create directory if it doesn't exist
 *
 * Checks directory existence and creates with 0700 permissions if missing.
 * Logs success/failure via debug_print.
 *
 * @param path Directory path to check/create
 */
void ensure_directory_exists(const char *path) {
  struct stat st;
  if (stat(path, &st) == -1) {
    debug_print("Directory %s does not exist, creating it", path);
    if (mkdir(path, 0700) == -1) {
      debug_print("Failed to create directory %s: %s", path, strerror(errno));
    } else {
      debug_print("Successfully created directory %s", path);
    }
  }
}

/**
 * @brief Generate a unique filename for saving images
 *
 * Creates a filename with timestamp and sequence number to prevent overwriting.
 *
 * @param prefix Prefix for the filename (e.g., "front" or "bottom")
 * @param extension File extension
 * @return Dynamically allocated string with filename
 */
char *generate_unique_filename(const char *prefix, const char *extension) {
  // Ensure images directory exists
  ensure_directory_exists("images");

  // Get current timestamp
  struct timeval tv;
  gettimeofday(&tv, NULL);

  // Use static counter to ensure unique filenames
  static uint32_t sequence_number = 0;
  sequence_number++;

  // Allocate buffer for filename
  char *filename = malloc(256 * sizeof(char));
  if (!filename) {
    debug_print("Failed to allocate memory for filename");
    return NULL;
  }

  // Create filename with timestamp and sequence number
  snprintf(filename, 256, "images/%s_%lu_%u.%s", prefix,
           (unsigned long)tv.tv_sec, sequence_number, extension);

  return filename;
}

/**
 * @brief Save raw YUV422 image data to file
 *
 * Writes UYVY format data directly to binary file. Handles null pointers
 * and file errors with debug logging.
 *
 * @param data UYVY422 image buffer
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @param filename Full output path including filename
 */
void save_yuv_image(const uint8_t *data, int width, int height,
                    const char *filename) {
  debug_print("Attempting to save YUV image to %s", filename);

  if (!data) {
    debug_print("YUV data pointer is NULL");
    return;
  }

  FILE *f = fopen(filename, "wb");
  if (!f) {
    debug_print("Failed to open file for writing: %s - Error: %s", filename,
                strerror(errno));
    return;
  }

  size_t total_bytes = (size_t)(width * height * 2);
  size_t bytes_written = fwrite(data, 1, total_bytes, f);
  if (bytes_written != total_bytes) {
    debug_print("Failed to write all data. Wrote %zu of %zu bytes",
                bytes_written, total_bytes);
  }

  fclose(f);
}