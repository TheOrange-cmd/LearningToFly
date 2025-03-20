#include "image_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include "debug_print.h"

DEFINE_DEBUG_PRINT("IMG_UTILS")

void ensure_directory_exists(const char* path) {
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

void save_yuv_image(const uint8_t* data, int width, int height, const char* filename) {
    debug_print("Attempting to save YUV image to %s", filename);
    
    if (!data) {
        debug_print("YUV data pointer is NULL");
        return;
    }

    FILE* f = fopen(filename, "wb");
    if (!f) {
        debug_print("Failed to open file for writing: %s - Error: %s", filename, strerror(errno));
        return;
    }

    size_t bytes_written = fwrite(data, 1, width * height * 2, f);
    if (bytes_written != width * height * 2) {
        debug_print("Failed to write all data. Wrote %zu of %d bytes", 
            bytes_written, width * height * 2);
    }

    fclose(f);
}

void save_rgb_image(const float* r_data, const float* g_data, const float* b_data, 
                    int width, int height, const char* filename) {
    debug_print("Attempting to save RGB image to %s", filename);
    
    if (!r_data || !g_data || !b_data) {
        debug_print("One or more RGB data pointers is NULL");
        return;
    }

    uint8_t* rgb = malloc(width * height * 3);
    if (!rgb) {
        debug_print("Failed to allocate RGB buffer for saving");
        return;
    }

    // Convert float [0,1] to byte [0,255]
    for (int i = 0; i < width * height; i++) {
        rgb[i*3] = (uint8_t)(r_data[i] * 255.0f);
        rgb[i*3+1] = (uint8_t)(g_data[i] * 255.0f);
        rgb[i*3+2] = (uint8_t)(b_data[i] * 255.0f);
    }

    FILE* f = fopen(filename, "wb");
    if (!f) {
        debug_print("Failed to open file for writing: %s - Error: %s", filename, strerror(errno));
        free(rgb);
        return;
    }

    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(rgb, 1, width * height * 3, f);
    
    fclose(f);
    free(rgb);
}