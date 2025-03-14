// Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/types.h>  // For struct stat
#include <sys/stat.h>   // For mkdir and struct stat
#include <errno.h>      // For errno
#include <string.h>     // For strerror
#include <unistd.h>     // For getcwd

#include "debug_print.h"
DEFINE_DEBUG_PRINT("OBSDET")

// Paparazzi includes
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "modules/computer_vision/lib/encoding/rtp.h"
#include "modules/computer_vision/lib/encoding/jpeg.h"
#include "modules/core/abi.h"

// Project includes
#include "video_stream.h"  // For streaming functionality
#include "image_convert.h" // For YUV to RGB conversion
#include "obstacle_detection.h"
#include "inference.h"     // For running inference
#include "model.h"        // For model dimensions

// Other includes
#include "udp_socket.h"
#include "mcu_periph/udp.h"

// Debug configuration
#define DEBUG_TAG "OBSDET"
#define MAX_LOG_LENGTH 256

// Camera configuration
#ifndef OBSTACLE_DETECTION_FRONT_CAMERA
#define OBSTACLE_DETECTION_FRONT_CAMERA front_camera
#endif

#ifndef OBSTACLE_DETECTION_BOTTOM_CAMERA
#define OBSTACLE_DETECTION_BOTTOM_CAMERA bottom_camera
#endif

#ifndef OBSTACLE_FRONT_RTP_PORT
#define OBSTACLE_FRONT_RTP_PORT 5100
#endif

#ifndef OBSTACLE_BOTTOM_RTP_PORT
#define OBSTACLE_BOTTOM_RTP_PORT 5101
#endif

// static void debug_print(const char* format, ...);
static struct image_t* front_camera_callback(struct image_t *img, uint8_t camera_id);
static struct image_t* bottom_camera_callback(struct image_t *img, uint8_t camera_id);

// Global variables
struct obstacle_detection_t obstacle_detection = {
    .front_enabled = true,
    .bottom_enabled = true,
    .stream_enabled = false
};

struct camera_data_t front_camera_data = {0};
struct camera_data_t bottom_camera_data = {0};
static struct video_listener* front_video_listener = NULL;
static struct video_listener* bottom_video_listener = NULL;

static void update_fps(struct timeval *last_time, float *fps) {
    struct timeval now;
    gettimeofday(&now, NULL);
    
    if (last_time->tv_sec != 0) {
        float dt = ((now.tv_sec - last_time->tv_sec) * 1000000.0f + 
                   (now.tv_usec - last_time->tv_usec)) / 1000000.0f;
        *fps = 0.95f * *fps + 0.05f * (1.0f / dt);
    }
    *last_time = now;
}

struct image_t* front_camera_callback(struct image_t *img, uint8_t camera_id __attribute__((unused))) {
    pthread_mutex_lock(&front_camera_data.frame_mutex);
    
    update_fps(&front_camera_data.last_frame_received_time, 
        &front_camera_data.received_fps);

    if (front_camera_data.frame == NULL) {
        front_camera_data.frame = malloc(sizeof(struct image_t));
        image_create(front_camera_data.frame, img->w, img->h, img->type);
    } else if (front_camera_data.frame->w != img->w || front_camera_data.frame->h != img->h) {
        image_free(front_camera_data.frame);
        image_create(front_camera_data.frame, img->w, img->h, img->type);
    }
    
    image_copy(img, front_camera_data.frame);
    front_camera_data.frame_ready = true;
    
    pthread_mutex_unlock(&front_camera_data.frame_mutex);
    return NULL;
}

struct image_t* bottom_camera_callback(struct image_t *img, uint8_t camera_id __attribute__((unused))) {
    if (!img) {
        debug_print("Received null image!");
        return NULL;
    }
    
    // Validate image dimensions
    if (img->w <= 0 || img->h <= 0 || img->w > 10000 || img->h > 10000) {
        debug_print("Invalid image dimensions: %dx%d", img->w, img->h);
        return NULL;
    }

    // debug_print("Bottom camera frame: %dx%d, type=%d, buf_size=%d", 
    //     (int)img->w, (int)img->h, (int)img->type, (int)img->buf_size);

    pthread_mutex_lock(&bottom_camera_data.frame_mutex);
    
    // Allocate or reallocate frame buffer if needed
    if (bottom_camera_data.frame == NULL) {
        bottom_camera_data.frame = malloc(sizeof(struct image_t));
        if (!bottom_camera_data.frame) {
            debug_print("Failed to allocate frame struct");
            pthread_mutex_unlock(&bottom_camera_data.frame_mutex);
            return NULL;
        }
        memset(bottom_camera_data.frame, 0, sizeof(struct image_t));
    }

    // Create image with explicit size check
    size_t required_size = (size_t)img->w * img->h * 2; // YUV422 format
    if (bottom_camera_data.frame->buf_size != required_size) {
        if (bottom_camera_data.frame->buf) {
            image_free(bottom_camera_data.frame);
        }
        debug_print("Creating new frame buffer: %d bytes", (int)required_size);
        
        // Instead of checking the return value, just call image_create
        image_create(bottom_camera_data.frame, img->w, img->h, img->type);
        
        // Then verify the buffer was created successfully
        if (!bottom_camera_data.frame->buf || bottom_camera_data.frame->buf_size != required_size) {
            debug_print("Failed to create frame buffer");
            pthread_mutex_unlock(&bottom_camera_data.frame_mutex);
            return NULL;
        }
    }
    
    // Copy with validation
    if (img->buf && img->buf_size <= bottom_camera_data.frame->buf_size) {
        memcpy(bottom_camera_data.frame->buf, img->buf, img->buf_size);
        bottom_camera_data.frame->w = img->w;
        bottom_camera_data.frame->h = img->h;
        bottom_camera_data.frame->buf_size = img->buf_size;
        bottom_camera_data.frame_ready = true;
    } else {
        debug_print("Invalid buffer sizes: src=%d, dst=%d", 
            (int)img->buf_size, (int)bottom_camera_data.frame->buf_size);
    }
    
    pthread_mutex_unlock(&bottom_camera_data.frame_mutex);
    return NULL;
}

static void ensure_directory_exists(const char* path) {
    struct stat st;
    if (stat(path, &st) == -1) {
        debug_print("Directory %s does not exist, creating it", path);
        if (mkdir(path, 0700) == -1) {
            debug_print("Failed to create directory %s: %s", path, strerror(errno));
        } else {
            debug_print("Successfully created directory %s", path);
        }
    } else {
        debug_print("Directory %s already exists", path);
    }
}

static void save_yuv_image(const uint8_t* data, int width, int height, const char* filename) {
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
    } else {
        debug_print("Successfully wrote %zu bytes to %s", bytes_written, filename);
    }

    fclose(f);
}

static void save_rgb_image(const float* r_data, const float* g_data, const float* b_data, 
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

    // Write PPM header
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    
    // Write image data
    size_t bytes_written = fwrite(rgb, 1, width * height * 3, f);
    if (bytes_written != width * height * 3) {
        debug_print("Failed to write all data. Wrote %zu of %d bytes", 
            bytes_written, width * height * 3);
    } else {
        debug_print("Successfully wrote %zu bytes to %s", bytes_written, filename);
    }
    
    fclose(f);
    free(rgb);
}

bool obstacle_detection_init(void) {
    debug_print("Init called");
    
    // Initialize mutexes for both cameras
    if (pthread_mutex_init(&front_camera_data.frame_mutex, NULL) != 0 ||
        pthread_mutex_init(&bottom_camera_data.frame_mutex, NULL) != 0) {
        debug_print("Failed to initialize mutexes");
        return false;
    }

    // Allocate RGB buffers
    front_camera_data.rgb_buffer_size = get_rgb_buffer_size(FRONT_CAMERA_WIDTH, FRONT_CAMERA_HEIGHT);
    size_t bottom_size = (size_t)BOTTOM_CAMERA_WIDTH * BOTTOM_CAMERA_HEIGHT * 3 * sizeof(float);
    debug_print("Allocating bottom camera RGB buffer: %dx%d = %zu bytes", 
        BOTTOM_CAMERA_WIDTH, BOTTOM_CAMERA_HEIGHT, bottom_size);
    
    front_camera_data.rgb_buffer = malloc(front_camera_data.rgb_buffer_size);
    bottom_camera_data.rgb_buffer_size = bottom_size;
    bottom_camera_data.rgb_buffer = malloc(bottom_size);

    if (!bottom_camera_data.rgb_buffer) {
        debug_print("Failed to allocate bottom camera RGB buffer (%zu bytes)", bottom_size);
        return false;
    }
    
    if (!front_camera_data.rgb_buffer) {
        debug_print("Failed to allocate front camera RGB buffers");
        cleanup_inference();
        return false;
    }

    // Register video callbacks for both cameras
    front_video_listener = cv_add_to_device(&OBSTACLE_DETECTION_FRONT_CAMERA, front_camera_callback, 10, 0);
    bottom_video_listener = cv_add_to_device(&OBSTACLE_DETECTION_BOTTOM_CAMERA, bottom_camera_callback, 10, 0);
    
    if (front_video_listener == NULL || bottom_video_listener == NULL) {
        debug_print("Failed to register video callbacks");
        pthread_mutex_destroy(&front_camera_data.frame_mutex);
        pthread_mutex_destroy(&bottom_camera_data.frame_mutex);
        return false;
    }

    // Initialize YUV to RGB conversion tables
    if (!init_yuv_conversion()) {
        debug_print("Failed to initialize YUV conversion tables");
        pthread_mutex_destroy(&front_camera_data.frame_mutex);
        pthread_mutex_destroy(&bottom_camera_data.frame_mutex);
        return false;
    }

    // Initialize stream contexts for both cameras
    memset(&front_camera_data.stream_ctx, 0, sizeof(struct stream_context_t));
    memset(&bottom_camera_data.stream_ctx, 0, sizeof(struct stream_context_t));
    
    front_camera_data.stream_ctx.img_jpeg = (struct image_t){
        .buf = NULL,
        .buf_size = 0,
        .w = 0,
        .h = 0,
        .type = IMAGE_JPEG
    };
    
    bottom_camera_data.stream_ctx.img_jpeg = (struct image_t){
        .buf = NULL,
        .buf_size = 0,
        .w = 0,
        .h = 0,
        .type = IMAGE_JPEG
    };

    // Initialize streams for both cameras
    if (!init_stream(&front_camera_data.stream_ctx, "127.0.0.1", OBSTACLE_FRONT_RTP_PORT) ||
        !init_stream(&bottom_camera_data.stream_ctx, "127.0.0.1", OBSTACLE_BOTTOM_RTP_PORT)) {
        debug_print("Failed to initialize video streams");
        pthread_mutex_destroy(&front_camera_data.frame_mutex);
        pthread_mutex_destroy(&bottom_camera_data.frame_mutex);
        return false;
    }

    // Initialize inference system
    if (!init_inference()) {
        debug_print("Failed to initialize inference");
        return false;
    }

    debug_print("Initialized successfully");
    return true;
}

void obstacle_detection_periodic(void) {
    // Process front camera (obstacle detection)
    if (obstacle_detection.front_enabled) {
        pthread_mutex_lock(&front_camera_data.frame_mutex);
        bool frame_ready = front_camera_data.frame_ready;
        struct image_t* frame = front_camera_data.frame;
        front_camera_data.frame_ready = false;
        pthread_mutex_unlock(&front_camera_data.frame_mutex);

        if (!frame_ready || !frame) {
            return;
        }

        // Update processing statistics
        front_camera_data.frames_processed++;
        if (front_camera_data.frames_processed % 300 == 0) {
            debug_print("Front camera frames received: %d, processed: %d", 
                front_camera_data.frames_received, front_camera_data.frames_processed);
        }

        // Convert YUV422 to RGB using pre-allocated buffer
        if (!convert_uyvy_to_rgb_front(frame->buf, frame->w, frame->h, 
                                     front_camera_data.rgb_buffer,
                                     front_camera_data.rgb_buffer_size)) {
            debug_print("Failed to convert YUV422 to RGB for front camera");
            return;
        }

        // Run inference for obstacle detection
        struct model_output_t model_output;
        if (run_obstacle_inference(front_camera_data.rgb_buffer, frame->w, frame->h, &model_output)) {
            // Print inference results periodically
            if (front_camera_data.frames_processed % 10 == 0) {
                char row_str[2 + (MODEL_OUTPUT_COL_SIZE * 7) + 2];
                for (int i = 0; i < MODEL_OUTPUT_ROW_SIZE; i++) {
                    int offset = sprintf(row_str, "  ");
                    for (int j = 0; j < MODEL_OUTPUT_COL_SIZE; j++) {
                        offset += sprintf(row_str + offset, "%6.3f ", model_output.values[i][j]);
                    }
                    sprintf(row_str + offset, " ");
                    debug_print(row_str);
                }
            }
        } else {
            debug_print("Front camera inference failed");
        }

        // Send ABI message for obstacle detection
        AbiSendMsgMODELDATA(ABI_BROADCAST, 
            MODEL_TYPE_OBSTACLE,
            MODEL_OUTPUT_ROW_SIZE,
            MODEL_OUTPUT_COL_SIZE,
            (float*)model_output.values  // Cast 2D array to 1D
        );

        // update_fps(&front_camera_data.last_frame_processed_time, 
        //     &front_camera_data.processed_fps);
  
        // // Print FPS periodically
        // if (front_camera_data.frames_processed % 300 == 0) {
        //     debug_print("Front camera FPS - Received: %.2f, Processed: %.2f", 
        //         front_camera_data.received_fps, front_camera_data.processed_fps);
        // }

        // Stream front camera frame if enabled
        if (obstacle_detection.stream_enabled) {
            if (!stream_frame(&front_camera_data.stream_ctx, frame)) {
                debug_print("Failed to stream front camera frame");
            }
        }
    }

    // Process bottom camera (border detection)
    if (obstacle_detection.bottom_enabled) {
        bottom_camera_data.frames_processed++;
        pthread_mutex_lock(&bottom_camera_data.frame_mutex);
        bool frame_ready = bottom_camera_data.frame_ready;
        struct image_t* current_frame = bottom_camera_data.frame;  // Changed name to avoid shadowing
        bottom_camera_data.frame_ready = false;
        pthread_mutex_unlock(&bottom_camera_data.frame_mutex);
    
        if (!frame_ready || !current_frame) {
            return;
        }
    
        // Convert YUV422 to RGB using pre-allocated buffer
        if (!convert_uyvy_to_rgb_bottom(current_frame->buf, current_frame->w, current_frame->h,
                                      bottom_camera_data.rgb_buffer,
                                      bottom_camera_data.rgb_buffer_size)) {
            debug_print("Failed to convert YUV422 to RGB for bottom camera");
            return;
        }
    
        // Run inference for border detection
        struct border_output_t border_output;
        if (run_border_inference(bottom_camera_data.rgb_buffer, current_frame->w, current_frame->h, &border_output)) {
            if (bottom_camera_data.frames_processed % 30 == 0) {
                debug_print("Bottom camera inference results: %.3f", border_output.value);
            }
        } else {
            debug_print("Bottom camera inference failed");
        }
    
        AbiSendMsgFLOORDATA(ABI_BROADCAST, border_output.value);
    
        if (bottom_camera_data.frames_processed % 5 == 0) {  // Save every 30th frame
            char cwd[512];
            if (getcwd(cwd, sizeof(cwd)) == NULL) {
                debug_print("Failed to get current working directory: %s", strerror(errno));
                return;
            }
            // debug_print("Current working directory: %s", cwd);
        
            char debug_dir[768];
            snprintf(debug_dir, sizeof(debug_dir), "%s/camera_debug", cwd);
            ensure_directory_exists(debug_dir);
        
            // Get image dimensions from the frame
            int width = current_frame->w;
            int height = current_frame->h;
            // debug_print("Image dimensions: %dx%d", width, height);
        
            char yuv_filename[1024];
            snprintf(yuv_filename, sizeof(yuv_filename), 
                "%s/frame_%d_yuv.raw", debug_dir, bottom_camera_data.frames_processed);
            save_yuv_image(current_frame->buf, width, height, yuv_filename);
        
            char rgb_filename[1024];
            snprintf(rgb_filename, sizeof(rgb_filename), 
                "%s/frame_%d_rgb.ppm", debug_dir, bottom_camera_data.frames_processed);
            save_rgb_image(bottom_camera_data.rgb_buffer,
                bottom_camera_data.rgb_buffer + (width * height),
                bottom_camera_data.rgb_buffer + (2 * width * height),
                width, height, rgb_filename);
        
            // debug_print("Attempted to save frame %d", bottom_camera_data.frames_processed);
        }
    
        // Stream bottom camera if enabled
        if (obstacle_detection.stream_enabled) {
            if (!stream_frame(&bottom_camera_data.stream_ctx, current_frame)) {
                debug_print("Failed to stream bottom camera frame");
            }
        }
    }
}

void obstacle_detection_cleanup(void) {
    // Free RGB buffers
    free(front_camera_data.rgb_buffer);
    free(bottom_camera_data.rgb_buffer);
    front_camera_data.rgb_buffer = NULL;
    bottom_camera_data.rgb_buffer = NULL;
    // Cleanup front camera resources
    if (front_camera_data.frame != NULL) {
        image_free(front_camera_data.frame);
        free(front_camera_data.frame);
        front_camera_data.frame = NULL;
    }
    if (front_camera_data.stream_ctx.img_jpeg.buf != NULL) {
        image_free(&front_camera_data.stream_ctx.img_jpeg);
    }
    cleanup_stream(&front_camera_data.stream_ctx);
    pthread_mutex_destroy(&front_camera_data.frame_mutex);

    // Cleanup bottom camera resources
    if (bottom_camera_data.frame != NULL) {
        image_free(bottom_camera_data.frame);
        free(bottom_camera_data.frame);
        bottom_camera_data.frame = NULL;
    }
    if (bottom_camera_data.stream_ctx.img_jpeg.buf != NULL) {
        image_free(&bottom_camera_data.stream_ctx.img_jpeg);
    }
    cleanup_stream(&bottom_camera_data.stream_ctx);
    pthread_mutex_destroy(&bottom_camera_data.frame_mutex);

    // Cleanup inference system
    cleanup_inference();
}