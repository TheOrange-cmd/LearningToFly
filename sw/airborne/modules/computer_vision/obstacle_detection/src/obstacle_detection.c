// Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdarg.h>

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
    pthread_mutex_lock(&bottom_camera_data.frame_mutex);
    
    update_fps(&bottom_camera_data.last_frame_received_time, 
        &bottom_camera_data.received_fps);

    if (bottom_camera_data.frame == NULL) {
        bottom_camera_data.frame = malloc(sizeof(struct image_t));
        image_create(bottom_camera_data.frame, img->w, img->h, img->type);
    } else if (bottom_camera_data.frame->w != img->w || bottom_camera_data.frame->h != img->h) {
        image_free(bottom_camera_data.frame);
        image_create(bottom_camera_data.frame, img->w, img->h, img->type);
    }
    
    image_copy(img, bottom_camera_data.frame);
    bottom_camera_data.frame_ready = true;
    
    pthread_mutex_unlock(&bottom_camera_data.frame_mutex);
    return NULL;
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
    bottom_camera_data.rgb_buffer_size = get_rgb_buffer_size(BOTTOM_CAMERA_WIDTH, BOTTOM_CAMERA_HEIGHT);
    
    front_camera_data.rgb_buffer = malloc(front_camera_data.rgb_buffer_size);
    bottom_camera_data.rgb_buffer = malloc(bottom_camera_data.rgb_buffer_size);
    
    if (!front_camera_data.rgb_buffer || !bottom_camera_data.rgb_buffer) {
        debug_print("Failed to allocate RGB buffers");
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
                    debug_print("Front camera inference results: %s", row_str);
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

        update_fps(&front_camera_data.last_frame_processed_time, 
            &front_camera_data.processed_fps);
  
        // Print FPS periodically
        if (front_camera_data.frames_processed % 300 == 0) {
            debug_print("Front camera FPS - Received: %.2f, Processed: %.2f", 
                front_camera_data.received_fps, front_camera_data.processed_fps);
        }

        // Stream front camera frame if enabled
        if (obstacle_detection.stream_enabled) {
            if (!stream_frame(&front_camera_data.stream_ctx, frame)) {
                debug_print("Failed to stream front camera frame");
            }
        }
    }

    // Process bottom camera (border detection)
    if (obstacle_detection.bottom_enabled) {
        pthread_mutex_lock(&bottom_camera_data.frame_mutex);
        bool frame_ready = bottom_camera_data.frame_ready;
        struct image_t* frame = bottom_camera_data.frame;
        bottom_camera_data.frame_ready = false;
        pthread_mutex_unlock(&bottom_camera_data.frame_mutex);

        if (!frame_ready || !frame) {
            return;
        }

        // Convert YUV422 to RGB using pre-allocated buffer
        if (!convert_uyvy_to_rgb_bottom(frame->buf, frame->w, frame->h,
                                      bottom_camera_data.rgb_buffer,
                                      bottom_camera_data.rgb_buffer_size)) {
            debug_print("Failed to convert YUV422 to RGB for bottom camera");
            return;
        }

        // Run inference for border detection
        struct border_output_t border_output;
        if (run_border_inference(bottom_camera_data.rgb_buffer, frame->w, frame->h, &border_output)) {
            if (bottom_camera_data.frames_processed % 10 == 0) {
                debug_print("Bottom camera inference results: %.3f", border_output.value);
            }
        } else {
            debug_print("Bottom camera inference failed");
        }

        // Send ABI message for border detection
        AbiSendMsgMODELDATA(ABI_BROADCAST,
            MODEL_TYPE_BORDER,
            1,  // rows
            1,  // cols
            &border_output.value
        );

        update_fps(&bottom_camera_data.last_frame_processed_time, 
            &bottom_camera_data.processed_fps);
  
        // Print FPS periodically
        if (bottom_camera_data.frames_processed % 300 == 0) {
            debug_print("Bottom camera FPS - Received: %.2f, Processed: %.2f", 
                bottom_camera_data.received_fps, bottom_camera_data.processed_fps);
        }

        // Stream bottom camera if enabled
        if (obstacle_detection.stream_enabled) {
            if (!stream_frame(&bottom_camera_data.stream_ctx, frame)) {
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