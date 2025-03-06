// Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <pthread.h>

// Paparazzi includes
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "modules/computer_vision/lib/encoding/rtp.h"
#include "modules/computer_vision/lib/encoding/jpeg.h"

// Project includes
#include "video_stream.h"  // For streaming functionality
#include "image_convert.h" // For YUV to RGB conversion
#include "obstacle_detection.h"
#include "inference.h"     // For running inference

// Other includes
#include "udp_socket.h"
#include "mcu_periph/udp.h"

//////
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#define DEBUG_TAG "OBSDET"
#define MAX_LOG_LENGTH 256
////////


#ifndef OBSTACLE_DETECTION_CAMERA
#define OBSTACLE_DETECTION_CAMERA front_camera
#endif

#ifndef OUTPUT_STREAM_PORT
#define OUTPUT_STREAM_PORT 5100
#endif

#ifndef VIEWVIDEO_QUALITY_FACTOR
#define VIEWVIDEO_QUALITY_FACTOR 50
#endif

// Shared data structure
struct shared_data_t shared_data = {
    .frame_ready = false,
    .frame = NULL,
    .frames_received = 0,
    .frames_processed = 0,
    .last_frame_time = {0, 0}
};


// Enable/disable obstacle detection and streaming
struct obstacle_detection_t obstacle_detection = {
    .enabled = true,
    .stream_enabled = true
};

// Global variables
static struct video_listener* video_listener = NULL;
static pthread_mutex_t frame_mutex;
static struct image_t* current_frame = NULL;
// static struct image_t frame_copy = {.buf=NULL};
// static struct shared_data_t shared_data = {0, 0, false};
static struct stream_context_t stream_ctx = {0};

static void debug_print(const char* format, ...) {
    char message[MAX_LOG_LENGTH];
    va_list args;
    va_start(args, format);
    
    #ifdef TARGET_AP
        // On actual drone, use ulogger
        vsnprintf(message, sizeof(message), format, args);
        char command[MAX_LOG_LENGTH + 32];
        snprintf(command, sizeof(command), "ulogger -t %s '%s'", DEBUG_TAG, message);
        system(command);
    #else
        // In simulation (NPS/Gazebo), use printf
        printf("[%s] ", DEBUG_TAG);
        vprintf(format, args);
        printf("\n");
        fflush(stdout);
    #endif
    
    va_end(args);
}

// Video callback function (runs in video thread)
struct image_t* video_callback(struct image_t *img, uint8_t camera_id __attribute__((unused))) {
    pthread_mutex_lock(&frame_mutex);
    
    shared_data.frames_received++;
    
    struct timeval now;
    gettimeofday(&now, NULL);
    if (shared_data.last_frame_time.tv_sec != 0) {
        float dt = ((now.tv_sec - shared_data.last_frame_time.tv_sec) * 1000000.0f + 
                   (now.tv_usec - shared_data.last_frame_time.tv_usec)) / 1000000.0f;
        if (shared_data.frames_received % 300 == 0) {
            printf("Received frame rate: %.2f FPS\n", 1.0f/dt);
        }
    }
    shared_data.last_frame_time = now;

    if (current_frame == NULL) {
        current_frame = malloc(sizeof(struct image_t));
        image_create(current_frame, img->w, img->h, img->type);
    } else if (current_frame->w != img->w || current_frame->h != img->h) {
        image_free(current_frame);
        image_create(current_frame, img->w, img->h, img->type);
    }
    
    image_copy(img, current_frame);
    shared_data.frame_ready = true;
    
    pthread_mutex_unlock(&frame_mutex);
    return NULL; 
}

bool obstacle_detection_init(void) {
    debug_print("ulogger -t [OBSDET] 'Init called'");
    // Initialize mutex first
    if (pthread_mutex_init(&frame_mutex, NULL) != 0) {
        printf("[Obstacle Detection] Failed to initialize mutex\n");
        return false;
    }

    // Register video callback before stream init
    video_listener = cv_add_to_device(&OBSTACLE_DETECTION_CAMERA, video_callback, 0, 0); // 0,0 means full FPS, cam id 0
    if (video_listener == NULL) {
        debug_print("[Obstacle Detection] Failed to register video callback\n");
        pthread_mutex_destroy(&frame_mutex);
        return false;
    }

    // Initialize YUV to RGB conversion tables
    if (!init_yuv_conversion()) {
        debug_print("[Obstacle Detection] Failed to initialize YUV conversion tables\n");
        pthread_mutex_destroy(&frame_mutex);
        return false;
    }

    // Initialize stream context
    memset(&stream_ctx, 0, sizeof(stream_ctx));
    stream_ctx.img_jpeg.buf = NULL;
    stream_ctx.img_jpeg.buf_size = 0;
    stream_ctx.img_jpeg.w = 0;
    stream_ctx.img_jpeg.h = 0;
    stream_ctx.img_jpeg.type = IMAGE_JPEG;

    // Initialize stream last
    if (!init_stream(&stream_ctx, "127.0.0.1", OUTPUT_STREAM_PORT)) {
        debug_print("[Obstacle Detection] Failed to initialize video stream\n");
        pthread_mutex_destroy(&frame_mutex);
        return false;
    }

    //  Init yuv conversion
    if (!init_yuv_conversion()) {
        debug_print("[Obstacle Detection] Failed to initialize YUV conversion tables\n");
        pthread_mutex_destroy(&frame_mutex);
        return false;
    }

    if (!init_inference()) {
        debug_print("[Obstacle Detection] Failed to initialize inference\n");
        return false;
    }

    debug_print("[Obstacle Detection] Initialized successfully\n");
    return true;
}

void obstacle_detection_periodic(void) {
    if (!obstacle_detection.enabled) {
        return;
    }

    pthread_mutex_lock(&frame_mutex);
    bool frame_ready = shared_data.frame_ready;
    struct image_t* frame = current_frame;
    shared_data.frame_ready = false;
    pthread_mutex_unlock(&frame_mutex);

    if (!frame_ready || !frame) {
        return;
    }

    // Update processing statistics
    if (frame_ready) {
        shared_data.frames_processed++;
        if (shared_data.frames_processed % 300 == 0) {
            debug_print("Frames received: %d, processed: %d\n", 
                shared_data.frames_received, shared_data.frames_processed);
        }
    }

    // Create a copy of the frame that we can modify
    struct image_t display_frame = {
        .w = frame->w,
        .h = frame->h,
        .type = IMAGE_YUV422,
        .buf_size = frame->buf_size,
        .buf = malloc(frame->buf_size)
    };

    if (!display_frame.buf) {
        debug_print("[Obstacle Detection] Failed to allocate display frame buffer\n");
        return;
    }

    // Convert YUV422 to RGB
    float* rgb = NULL;
    if (!convert_uyvy_to_rgb(frame->buf, frame->w, frame->h, &rgb)) {
        debug_print("[Obstacle Detection] Failed to convert YUV422 to RGB\n");
        free(display_frame.buf);
        return;
    }

    // Run inference
    struct model_output_t model_output;
    if (run_inference(rgb, frame->w, frame->h, &model_output)) {
        // Print inference results periodically
        if (shared_data.frames_processed % 30 == 0) {
            debug_print("\nInference results:\n");
            debug_print("Grid values: ");
            for (int i = 0; i < 5; i++) {
                debug_print("%.3f ", model_output.values[i]);
            }
            debug_print("\n");
        }
    } else {
        debug_print("[Obstacle Detection] Inference failed\n");
    }

    // Copy the original frame and stream it
    memcpy(display_frame.buf, frame->buf, frame->buf_size);
    // if (!stream_frame(&stream_ctx, &display_frame)) {
    //     printf("[Obstacle Detection] Failed to stream frame\n");
    // }

    // Cleanup
    free(display_frame.buf);
    free(rgb);
}

void obstacle_detection_cleanup(void) {
    if (current_frame != NULL) {
        image_free(current_frame);
        free(current_frame);
        current_frame = NULL;
    }
    if (stream_ctx.img_jpeg.buf != NULL) {
        image_free(&stream_ctx.img_jpeg);
    }
    cleanup_stream(&stream_ctx);
    cleanup_inference();
    pthread_mutex_destroy(&frame_mutex);
}