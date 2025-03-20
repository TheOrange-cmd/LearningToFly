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
#include "lib/v4l/v4l2.h"

// Project includes
#include "video_stream.h"  // For streaming functionality
#include "image_convert.h" // For YUV to RGB conversion
#include "obstacle_detection.h"
#include "inference.h"     // For running inference
#include "model.h"        // For model dimensions
#include "queue.h"       // For processing queue
#include "image_utils.h" // For image saving

// Other includes
#include "udp_socket.h"
#include "mcu_periph/udp.h"

// Debug configuration
#define DEBUG_TAG "OBSDET"
#define MAX_LOG_LENGTH 256


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

// bool for debug prints
static bool debug = false;
static bool debug_model = true;

static uint32_t front_frames_received = 0;
static uint32_t front_frames_processed = 0;
static uint32_t bottom_frames_received = 0;
static uint32_t bottom_frames_processed = 0;

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

// Processing thread functions
static void* front_processing_thread(void* arg) {
    struct processing_thread_t* proc = (struct processing_thread_t*)arg;
    
    while (proc->running) {
        struct image_t* img = queue_pop(&proc->queue);
        if (img) {
            struct timeval t1, t2, t3;
            gettimeofday(&t1, NULL);

            bool conv_success = convert_uyvy_to_yuv_crop(img->buf, 
                img->w, img->h,
                240, 240,
                front_camera_data.processing.rgb_buffer,
                front_camera_data.processing.rgb_buffer_size);

            if (conv_success) {
                struct model_output_t model_output;
                bool inf_success = run_obstacle_inference(
                    front_camera_data.processing.rgb_buffer,
                    240, 240, &model_output);
                if (inf_success) {
                    // Send ABI message for obstacle detection
 		AbiSendMsgMODELOUTPUT(38, get_sys_time_usec(), (float*)model_output.values  // Cast 2D array to 1D
                    );
                    if (debug_model) {
                        debug_print("Front model outputs: %.2f, %.2f, %.2f",
                            model_output.values[0][0], model_output.values[0][1],
                            model_output.values[0][2]);
                    }
                }
                else {
                    debug_print("Failed to run front inference");
                }
                gettimeofday(&t3, NULL);
                
                float conv_time = (t2.tv_sec - t1.tv_sec) * 1000.0f + 
                                (t2.tv_usec - t1.tv_usec) / 1000.0f;
                float inf_time = (t3.tv_sec - t2.tv_sec) * 1000.0f + 
                                (t3.tv_usec - t2.tv_usec) / 1000.0f;
                if (debug) {
                    debug_print("Front processing times - Convert: %.1fms, Inference: %.1fms",
                            conv_time, inf_time);
                }
            }
            else {
                debug_print("Failed to convert front UYVY to YUV");
            }

            image_free(img);
            free(img);
        }
    }
    return NULL;
}

struct image_t* front_camera_callback(struct image_t *img, uint8_t camera_id __attribute__((unused))) {
    static uint32_t callback_count = 0;
    static struct timeval last_callback = {0, 0};
    struct timeval now;
    gettimeofday(&now, NULL);
    if (debug) {
        callback_count++;
    
        if (last_callback.tv_sec != 0) {
            float dt = (now.tv_sec - last_callback.tv_sec) * 1000.0f + 
                       (now.tv_usec - last_callback.tv_usec) / 1000.0f;
            debug_print("Front callback %d - Time since last: %.1fms", callback_count, dt);
        }
        last_callback = now;
    }


    // print shape of received image
    // debug_print("Received front image with shape: %dx%d", img->w, img->h);

    // Save image to file for debugging
    // ensure_directory_exists("bebop_cam_debug");
    // char filename[256];
    // snprintf(filename, 256, "bebop_cam_debug/front_%d.raw", callback_count);
    // save_yuv_image(img->buf, img->w, img->h, filename);


    if (obstacle_detection.front_enabled) {
        struct image_t* proc_img = malloc(sizeof(struct image_t));
        if (proc_img) {
            image_create(proc_img, img->w, img->h, img->type);
            image_copy(img, proc_img);
            queue_push(&obstacle_detection.front_processor.queue, proc_img);
        }
    }
    
    return NULL;
}

struct image_t* bottom_camera_callback(struct image_t *img, uint8_t camera_id __attribute__((unused))) {
    static uint32_t callback_count = 0;
    static struct timeval last_callback = {0, 0};
    struct timeval now;
    if (debug) {
        gettimeofday(&now, NULL);
    
        callback_count++;
        
        if (last_callback.tv_sec != 0) {
            float dt = (now.tv_sec - last_callback.tv_sec) * 1000.0f + 
                       (now.tv_usec - last_callback.tv_usec) / 1000.0f;
            debug_print("Bottom callback %d - Time since last: %.1fms", callback_count, dt);
        }
        last_callback = now;
    }


    // print shape of received image
    // debug_print("Received bottom image with shape: %dx%d", img->w, img->h);

    if (obstacle_detection.bottom_enabled) {
        struct image_t* proc_img = malloc(sizeof(struct image_t));
        if (proc_img) {
            image_create(proc_img, img->w, img->h, img->type);
            image_copy(img, proc_img);
            queue_push(&obstacle_detection.bottom_processor.queue, proc_img);
        }
    }
    
    return NULL;
}

static void* bottom_processing_thread(void* arg) {
    struct processing_thread_t* proc = (struct processing_thread_t*)arg;
    static uint32_t process_count = 0;
    struct timeval t1, t2, t3;    
    while (proc->running) {
        struct image_t* img = queue_pop(&proc->queue);
        if (img) {
            if (debug) {
                process_count++;
                debug_print("Bottom processing - Frame %d", process_count);
                

                gettimeofday(&t1, NULL);
            }

            
            bool conv_success = convert_uyvy_to_yuv_downscale(img->buf, img->w, img->h,
                                bottom_camera_data.processing.rgb_buffer,
                                bottom_camera_data.processing.rgb_buffer_size, 8);
            if (debug) {
                gettimeofday(&t2, NULL);
            }
            
            if (conv_success) {
                struct border_output_t border_output;
                bool inf_success = run_border_inference(
                    bottom_camera_data.processing.rgb_buffer,
                    30, 30, &border_output);
                if(inf_success) {
                    // Send ABI message for border detection
                    //AbiSendMsgMODELDATA(ABI_BROADCAST,
                      //  MODEL_TYPE_BORDER,
                        //1,  // rows
                        //1,  // cols
                        //&border_output.value
                    //);
                    if (debug_model) {
                        debug_print("Bottom model output: %.2f", border_output.value);
                    }
                }
                if (debug) {
                    gettimeofday(&t3, NULL);
                
                    float conv_time = (t2.tv_sec - t1.tv_sec) * 1000.0f + 
                                    (t2.tv_usec - t1.tv_usec) / 1000.0f;
                    float inf_time = (t3.tv_sec - t2.tv_sec) * 1000.0f + 
                                    (t3.tv_usec - t2.tv_usec) / 1000.0f;
                    
                    debug_print("Bottom processing times - Convert: %.1fms, Inference: %.1fms",
                              conv_time, inf_time);
                    
                }
            }
            image_free(img);
            free(img);
        }
    }
    return NULL;
}


bool obstacle_detection_init(void) {
    debug_print("Init called");

    // Initialize processing threads
    obstacle_detection.front_processor.running = true;
    queue_init(&obstacle_detection.front_processor.queue);
    pthread_create(&obstacle_detection.front_processor.thread_id, NULL, 
                  front_processing_thread, &obstacle_detection.front_processor);
    
    obstacle_detection.bottom_processor.running = true;
    queue_init(&obstacle_detection.bottom_processor.queue);
    pthread_create(&obstacle_detection.bottom_processor.thread_id, NULL, 
                  bottom_processing_thread, &obstacle_detection.bottom_processor);
    
    // Initialize mutexes for both cameras
    if (pthread_mutex_init(&front_camera_data.processing.processing_mutex, NULL) != 0 ||
        pthread_mutex_init(&front_camera_data.streaming.streaming_mutex, NULL) != 0 ||
        pthread_mutex_init(&bottom_camera_data.processing.processing_mutex, NULL) != 0 ||
        pthread_mutex_init(&bottom_camera_data.streaming.streaming_mutex, NULL) != 0) {
        debug_print("Failed to initialize mutexes");
        return false;
    }

    // Allocate RGB buffers
    // front_camera_data.processing.rgb_buffer_size = (size_t)FRONT_CAMERA_WIDTH * FRONT_CAMERA_HEIGHT * 3 * sizeof(float);
    front_camera_data.processing.rgb_buffer_size = (size_t)240 * 240 * 3 * sizeof(float);
    size_t bottom_size = (size_t)30 * 30 * 3 * sizeof(float);
    debug_print("Allocating bottom camera RGB buffer: %dx%d = %zu bytes", 
        30, 30, bottom_size);
    debug_print("Allocating front camera RGB buffer: %dx%d = %zu bytes", 
        FRONT_CAMERA_WIDTH, FRONT_CAMERA_HEIGHT, front_camera_data.processing.rgb_buffer_size);
    
    front_camera_data.processing.rgb_buffer = malloc(front_camera_data.processing.rgb_buffer_size);
    bottom_camera_data.processing.rgb_buffer_size = bottom_size;
    bottom_camera_data.processing.rgb_buffer = malloc(bottom_size);

    if (!bottom_camera_data.processing.rgb_buffer) {
        debug_print("Failed to allocate bottom camera RGB buffer (%zu bytes)", bottom_size);
        return false;
    }
    
    if (!front_camera_data.processing.rgb_buffer) {
        debug_print("Failed to allocate front camera RGB buffers");
        cleanup_inference();
        return false;
    } 

    front_video_listener = cv_add_to_device(&OBSTACLE_DETECTION_FRONT_CAMERA, front_camera_callback, 5, 0);
    bottom_video_listener = cv_add_to_device(&OBSTACLE_DETECTION_BOTTOM_CAMERA, bottom_camera_callback, 5, 0);

    
    if (front_video_listener == NULL || bottom_video_listener == NULL) {
        debug_print("Failed to register video callbacks");
        obstacle_detection_cleanup();
        return false;
    }
    else {
        debug_print("Registered video callbacks");
    }

    // Initialize YUV to RGB conversion tables
    if (!init_yuv_conversion()) {
        debug_print("Failed to initialize YUV conversion tables");
        obstacle_detection_cleanup();
        return false;
    }

    if (obstacle_detection.stream_enabled) {
        // Initialize stream contexts for both cameras
        memset(&front_camera_data.streaming.stream_ctx, 0, sizeof(struct stream_context_t));
        memset(&bottom_camera_data.streaming.stream_ctx, 0, sizeof(struct stream_context_t));
        
        front_camera_data.streaming.stream_ctx.img_jpeg = (struct image_t){
            .buf = NULL,
            .buf_size = 0,
            .w = 0,
            .h = 0,
            .type = IMAGE_JPEG
        };
        
        bottom_camera_data.streaming.stream_ctx.img_jpeg = (struct image_t){
            .buf = NULL,
            .buf_size = 0,
            .w = 0,
            .h = 0,
            .type = IMAGE_JPEG
        };

        // Initialize streams for both cameras
        if (!init_stream(&front_camera_data.streaming.stream_ctx, "127.0.0.1", OBSTACLE_FRONT_RTP_PORT) ||
            !init_stream(&bottom_camera_data.streaming.stream_ctx, "127.0.0.1", OBSTACLE_BOTTOM_RTP_PORT)) {
            debug_print("Failed to initialize video streams");
            obstacle_detection_cleanup();
            return false;
        }
    }

    // Initialize inference system
    if (!init_inference()) {
        debug_print("Failed to initialize inference");
        obstacle_detection_cleanup();
        return false;
    }

    debug_print("Initialized successfully");
    return true;
}

void obstacle_detection_periodic(void) {
    // if (obstacle_detection.stream_enabled) {
    //     // Stream front camera
    //     pthread_mutex_lock(&front_camera_data.streaming.streaming_mutex);
    //     if (front_camera_data.streaming.frame_ready && front_camera_data.streaming.frame) {
    //         stream_frame(&front_camera_data.streaming.stream_ctx, front_camera_data.streaming.frame);
    //         front_camera_data.streaming.frame_ready = false;
    //     }
    //     pthread_mutex_unlock(&front_camera_data.streaming.streaming_mutex);

    //     // Stream bottom camera
    //     pthread_mutex_lock(&bottom_camera_data.streaming.streaming_mutex);
    //     if (bottom_camera_data.streaming.frame_ready && bottom_camera_data.streaming.frame) {
    //         stream_frame(&bottom_camera_data.streaming.stream_ctx, bottom_camera_data.streaming.frame);
    //         bottom_camera_data.streaming.frame_ready = false;
    //     }
    //     pthread_mutex_unlock(&bottom_camera_data.streaming.streaming_mutex);
    // }
}

void obstacle_detection_cleanup(void) {
    // Stop processing threads
    obstacle_detection.front_processor.running = false;
    obstacle_detection.bottom_processor.running = false;
    
    // Signal threads to wake up and exit
    pthread_cond_signal(&obstacle_detection.front_processor.queue.not_empty);
    pthread_cond_signal(&obstacle_detection.bottom_processor.queue.not_empty);
    
    // Wait for threads to finish
    pthread_join(obstacle_detection.front_processor.thread_id, NULL);
    pthread_join(obstacle_detection.bottom_processor.thread_id, NULL);

    // Free RGB buffers
    free(front_camera_data.processing.rgb_buffer);
    free(bottom_camera_data.processing.rgb_buffer);
    front_camera_data.processing.rgb_buffer = NULL;
    bottom_camera_data.processing.rgb_buffer = NULL;

    // Cleanup front camera resources
    if (front_camera_data.streaming.frame != NULL) {
        image_free(front_camera_data.streaming.frame);
        free(front_camera_data.streaming.frame);
        front_camera_data.streaming.frame = NULL;
    }
    if (front_camera_data.streaming.stream_ctx.img_jpeg.buf != NULL) {
        image_free(&front_camera_data.streaming.stream_ctx.img_jpeg);
    }
    cleanup_stream(&front_camera_data.streaming.stream_ctx);

    // Cleanup bottom camera resources
    if (bottom_camera_data.streaming.frame != NULL) {
        image_free(bottom_camera_data.streaming.frame);
        free(bottom_camera_data.streaming.frame);
        bottom_camera_data.streaming.frame = NULL;
    }
    if (bottom_camera_data.streaming.stream_ctx.img_jpeg.buf != NULL) {
        image_free(&bottom_camera_data.streaming.stream_ctx.img_jpeg);
    }
    cleanup_stream(&bottom_camera_data.streaming.stream_ctx);

    // Destroy all mutexes
    pthread_mutex_destroy(&front_camera_data.processing.processing_mutex);
    pthread_mutex_destroy(&front_camera_data.streaming.streaming_mutex);
    pthread_mutex_destroy(&bottom_camera_data.processing.processing_mutex);
    pthread_mutex_destroy(&bottom_camera_data.streaming.streaming_mutex);

    // Cleanup inference system
    cleanup_inference();
}
