#ifndef OBSTACLE_DETECTION_H
#define OBSTACLE_DETECTION_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>  
#include <pthread.h>
#include "modules/computer_vision/lib/vision/image.h"
#include "video_stream.h"  

#define MODEL_TYPE_OBSTACLE 0
#define MODEL_TYPE_BORDER 1

// Queue structure for image processing
#define MAX_QUEUE_SIZE 4

struct image_queue_t {
    struct image_t* images[MAX_QUEUE_SIZE];
    int front;
    int rear;
    int size;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
};

// Thread control structure
struct processing_thread_t {
    pthread_t thread_id;
    bool running;
    struct image_queue_t queue;
};

// Configuration and state
struct obstacle_detection_t {
    bool front_enabled;
    bool bottom_enabled;
    bool stream_enabled;
    struct processing_thread_t front_processor;
    struct processing_thread_t bottom_processor;
};

// Shared data for each camera
struct camera_processing_t {
    float* rgb_buffer;
    size_t rgb_buffer_size;
    pthread_mutex_t processing_mutex;
};

struct camera_streaming_t {
    bool frame_ready;
    struct image_t* frame;
    struct stream_context_t stream_ctx;
    pthread_mutex_t streaming_mutex;
};

struct camera_data_t {
    struct camera_processing_t processing;
    struct camera_streaming_t streaming;
    uint32_t frames_received;
    uint32_t frames_processed;
    struct timeval last_frame_received_time;
    struct timeval last_frame_processed_time;
    float received_fps;
    float processed_fps;
};

// External declarations
extern struct obstacle_detection_t obstacle_detection;
extern struct camera_data_t front_camera_data;
extern struct camera_data_t bottom_camera_data;

// Function declarations
bool obstacle_detection_init(void);
void obstacle_detection_periodic(void);
void obstacle_detection_cleanup(void);

#endif // OBSTACLE_DETECTION_H