#ifndef OBSTACLE_DETECTION_H
#define OBSTACLE_DETECTION_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>  
#include "modules/computer_vision/lib/vision/image.h"
#include "video_stream.h"  

// Configuration and state
struct obstacle_detection_t {
    bool front_enabled;
    bool bottom_enabled;
    bool stream_enabled;
};

// Shared data for each camera
struct camera_data_t {
    bool frame_ready;
    struct image_t* frame;
    uint32_t frames_received;
    uint32_t frames_processed;
    struct timeval last_frame_received_time;
    struct timeval last_frame_processed_time;
    float received_fps;
    float processed_fps;
    struct stream_context_t stream_ctx;
    pthread_mutex_t frame_mutex;
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