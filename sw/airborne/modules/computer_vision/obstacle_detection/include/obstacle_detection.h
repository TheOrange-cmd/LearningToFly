#ifndef OBSTACLE_DETECTION_H
#define OBSTACLE_DETECTION_H

#include <stdbool.h>
#include <stdint.h>

// Bools for enabling and disabling obstacle detection and streaming
struct obstacle_detection_t {
    bool enabled;
    bool stream_enabled;
};

// External declaration
extern struct obstacle_detection_t obstacle_detection;

struct shared_data_t {
    bool frame_ready;
    struct image_t* frame;
    uint32_t frames_received;
    uint32_t frames_processed;
    struct timeval last_frame_time;
};

// Function declarations
extern struct shared_data_t shared_data;
extern bool obstacle_detection_init(void);
extern void obstacle_detection_periodic(void);
extern void obstacle_detection_cleanup(void);
struct image_t* video_callback(struct image_t *img, uint8_t camera_id);  
#endif // OBSTACLE_DETECTION_H