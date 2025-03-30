#ifndef OBSTACLE_DETECTION_H
#define OBSTACLE_DETECTION_H

#include "modules/computer_vision/lib/vision/image.h"
#include "queue.h"
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/time.h>

#define MODEL_TYPE_OBSTACLE 0
#define MODEL_TYPE_BORDER 1

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
  struct processing_thread_t front_processor;
  struct processing_thread_t bottom_processor;
};

// Shared data for each camera
struct camera_processing_t {
  float *yuv_buffer;
  size_t yuv_buffer_size;
  pthread_mutex_t processing_mutex;
};

struct camera_data_t {
  struct camera_processing_t processing;
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
extern bool detection_debug;
extern bool detection_debug_model;
extern bool save_images;

// Function declarations
bool obstacle_detection_init(void);
void obstacle_detection_periodic(void);
void obstacle_detection_cleanup(void);

#endif // OBSTACLE_DETECTION_H