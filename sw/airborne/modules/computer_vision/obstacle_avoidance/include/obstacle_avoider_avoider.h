// avoider.h
#ifndef OBSTACLE_AVOIDER_AVOIDER_H
#define OBSTACLE_AVOIDER_AVOIDER_H

#include "std.h"
#include <stdarg.h>

// Constants
#define FILTER_BUFFER_SIZE 10
#define MAX_OBSTACLE_DIMS 10
#define MODEL_TYPE_OBSTACLE 0
#define MODEL_TYPE_BORDER 1

// Tunable parameters (with defaults)
extern float oag_max_speed;
extern float oag_min_speed;
extern float oag_min_heading_rate;
extern float oag_max_heading_rate;
extern float obstacle_weight;
extern float floor_weight;
extern uint8_t obstacle_filter_window; 
extern uint8_t boundary_filter_window;
extern float danger_threshold;
extern float oag_smoothing_factor; 
extern float oag_trend_weight;
extern bool use_border_detection;
extern float border_threshold;

// Structure for filtered data
struct filtered_data_t {
    float* obstacle_values[MAX_OBSTACLE_DIMS][FILTER_BUFFER_SIZE];
    float floor_value[FILTER_BUFFER_SIZE];
    uint8_t current_index;
    uint8_t rows;
    uint8_t cols;
    uint32_t frames_processed; 
};

// Functions
extern void obstacle_avoider_init(void);
extern void obstacle_avoider_periodic(void);
extern void obstacle_avoider_cleanup(void);
void start_avoider(void);
void stop_avoider(void);

#endif