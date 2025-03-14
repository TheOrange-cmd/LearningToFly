#include "obstacle_avoider_avoider.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <stdio.h>
#include "std.h" 
#include <stdlib.h>

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef SATURATE
#define SATURATE(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#endif

#define NUM_REGIONS 5  // We have 5 regions from left to right
#define SMOOTHING_FACTOR 0.3f  // For trend calculation

// Initialize parameters with defaults
float oag_max_speed = 0.5f;
float oag_min_speed = 0.1f;
float oag_min_heading_rate = RadOfDeg(20.f);
float oag_max_heading_rate = RadOfDeg(60.f);
float obstacle_weight = 1.0f;
float floor_weight = 1.0f;
float danger_threshold = 0.8f;
uint8_t obstacle_filter_window = 3;  
uint8_t boundary_filter_window = 1; 
float oag_smoothing_factor = 0.3f;
float oag_trend_weight = 1.0f;


// Static variables
static struct filtered_data_t filtered_data = {0};
static abi_event modeldata_event;
static bool initialized = false;
static bool avoider_enabled = false;
static float latest_floor_value = 0.0f;

// Debug configuration
#define DEBUG_TAG "AVOIDER"
#define MAX_LOG_LENGTH 256

static void modeldata_handler(uint8_t sender_id, uint8_t output_type, uint8_t rows, uint8_t cols, float* values);

static void debug_print(const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    #ifdef TARGET_AP
        // On actual drone, use ulogger
        char command[MAX_LOG_LENGTH + 32];
        vsnprintf(command, sizeof(command), format, args);
        snprintf(command, sizeof(command), "ulogger -t %s '%s'", DEBUG_TAG, command);
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

// Filter function
static float moving_average(float values[], uint8_t window) {
    float sum = 0;
    for (uint8_t i = 0; i < window; i++) {
        sum += values[i];
    }
    return sum / window;
}

struct region_danger {
    float danger_level;
    float trend;          // Rate of change in danger
    float previous_danger;
    float weight;         // Weight for steering calculation (negative for left, positive for right)
};

static struct region_danger regions[NUM_REGIONS];

void init_region_weights(void) {
    // Initialize weights from left to right: [-1.0, -0.5, 0.0, 0.5, 1.0]
    for (int i = 0; i < NUM_REGIONS; i++) {
        regions[i].weight = -1.0f + (2.0f * i / (NUM_REGIONS - 1));
    }
}

float calculate_region_danger(int start_col, int end_col, uint8_t filter_window) {
    float danger = 0;
    int count = 0;
    
    for (int i = 0; i < filtered_data.rows; i++) {
        for (int j = start_col; j < end_col; j++) {
            for (int k = 0; k < filter_window; k++) {
                int idx = (filtered_data.current_index - k + FILTER_BUFFER_SIZE) % FILTER_BUFFER_SIZE;
                danger += filtered_data.obstacle_values[i][idx][j];
                count++;
            }
        }
    }
    
    return (count > 0) ? danger / count : 0;
}

float calculate_steering_command(float *speed_sp, float *heading_rate_sp) {
    // Check if we're near testing area boundary first - with shorter filter
    float boundary_danger = latest_floor_value;

    // If we're near boundary, override normal obstacle avoidance
    if (boundary_danger > danger_threshold) {
        *heading_rate_sp = oag_max_heading_rate;
        *speed_sp = oag_min_speed;
        debug_print("BOUNDARY DETECTED: %.2f - Turning around", boundary_danger);
        return boundary_danger;
    }

    // Normal obstacle avoidance when within testing area
    int cols_per_region = filtered_data.cols / NUM_REGIONS;
    float weighted_danger_sum = 0;
    float total_danger = 0;
    float max_danger = 0;
    
    // Calculate dangers and trends for each region - with original filter window
    for (int r = 0; r < NUM_REGIONS; r++) {
        int start_col = r * cols_per_region;
        int end_col = (r == NUM_REGIONS-1) ? filtered_data.cols : (r + 1) * cols_per_region;
        
        regions[r].previous_danger = regions[r].danger_level;
        regions[r].danger_level = calculate_region_danger(start_col, end_col, obstacle_filter_window) * obstacle_weight;
        
        regions[r].trend = oag_smoothing_factor * (regions[r].danger_level - regions[r].previous_danger) + 
                          (1 - oag_smoothing_factor) * regions[r].trend;
        
        float effective_danger = regions[r].danger_level + (regions[r].trend * oag_trend_weight);
        
        weighted_danger_sum += effective_danger * regions[r].weight;
        total_danger += effective_danger;
        max_danger = MAX(max_danger, effective_danger);
    }

    // Calculate steering direction for obstacle avoidance
    float steering_direction = 0;
    if (total_danger > 0) {
        steering_direction = weighted_danger_sum / total_danger;
    }

    // Calculate heading rate based on obstacle danger
    float danger_scale = (max_danger > danger_threshold) ? 
        (max_danger - danger_threshold) / (1.0f - danger_threshold) : 
        (max_danger / danger_threshold);
    
    float base_rate = oag_min_heading_rate + 
        (oag_max_heading_rate - oag_min_heading_rate) * danger_scale;
    
    *heading_rate_sp = base_rate * steering_direction;

    // Calculate speed
    *speed_sp = oag_max_speed;
    if (max_danger > 0) {
        float center_trend = regions[NUM_REGIONS/2].trend;
        float speed_reduction = danger_scale + MAX(0, center_trend * oag_trend_weight);
        speed_reduction = SATURATE(speed_reduction, 0, 1);
        
        *speed_sp = oag_max_speed * (1.0f - speed_reduction);
        *speed_sp = MAX(*speed_sp, oag_min_speed);
    }

    debug_print("Boundary:%.2f Obstacles:%.2f %.2f %.2f %.2f %.2f Dir:%.2f Spd:%.2f Rate:%.2f", 
                boundary_danger,
                regions[0].danger_level, 
                regions[1].danger_level,
                regions[2].danger_level,
                regions[3].danger_level,
                regions[4].danger_level,
                steering_direction,
                *speed_sp, 
                *heading_rate_sp);

    return max_danger;
}

// Callback function for model data
void modeldata_handler(uint8_t sender_id, uint8_t output_type, uint8_t rows, uint8_t cols, float* values) {
    if (!initialized) return;

    if (output_type == MODEL_TYPE_OBSTACLE) {
        // printf("[Avoider] Obstacle data received\n");
        // Reallocate filtered data if dimensions change
        if (filtered_data.rows != rows || filtered_data.cols != cols) {
            debug_print("Updating grid dimensions from %dx%d to %dx%d", 
                       filtered_data.rows, filtered_data.cols, rows, cols);

            // Free existing arrays if they exist
            if (filtered_data.obstacle_values[0][0] != NULL) {
                for (int i = 0; i < filtered_data.rows; i++) {
                    for (int j = 0; j < FILTER_BUFFER_SIZE; j++) {
                        free(filtered_data.obstacle_values[i][j]);
                    }
                }
            }
            
            // Update dimensions
            filtered_data.rows = rows;
            filtered_data.cols = cols;
            
            // Allocate new arrays
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < FILTER_BUFFER_SIZE; j++) {
                    filtered_data.obstacle_values[i][j] = calloc(cols, sizeof(float));
                    if (filtered_data.obstacle_values[i][j] == NULL) {
                        debug_print("Failed to allocate memory for obstacle values");
                        return;
                    }
                }
            }
        }
        
        // Update values in current slot
        uint8_t idx = filtered_data.current_index;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                filtered_data.obstacle_values[i][idx][j] = values[i * cols + j];
            }
        }
    }
    else if (output_type == MODEL_TYPE_BORDER) {
        debug_print("Raw border value received: %.6f", values[0]);
        latest_floor_value = values[0] * floor_weight;
        debug_print("Weighted border value: %.6f", latest_floor_value);
    }
    
    // Update counters
    filtered_data.frames_processed++;
    filtered_data.current_index = (filtered_data.current_index + 1) % FILTER_BUFFER_SIZE;
}

void obstacle_avoider_init(void) {
    // Initialize structure
    filtered_data.current_index = 0;
    filtered_data.rows = 0;
    filtered_data.cols = 0;
    filtered_data.frames_processed = 0;

    // Clear floor values
    for (int i = 0; i < FILTER_BUFFER_SIZE; i++) {
        filtered_data.floor_value[i] = 0;
    }

    // Initialize obstacle values to NULL
    for (int i = 0; i < MAX_OBSTACLE_DIMS; i++) {
        for (int j = 0; j < FILTER_BUFFER_SIZE; j++) {
            filtered_data.obstacle_values[i][j] = NULL;
        }
    }
    
    // Register ABI listener
    AbiBindMsgMODELDATA(ABI_BROADCAST, &modeldata_event, modeldata_handler);
    
    initialized = true;
    debug_print("Avoider initialized");
}

void start_avoider(void) {
    avoider_enabled = true;
    guidance_h_mode_changed(GUIDANCE_H_MODE_GUIDED);
    debug_print("Avoider enabled - start flying!");
}

void obstacle_avoider_cleanup(void) {
    // Free allocated memory
    if (filtered_data.obstacle_values[0][0] != NULL) {
        for (int i = 0; i < filtered_data.rows; i++) {
            for (int j = 0; j < FILTER_BUFFER_SIZE; j++) {
                free(filtered_data.obstacle_values[i][j]);
            }
        }
    }
    initialized = false;
    avoider_enabled = false;
}

void obstacle_avoider_periodic(void) {
    if (!avoider_enabled) {
        return;
    }
    
    float speed_sp, heading_rate;
    calculate_steering_command(&speed_sp, &heading_rate);
    
    guidance_h_set_body_vel(speed_sp, 0);
    guidance_h_set_heading_rate(heading_rate);
}