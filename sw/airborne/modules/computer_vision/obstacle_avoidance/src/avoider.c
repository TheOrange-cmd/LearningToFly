#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "generated/airframe.h"
#include "generated/flight_plan.h"
#include "inference.h"
#include "modules/core/abi.h"
#include "obstacle_avoider_avoider.h"
#include "state.h"
#include "std.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef SATURATE
#define SATURATE(x, min, max)                                                  \
  ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))
#endif

#define NUM_REGIONS 3         // ⚠️ Changed from 5 → 3 danger columns
#define SMOOTHING_FACTOR 0.3f // For trend calculation
#define AVOIDANCE_HISTORY_SIZE 4

#define MAX_FILTER_SIZE 30 //for the heading buffer filter
#define DEFAULT_FILTER_SIZE 5


// Initialize parameters with defaults
float oag_max_speed = 0.3f;
float oag_min_speed = 0.1f;
float oag_min_heading_rate = RadOfDeg(20.f);
float oag_max_heading_rate = RadOfDeg(60.f);
float obstacle_weight = 1.0f;
float floor_weight = 1.0f;
float danger_threshold = 0.4f;
float stop_danger_threshold = 0.9f;
uint8_t obstacle_filter_window = 3;
uint8_t boundary_filter_window = 1;
float oag_smoothing_factor = 0.3f;
float oag_trend_weight = 1.0f;
float danger_columns[NUM_REGIONS] = {0, 0, 0}; // ⚠️ Updated for 3 regions
float obstacle_free_confidence = 0;

float speed_sp = 0.0f;
float avoidance_heading_direction = 0;
bool use_border_detection = false;
float border_threshold = 0.5f;
bool use_heading_history = true;

float avoidance_heading_history[AVOIDANCE_HISTORY_SIZE] = {0.0f};
static int avoidance_history_index = 0;

//Heading buffer variables
typedef struct {
  float buffer[MAX_FILTER_SIZE]; 
  float weights[MAX_FILTER_SIZE]; 
  uint8_t max_size;
  uint8_t current_size;
  uint8_t index;
} HeadingFilter;

HeadingFilter heading_filter;


static struct timeval last_model_update_time;
static float last_avoidance_heading_direction = 0.0f;
static bool debug = true;

enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS,
  REENTER_ARENA
};

enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;

// Static variables
static struct filtered_data_t filtered_data = {0};
abi_event ev_model_output;
static bool avoider_enabled = false;
static float latest_floor_value = 0.0f;

// Time logging variables
static struct timeval start_time;
static struct timeval current_time;

// Debug configuration
#define DEBUG_TAG "AVOIDER"
#define MAX_LOG_LENGTH 256

static void debug_print(const char *format, ...) {
  va_list args;
  va_start(args, format);

#ifdef TARGET_AP
  char command[MAX_LOG_LENGTH + 32];
  vsnprintf(command, sizeof(command), format, args);
  snprintf(command, sizeof(command), "ulogger -t %s '%s'", DEBUG_TAG, command);
  system(command);
#else
  printf("[%s] ", DEBUG_TAG);
  vprintf(format, args);
  printf("\n");
  fflush(stdout);
#endif

  va_end(args);
}


//heading buffer filter
void heading_filter_init(HeadingFilter *filter, uint8_t size, float *custom_weights) {
  filter->max_size = (size > MAX_FILTER_SIZE) ? MAX_FILTER_SIZE : size;
  filter->current_size = 0;
  filter->index = 0;

  if (custom_weights) {
      memcpy(filter->weights, custom_weights, sizeof(float) * filter->max_size);
  } else {
      float sum_weights = 0.0f;
      for (uint8_t i = 0; i < filter->max_size; i++) {
          filter->weights[i] = expf(-0.3f * i);  // Exponential decay
          sum_weights += filter->weights[i];
      }
      // Normalize weights
      for (uint8_t i = 0; i < filter->max_size; i++) {
          filter->weights[i] /= sum_weights;
      }
  }
  
  memset(filter->buffer, 0, sizeof(float) * MAX_FILTER_SIZE);
}


float heading_filter_update(HeadingFilter *filter, float new_value) {
  filter->buffer[filter->index] = new_value;
  
  float weighted_sum = 0.0f;
  float weight_sum = 0.0f;
  uint8_t active_size = (filter->current_size < filter->max_size) 
                         ? filter->current_size 
                         : filter->max_size;

  for (uint8_t i = 0; i < active_size; i++) {
      uint8_t buf_index = (filter->index - i + filter->max_size) % filter->max_size;
      weighted_sum += filter->buffer[buf_index] * filter->weights[i];
      weight_sum += filter->weights[i];
  }

  filter->index = (filter->current_size < filter->max_size) ? filter->current_size++ : (filter->index + 1) % filter->max_size;
  if (filter->current_size < filter->max_size) {
      filter->current_size++;
  }

  if (weight_sum == 0) return new_value; // Avoid division by zero

  return weighted_sum / weight_sum;
}


// Callback function for processing model data
void myModelOutputHandler(uint8_t sender_id, uint32_t stamp,
                          unified_model_output_t *output) {
  if (!avoider_enabled) {
    return;
  }
  struct timeval now;
  gettimeofday(&now, NULL);

  // debug_print("Received model output from %d at time %u", sender_id, stamp);
  // Handle different model types
  if (output->type == 0) { // Obstacle detection model
    debug_print(
        "Received obstacle output from %d at time %u: [%.2f, %.2f, %.2f]",
        sender_id, stamp, output->data.obstacle.values[0][0],
        output->data.obstacle.values[0][1], output->data.obstacle.values[0][2]);

    // Find the highest danger column
    int max_index = 0;
    float max_value = output->data.obstacle.values[0][0];
    float min_value = output->data.obstacle.values[0][0];

    for (int i = 1; i < NUM_REGIONS; i++) {
      if (output->data.obstacle.values[0][i] > max_value) {
        max_value = output->data.obstacle.values[0][i];
        max_index = i;
      }
      else if (output->data.obstacle.values[0][i] < min_value) {
        min_value = output->data.obstacle.values[0][i];
      }      
    }

    float new_avoidance_heading_direction = 0.0f; // Default: Move forward
    float new_speed_sp = 0.0f;

    if(min_value < danger_threshold) {
      new_speed_sp = MAX(oag_max_speed - MAX(min_value - danger_threshold, 0) * (oag_max_speed - oag_min_speed), oag_min_speed);
      
      debug_print("No direct danger → setting speed to %.2f", new_speed_sp);
    }
    else if (min_value > stop_danger_threshold)
    {
      new_speed_sp = - oag_min_speed;
    }
    

    // Control movement based on the highest danger value
    if (max_value > danger_threshold) {
      if (max_index == 0) {
        debug_print("⚠️ Danger on LEFT → Steering RIGHT");
        new_avoidance_heading_direction = oag_max_heading_rate / 2;
      } else if (max_index == 1) {
        // new_avoidance_heading_direction = last_avoidance_heading_direction;
        // debug_print("⚠️ Danger CENTER → Steering like last time");
        if(output->data.obstacle.values[0][0] > output->data.obstacle.values[0][2]) {
          debug_print("⚠️ Danger CENTER → Steering RIGHT");
          new_avoidance_heading_direction = oag_max_heading_rate / 2;
        } else {
          debug_print("⚠️ Danger CENTER → Steering LEFT");
          new_avoidance_heading_direction = -oag_max_heading_rate / 2;
        }
      } else {
        debug_print("⚠️ Danger on RIGHT → Steering LEFT");
        new_avoidance_heading_direction = -oag_max_heading_rate / 2;
      }
    }


    // Store the latest avoidance direction and update timestamp

    //avoidance_heading_direction = new_avoidance_heading_direction;
  
    speed_sp = new_speed_sp;
    last_avoidance_heading_direction = avoidance_heading_direction;
    if(!use_heading_history){
      avoidance_heading_direction = new_avoidance_heading_direction;
    }
    else{
    avoidance_heading_direction = heading_filter_update(&heading_filter, new_avoidance_heading_direction); //heading buffer filter
    }
    last_model_update_time = now;
    
  } else if (output->type == 1) { // Border detection model
    if (debug) {
      debug_print("Received border output from %d at time %u: [%.2f]",
                  sender_id, stamp, output->data.border.value);
    }
    // Handle border detection logic
    if (use_border_detection) {

      if (output->data.border.value > border_threshold) {
        debug_print("⚠️ Border detected - Initiating turnaround");
        navigation_state = OUT_OF_BOUNDS;
        guidance_h_set_body_vel(0, 0);
      }
    }
  }
}

void obstacle_avoider_init(void) {
  heading_filter_init(&heading_filter, DEFAULT_FILTER_SIZE, NULL);  // Initialize buffer


  // Initialize structure
  filtered_data.current_index = 0;
  filtered_data.rows = 0;
  filtered_data.cols = 0;
  filtered_data.frames_processed = 0;

  // Initialize ABI message handling - bind to both sender types
  AbiBindMsgMODELOUTPUT(ABI_BROADCAST, &ev_model_output, myModelOutputHandler);

  debug_print("✅ Avoider initialized");
  gettimeofday(&start_time, NULL);
}

void start_avoider(void) {
  avoider_enabled = true;
  guidance_h_mode_changed(GUIDANCE_H_MODE_GUIDED);
  debug_print("🚀 Avoider enabled - Start flying!");
}

void stop_avoider(void) {
  avoider_enabled = false;
  // Reset state variables
  navigation_state = SEARCH_FOR_SAFE_HEADING;
  obstacle_free_confidence = 0;
  avoidance_heading_direction = 0;
  last_avoidance_heading_direction = 0.0f;

  // Reset filtered data
  filtered_data.current_index = 0;
  filtered_data.frames_processed = 0;

  // Log the cleanup
  debug_print("🛑 Avoider stopped and reset");
}

void obstacle_avoider_periodic(void) {
  if (!avoider_enabled) {
    return;
  }
  if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
    navigation_state = SEARCH_FOR_SAFE_HEADING;
    return;
  }

  struct timeval now;
  gettimeofday(&now, NULL);

  float elapsed_time =
      (now.tv_sec - last_model_update_time.tv_sec) +
      (now.tv_usec - last_model_update_time.tv_usec) / 1000000.0f;

  // If no new model updates in 1 second, gradually return to forward motion
  if (elapsed_time > 1.0) {
    avoidance_heading_direction *= 0.5f; // Gradual decay
    if (fabs(avoidance_heading_direction) < 0.01f) {
      avoidance_heading_direction = 0.0f; // Reset to straight movement
    }
  }


  switch (navigation_state) {
  case SAFE:
    if (!use_border_detection) {
      if (!InsideObstacleZone(stateGetPositionEnu_f()->x +
                                  0.4 * sinf(stateGetNedToBodyEulers_f()->psi),
                              stateGetPositionEnu_f()->y +
                                  0.4 *
                                      cosf(stateGetNedToBodyEulers_f()->psi))) {
        navigation_state = OUT_OF_BOUNDS;
      } else {
        guidance_h_set_body_vel(speed_sp, 0);
      }
      guidance_h_set_heading_rate(avoidance_heading_direction);
      break;
    } else {
      guidance_h_set_body_vel(speed_sp, 0);
      guidance_h_set_heading_rate(avoidance_heading_direction);
      break;
    }

  case SEARCH_FOR_SAFE_HEADING:
    navigation_state = SAFE;
    break;

  case OUT_OF_BOUNDS:
    guidance_h_set_body_vel(0, 0);
    guidance_h_set_heading_rate(RadOfDeg(60.f));
    navigation_state = REENTER_ARENA;
    debug_print("🔄 Re-entering arena...");
    break;

  case REENTER_ARENA:
    if (InsideObstacleZone(stateGetPositionEnu_f()->x +
                               1 * sinf(stateGetNedToBodyEulers_f()->psi),
                           stateGetPositionEnu_f()->y +
                               1 * cosf(stateGetNedToBodyEulers_f()->psi))) {
      obstacle_free_confidence += 1; // Increment confidence
    } else {
      obstacle_free_confidence = 0; // Reset if still outside
    }

    if (obstacle_free_confidence >
        5) { // Require multiple confirmations before switching to SAFE
      guidance_h_set_heading(stateGetNedToBodyEulers_f()->psi);
      obstacle_free_confidence = 0;
      navigation_state = SAFE;
    }
    break;

  default:
    break;
  }
}