/*
 * Copyright (C) Kirk Scheper <kirkscheper@gmail.com>
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider_guided.c"
 * @author Kirk Scheper
 * This module is an example module for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 * This module is used in combination with a color filter (cv_detect_color_object) and the guided mode of the autopilot.
 * The avoidance strategy is to simply count the total number of orange pixels. When above a certain percentage threshold,
 * (given by color_count_frac) we assume that there is an obstacle and we turn.
 *
 * The color filter settings are set using the cv_detect_color_object. This module can run multiple filters simultaneously
 * so you have to define which filter to use with the ORANGE_AVOIDER_VISUAL_DETECTION_ID setting.
 * This module differs from the simpler orange_avoider.xml in that this is flown in guided mode. This flight mode is
 * less dependent on a global positioning estimate as witht the navigation mode. This module can be used with a simple
 * speed estimate rather than a global position.
 *
 * Here we also need to use our onboard sensors to stay inside of the cyberzoo and not collide with the nets. For this
 * we employ a simple color detector, similar to the orange poles but for green to detect the floor. When the total amount
 * of green drops below a given threshold (given by floor_count_frac) we assume we are near the edge of the zoo and turn
 * around. The color detection is done by the cv_detect_color_object module, use the FLOOR_VISUAL_DETECTION_ID setting to
 * define which filter to use.
 */

#include "avoider.h"
#include "firmwares/rotorcraft/guidance/guidance_h.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <stdio.h>
#include <time.h>

#define DEBUG_TAG "AVOIDER"
#define MAX_LOG_LENGTH 256

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#ifndef MODELDATA_LISTENER_H
#define MODELDATA_LISTENER_H

void register_modeldata_listener(void);  // Function declaration

#endif  // MODELDATA_LISTENER_H

float danger_columns[5] = {0, 0, 0, 0, 0};

enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS,
  REENTER_ARENA
};

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

// define and initialise global variables
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;   // current state in state machine
int32_t color_count = 0;                // orange color count from color filter for obstacle detection
int32_t floor_count = 0;                // green color count from color filter for floor detection
int32_t floor_centroid = 0;             // floor detector centroid in y direction (along the horizon)
float avoidance_heading_direction = 0;  // heading change direction for avoidance [rad/s]
int16_t obstacle_free_confidence = 0;   // a measure of how certain we are that the way ahead if safe.
float oag_max_speed = 0.5f;               // max flight speed [m/s]
float oag_heading_rate = RadOfDeg(60.f);

const int16_t max_trajectory_confidence = 5;  // number of consecutive negative object detections to be sure we are obstacle free


// Callback function
void modeldata_message_handler(uint8_t sender_id, float v1, float v2, float v3, float v4, float v5) {
    debug_print("\nReceived MODELDATA message from sender %d: %f, %f, %f, %f, %f\n", 
           sender_id, v1, v2, v3, v4, v5);
    danger_columns[0] = v1;
    danger_columns[1] = v2;
    danger_columns[2] = v3;
    danger_columns[3] = v4;
    danger_columns[4] = v5;
}

// Function to register listener
void register_modeldata_listener(void) {
    static abi_event modeldata_event;
    AbiBindMsgMODELDATA(38, &modeldata_event, modeldata_message_handler);
}

/*
 * Initialisation function
 */
void orange_avoider_guided_init(void)
{
 register_modeldata_listener();
 debug_print("Avoider ready!");
}

/*
 * Function that checks it is safe to move forwards, and then sets a forward velocity setpoint or changes the heading
 */
void orange_avoider_guided_periodic(void)
{
  if (guidance_h.mode != GUIDANCE_H_MODE_GUIDED) {
    navigation_state = SEARCH_FOR_SAFE_HEADING;
    obstacle_free_confidence = 3;
    return;
  }

  // Bound obstacle_free_confidence
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  float speed_sp = oag_max_speed;

  // Find the highest danger column
  int max_danger_index = 0;
  float max_danger_value = danger_columns[0];
  for (int i = 1; i < 5; i++) {
    if (danger_columns[i] > max_danger_value) {
      max_danger_value = danger_columns[i];
      max_danger_index = i;
    }
  }

  // Determine heading direction based on the most dangerous side
  if (max_danger_index < 2) {
    // Danger is more on the left, turn right
    avoidance_heading_direction = oag_heading_rate;
  } else if (max_danger_index > 2) {
    // Danger is more on the right, turn left
    avoidance_heading_direction = -oag_heading_rate;
  } else {
    // Danger is in the center, stop and turn in place
    avoidance_heading_direction = oag_heading_rate;
  }
  
  guidance_h_set_body_vel(speed_sp, 0);
  guidance_h_set_heading_rate(avoidance_heading_direction * RadOfDeg(15));

  return;
}

