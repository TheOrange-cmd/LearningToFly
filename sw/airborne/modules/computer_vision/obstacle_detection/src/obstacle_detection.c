/**
 *
 * Copyright (C) 2025 Daniel Rugge <d.j.rugge@student.tudelft.nl>
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */
/**
 * @file
 * sw/airborne/modules/computer_vision/obstacle_detection/src/obstacle_detection.c
 *
 * Main file for the sensing part of the two part module for obstacle detection
 * and avoidance for the TU Delft Autonomous MAV course.
 *
 * The sensing part uses the front camera to detect obstacles and the bottom
 * camera to detect the boundary of the cyberzoo. The module uses neural network
 * models to perform the detection and sends the results to the autopilot using
 * the ABI. The detector using the front camera is trained using MiDaS
 * (https://arxiv.org/abs/1907.01341) generated depth maps that were reduced to
 * simplified danger values in three column regions in the center of the image.
 * The bottom module is trained using a small amount of manually labeled data.
 *
 * @note Developement assisted by Claude Sonnet 3.5
 */

// Standard includes
#include <errno.h>     // For errno
#include <pthread.h>   // For pthreads and mutexes
#include <stdbool.h>   // For bool type
#include <stdio.h>     // For printf
#include <stdlib.h>    // For malloc
#include <string.h>    // For strerror
#include <sys/stat.h>  // For mkdir and struct stat
#include <sys/time.h>  // For gettimeofday
#include <sys/types.h> // For struct stat
#include <unistd.h>    // For getcwd

#include "debug_print.h"
DEFINE_DEBUG_PRINT("OBSDET")

// Paparazzi includes
#include "lib/v4l/v4l2.h"
#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/encoding/jpeg.h"
#include "modules/computer_vision/lib/encoding/rtp.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "modules/core/abi.h"

// Project includes
#include "image_convert.h"      // For YUV to RGB conversion
#include "image_utils.h"        // For image saving
#include "inference.h"          // For running inference
#include "model.h"              // For model dimensions
#include "obstacle_detection.h" // For structs and function declarations
#include "queue.h"              // For processing queue

// ABI message includes
#include "mcu_periph/udp.h"
#include "udp_socket.h"

// Debug configuration
#define DEBUG_TAG "OBSDET"
#define MAX_LOG_LENGTH 256

// Forward declarations
static struct image_t *front_camera_callback(struct image_t *img,
                                             uint8_t camera_id);
static struct image_t *bottom_camera_callback(struct image_t *img,
                                              uint8_t camera_id);

// Settings to disable processing for each camera (exposed in GCS)
struct obstacle_detection_t obstacle_detection = {.front_enabled = true,
                                                  .bottom_enabled = true};

// bools for debug prints and image saving (exposed in GCS)
bool detection_debug = false;
bool detection_debug_model = true;
bool save_images = false;

// Camera data structures
struct camera_data_t front_camera_data = {0};
struct camera_data_t bottom_camera_data = {0};
static struct video_listener *front_video_listener = NULL;
static struct video_listener *bottom_video_listener = NULL;

/**
 * @brief Front camera processing thread function
 *
 * This thread continuously processes images from the front camera queue.
 * Performs image conversion, runs obstacle detection inference, and sends
 * results via ABI messages. Times and logs processing stages when debug
 * enabled.
 *
 * @param arg Pointer to processing_thread_t structure containing thread context
 * @return NULL on thread exit
 */
static void *front_processing_thread(void *arg) {
  struct processing_thread_t *proc = (struct processing_thread_t *)arg;

  while (proc->running) {
    struct image_t *img = queue_pop(&proc->queue);
    if (img) {
      struct timeval t1, t2, t3;
      gettimeofday(&t1, NULL);

      // Save raw input image if flag is set
      if (save_images) {
        char *filename = generate_unique_filename("front_raw", "yuv");
        if (filename) {
          save_yuv_image(img->buf, img->w, img->h, filename);
          free(filename);
        }
      }

      bool conv_success = convert_uyvy_to_yuv_crop_with_scale(
          img->buf, img->w, img->h, 240, 240,
          front_camera_data.processing.yuv_buffer,
          front_camera_data.processing.yuv_buffer_size,
          240 / FRONT_MODEL_INPUT_SIZE);

      if (conv_success) {
        // Save converted image if flag is set
        if (save_images) {
          char *filename = generate_unique_filename("front_converted", "yuv");
          if (filename) {
            save_yuv_image((uint8_t *)front_camera_data.processing.yuv_buffer,
                           240, 240, filename);
            free(filename);
          }
        }
        gettimeofday(&t2, NULL);
        struct model_output_t model_output;
        bool inf_success = run_obstacle_inference(
            front_camera_data.processing.yuv_buffer, FRONT_MODEL_INPUT_SIZE,
            FRONT_MODEL_INPUT_SIZE, &model_output);
        if (inf_success) {
          unified_model_output_t unified_output;
          unified_output.type = 0; // 0 for obstacle detection

          // Copy values
          for (int i = 0; i < FRONT_MODEL_OUTPUT_ROW_SIZE; i++) {
            for (int j = 0; j < FRONT_MODEL_OUTPUT_COL_SIZE; j++) {
              unified_output.data.obstacle.values[i][j] =
                  model_output.values[i][j];
            }
          }

          // Send ABI message
          AbiSendMsgMODELOUTPUT(1, get_sys_time_usec(), &unified_output);

          if (detection_debug_model) {
            debug_print("Front model outputs: %.2f, %.2f, %.2f",
                        unified_output.data.obstacle.values[0][0],
                        unified_output.data.obstacle.values[0][1],
                        unified_output.data.obstacle.values[0][2]);
          }
        } else {
          debug_print("Failed to run front inference");
        }
        gettimeofday(&t3, NULL);

        float conv_time = (t2.tv_sec - t1.tv_sec) * 1000.0f +
                          (t2.tv_usec - t1.tv_usec) / 1000.0f;
        float inf_time = (t3.tv_sec - t2.tv_sec) * 1000.0f +
                         (t3.tv_usec - t2.tv_usec) / 1000.0f;
        if (detection_debug) {
          debug_print(
              "Front processing times - Convert: %.1fms, Inference: %.1fms",
              conv_time, inf_time);
        }
      } else {
        debug_print("Failed to convert front UYVY to YUV");
      }

      image_free(img);
      free(img);
    }
  }
  return NULL;
}

/**
 * @brief Front camera image capture callback
 *
 * Receives images from front camera, copies them to processing queue.
 * Logs timing statistics when debug enabled.
 *
 * @param img Captured image pointer
 * @param camera_id Unused camera identifier
 * @return NULL as image is consumed by processing thread
 */
struct image_t *front_camera_callback(struct image_t *img, uint8_t camera_id
                                      __attribute__((unused))) {
  static uint32_t callback_count = 0;
  static struct timeval last_callback = {0, 0};
  struct timeval now;
  gettimeofday(&now, NULL);
  callback_count++;

  if (detection_debug) {
    if (last_callback.tv_sec != 0) {
      float dt = (now.tv_sec - last_callback.tv_sec) * 1000.0f +
                 (now.tv_usec - last_callback.tv_usec) / 1000.0f;
      debug_print("Front callback %d - Time since last: %.1fms", callback_count,
                  dt);
    }
    last_callback = now;
  }

  if (obstacle_detection.front_enabled) {
    struct image_t *proc_img = malloc(sizeof(struct image_t));
    if (proc_img) {
      image_create(proc_img, img->w, img->h, img->type);
      image_copy(img, proc_img);
      queue_push(&obstacle_detection.front_processor.queue, proc_img);
    }
  }

  return NULL;
}

/**
 * @brief Bottom camera image capture callback
 *
 * Receives images from bottom camera, copies them to processing queue.
 * Logs timing statistics when debug enabled.
 *
 * @param img Captured image pointer
 * @param camera_id Unused camera identifier
 * @return NULL as image is consumed by processing thread
 */
struct image_t *bottom_camera_callback(struct image_t *img, uint8_t camera_id
                                       __attribute__((unused))) {
  static uint32_t callback_count = 0;
  static struct timeval last_callback = {0, 0};
  struct timeval now;
  if (detection_debug) {
    gettimeofday(&now, NULL);

    callback_count++;

    if (last_callback.tv_sec != 0) {
      float dt = (now.tv_sec - last_callback.tv_sec) * 1000.0f +
                 (now.tv_usec - last_callback.tv_usec) / 1000.0f;
      debug_print("Bottom callback %d - Time since last: %.1fms",
                  callback_count, dt);
    }
    last_callback = now;
  }

  if (obstacle_detection.bottom_enabled) {
    struct image_t *proc_img = malloc(sizeof(struct image_t));
    if (proc_img) {
      image_create(proc_img, img->w, img->h, img->type);
      image_copy(img, proc_img);
      queue_push(&obstacle_detection.bottom_processor.queue, proc_img);
    }
  }

  return NULL;
}

/**
 * @brief Bottom camera processing thread function
 *
 * This thread continuously processes images from the bottom camera queue.
 * Performs image conversion, runs border detection inference, and sends
 * results via ABI messages. Times and logs processing stages when debug
 * enabled.
 *
 * @param arg Pointer to processing_thread_t structure containing thread context
 * @return NULL on thread exit
 */
static void *bottom_processing_thread(void *arg) {
  struct processing_thread_t *proc = (struct processing_thread_t *)arg;
  static uint32_t process_count = 0;
  struct timeval t1, t2, t3;
  while (proc->running) {
    struct image_t *img = queue_pop(&proc->queue);
    if (img) {
      process_count++;
      if (detection_debug) {
        debug_print("Bottom processing - Frame %d", process_count);
        gettimeofday(&t1, NULL);
      }

      // Save raw input image if flag is set
      if (save_images) {
        char *filename = generate_unique_filename("bottom_raw", "yuv");
        if (filename) {
          save_yuv_image(img->buf, img->w, img->h, filename);
          free(filename);
        }
      }

      bool conv_success = convert_uyvy_to_yuv_downscale(
          img->buf, img->w, img->h, bottom_camera_data.processing.yuv_buffer,
          bottom_camera_data.processing.yuv_buffer_size,
          240 / BOTTOM_MODEL_INPUT_SIZE);
      if (detection_debug) {
        gettimeofday(&t2, NULL);
      }

      if (conv_success) {
        if (save_images) {
          char *filename = generate_unique_filename("bottom_converted", "yuv");
          if (filename) {
            save_yuv_image((uint8_t *)bottom_camera_data.processing.yuv_buffer,
                           240, 240, filename);
            free(filename);
          }
        }
        struct border_output_t border_output;
        bool inf_success = run_border_inference(
            bottom_camera_data.processing.yuv_buffer, BOTTOM_MODEL_INPUT_SIZE,
            BOTTOM_MODEL_INPUT_SIZE, &border_output);
        if (inf_success) {
          unified_model_output_t unified_output;
          unified_output.type = 1; // 1 for border detection
          unified_output.data.border.value = border_output.value;

          // Send ABI message
          AbiSendMsgMODELOUTPUT(2, get_sys_time_usec(), &unified_output);

          if (detection_debug_model) {
            debug_print("Bottom model output: %.2f",
                        unified_output.data.border.value);
          }
        }
        if (detection_debug) {
          gettimeofday(&t3, NULL);

          float conv_time = (t2.tv_sec - t1.tv_sec) * 1000.0f +
                            (t2.tv_usec - t1.tv_usec) / 1000.0f;
          float inf_time = (t3.tv_sec - t2.tv_sec) * 1000.0f +
                           (t3.tv_usec - t2.tv_usec) / 1000.0f;

          debug_print(
              "Bottom processing times - Convert: %.1fms, Inference: %.1fms",
              conv_time, inf_time);
        }
      }
      image_free(img);
      free(img);
    }
  }
  return NULL;
}

/**
 * @brief Initialize the obstacle detection module
 *
 * Starts processing threads, initializes synchronization primitives,
 * allocates image buffers, registers camera callbacks, and initializes
 * neural network inference system.
 *
 * @return true if initialization succeeded
 * @return false if any initialization step failed
 */
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
                 bottom_processing_thread,
                 &obstacle_detection.bottom_processor);

  // Initialize mutexes for both cameras
  if (pthread_mutex_init(&front_camera_data.processing.processing_mutex,
                         NULL) != 0 ||
      pthread_mutex_init(&bottom_camera_data.processing.processing_mutex,
                         NULL) != 0) {
    debug_print("Failed to initialize mutexes");
    return false;
  }

  // Allocate YUV buffers
  front_camera_data.processing.yuv_buffer_size =
      (size_t)FRONT_MODEL_INPUT_SIZE * FRONT_MODEL_INPUT_SIZE * 3 *
      sizeof(float);
  size_t bottom_size = (size_t)BOTTOM_MODEL_INPUT_SIZE *
                       BOTTOM_MODEL_INPUT_SIZE * 3 * sizeof(float);
  debug_print("Allocating bottom camera YUV buffer: %dx%d = %zu bytes",
              BOTTOM_MODEL_INPUT_SIZE, BOTTOM_MODEL_INPUT_SIZE, bottom_size);
  debug_print("Allocating front camera YUV buffer: %dx%d = %zu bytes",
              FRONT_MODEL_INPUT_SIZE, FRONT_MODEL_INPUT_SIZE,
              front_camera_data.processing.yuv_buffer_size);

  front_camera_data.processing.yuv_buffer =
      malloc(front_camera_data.processing.yuv_buffer_size);
  bottom_camera_data.processing.yuv_buffer_size = bottom_size;
  bottom_camera_data.processing.yuv_buffer = malloc(bottom_size);

  if (!bottom_camera_data.processing.yuv_buffer) {
    debug_print("Failed to allocate bottom camera RGB buffer (%zu bytes)",
                bottom_size);
    return false;
  }

  if (!front_camera_data.processing.yuv_buffer) {
    debug_print("Failed to allocate front camera RGB buffers");
    cleanup_inference();
    return false;
  }

  front_video_listener = cv_add_to_device(
      &OBSTACLE_DETECTION_FRONT_CAMERA, front_camera_callback, 10,
      0); // requesting 10 fps, though it seems to be ignored
  bottom_video_listener = cv_add_to_device(
      &OBSTACLE_DETECTION_BOTTOM_CAMERA, bottom_camera_callback, 10,
      0); // requesting 10 fps, though it seems to be ignored

  if (front_video_listener == NULL || bottom_video_listener == NULL) {
    debug_print("Failed to register video callbacks");
    obstacle_detection_cleanup();
    return false;
  } else {
    debug_print("Registered video callbacks");
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

/**
 * @brief Periodic function for obstacle detection module
 *
 * Currently unused placeholder for periodic tasks as all periodic tasks are
 * handled in the cv vision threads. Model outputs are immediately sent to the
 * avoider module.
 */
void obstacle_detection_periodic(void) {}

/**
 * @brief Cleanup and shutdown obstacle detection module
 *
 * Stops processing threads, frees allocated memory, releases synchronization
 * primitives, and cleans up inference resources. Called in case initialization
 * fails.
 */
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

  // Free YUV buffers
  free(front_camera_data.processing.yuv_buffer);
  free(bottom_camera_data.processing.yuv_buffer);
  front_camera_data.processing.yuv_buffer = NULL;
  bottom_camera_data.processing.yuv_buffer = NULL;

  // Destroy mutexes
  pthread_mutex_destroy(&front_camera_data.processing.processing_mutex);
  pthread_mutex_destroy(&bottom_camera_data.processing.processing_mutex);

  // Cleanup inference system
  cleanup_inference();
}
