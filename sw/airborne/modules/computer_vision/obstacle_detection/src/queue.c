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
 * sw/airborne/modules/computer_vision/obstacle_detection/src/queue.c
 *
 * @brief Queue functions for vision processing
 *
 * @note Developement assisted by Claude Sonnet 3.5
 */

#include "queue.h"

/**
 * @brief Initialize a thread-safe image queue
 *
 * Initializes queue structure with empty state and synchronization primitives.
 * Must be called before using the queue.
 *
 * @param q Queue structure to initialize
 * @return true Always returns true
 */
bool queue_init(struct image_queue_t *q) {
  q->front = 0;
  q->rear = -1;
  q->size = 0;
  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->not_empty, NULL);
  pthread_cond_init(&q->not_full, NULL);
  return true;
}

/**
 * @brief Push an image frame into the queue
 *
 * Thread-safe operation that adds image to queue. If queue is full,
 * drops oldest frame to make space (frame count limited by MAX_QUEUE_SIZE).
 * Signals waiting consumers when new data is available.
 *
 * @param q Queue structure to modify
 * @param img Image to add to queue (will be copied)
 * @return true On successful push (always succeeds, may drop old frames)
 */
bool queue_push(struct image_queue_t *q, struct image_t *img) {
  pthread_mutex_lock(&q->mutex);

  // If queue is full, drop the oldest frame
  if (q->size >= MAX_QUEUE_SIZE) {
    struct image_t *old_img = q->images[q->front];
    if (old_img) {
      image_free(old_img);
      free(old_img);
    }

    q->front = (q->front + 1) % MAX_QUEUE_SIZE;
    q->size--;
  }

  q->rear = (q->rear + 1) % MAX_QUEUE_SIZE;
  q->images[q->rear] = img;
  q->size++;

  pthread_cond_signal(&q->not_empty);
  pthread_mutex_unlock(&q->mutex);
  return true;
}

/**
 * @brief Pop an image frame from the queue
 *
 * Thread-safe blocking operation that removes and returns the oldest image.
 * Will wait indefinitely if queue is empty until new frames arrive.
 *
 * @param q Queue structure to read from
 * @return struct image_t* Popped image pointer, NULL only if interrupted
 */
struct image_t *queue_pop(struct image_queue_t *q) {
  pthread_mutex_lock(&q->mutex);

  while (q->size == 0) {
    pthread_cond_wait(&q->not_empty, &q->mutex);
  }

  struct image_t *img = q->images[q->front];
  q->front = (q->front + 1) % MAX_QUEUE_SIZE;
  q->size--;

  pthread_cond_signal(&q->not_full);
  pthread_mutex_unlock(&q->mutex);
  return img;
}