#include "queue.h"

bool queue_init(struct image_queue_t *q) {
  q->front = 0;
  q->rear = -1;
  q->size = 0;
  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->not_empty, NULL);
  pthread_cond_init(&q->not_full, NULL);
  return true;
}

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