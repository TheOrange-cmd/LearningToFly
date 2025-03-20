#ifndef QUEUE_H
#define QUEUE_H

#include <pthread.h>
#include <stdbool.h>
#include "modules/computer_vision/lib/vision/image.h"

#define MAX_QUEUE_SIZE 10

struct image_queue_t {
    struct image_t* images[MAX_QUEUE_SIZE];
    int front;
    int rear;
    int size;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty;
    pthread_cond_t not_full;
};

bool queue_init(struct image_queue_t* q);
bool queue_push(struct image_queue_t* q, struct image_t* img);
struct image_t* queue_pop(struct image_queue_t* q);

#endif // QUEUE_H