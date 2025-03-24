#include "video_stream.h"
#include "mcu_periph/udp.h"
#include "modules/computer_vision/lib/encoding/jpeg.h"
#include "modules/computer_vision/lib/encoding/rtp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef VIEWVIDEO_QUALITY_FACTOR
#define VIEWVIDEO_QUALITY_FACTOR 50
#endif

bool init_stream(struct stream_context_t *ctx, const char *ip, int port) {
  printf("[Stream] Initializing stream context\n");

  // Initialize all fields
  memset(ctx, 0, sizeof(struct stream_context_t));
  ctx->img_jpeg.buf = NULL; // Explicitly set buffer to NULL

  char ip_copy[128];
  strncpy(ip_copy, ip, sizeof(ip_copy) - 1);
  ip_copy[sizeof(ip_copy) - 1] = '\0';

  int ret = udp_socket_create(&ctx->video_sock, ip_copy, port, -1, FALSE);
  if (ret < 0) {
    printf("[Stream] Failed to create UDP socket (error: %d)\n", ret);
    return false;
  }
  printf("[Stream] UDP socket created successfully\n");
  return true;
}

// In video_stream.c
bool stream_frame(struct stream_context_t *ctx, struct image_t *original_img) {
  if (!ctx || !original_img || original_img->type != IMAGE_YUV422) {
    printf("[Stream] Invalid parameters for streaming\n");
    return false;
  }

  if (ctx->video_sock.sockfd <= 0) {
    printf("[Stream] Invalid socket\n");
    return false;
  }

  // Reallocate JPEG buffer if needed
  if (ctx->img_jpeg.w != original_img->w ||
      ctx->img_jpeg.h != original_img->h) {
    if (ctx->img_jpeg.buf != NULL) {
      image_free(&ctx->img_jpeg);
    }
    image_create(&ctx->img_jpeg, original_img->w, original_img->h, IMAGE_JPEG);
  }

  // printf("[Stream] Converting %dx%d YUV422 image to JPEG\n", original_img->w,
  // original_img->h);

  // Convert YUV422 to JPEG
  jpeg_encode_image(original_img, &ctx->img_jpeg, VIEWVIDEO_QUALITY_FACTOR, 0);

  // Check if JPEG encoding was successful
  if (!ctx->img_jpeg.buf || ctx->img_jpeg.buf_size == 0) {
    printf("[Stream] JPEG encoding failed - buffer: %p, size: %d\n",
           ctx->img_jpeg.buf, ctx->img_jpeg.buf_size);
    return false;
  }

  // printf("[Stream] JPEG encoding successful - size: %d bytes\n",
  // ctx->img_jpeg.buf_size);

  // Send using RTP
  rtp_frame_send(&ctx->video_sock, &ctx->img_jpeg,
                 0, // format code for YUV422
                 VIEWVIDEO_QUALITY_FACTOR,
                 0,     // no DRI header
                 30.0f, // full framerate of the bebop
                 &ctx->rtp_packet_nr, &ctx->rtp_frame_time);

  return true;
}

void cleanup_stream(struct stream_context_t *ctx) {
  if (ctx) {
    if (ctx->img_jpeg.buf) {
      free(ctx->img_jpeg.buf);
      ctx->img_jpeg.buf = NULL;
    }
    if (ctx->video_sock.sockfd > 0) {
      close(ctx->video_sock.sockfd);
    }
  }
}