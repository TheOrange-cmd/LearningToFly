#ifndef VIDEO_STREAM_H
#define VIDEO_STREAM_H

#include "mcu_periph/udp.h" // For struct UdpSocket
#include "modules/computer_vision/lib/encoding/rtp.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "std.h"

struct stream_context_t {
  struct UdpSocket video_sock;
  uint16_t rtp_packet_nr;
  uint32_t rtp_frame_time;
  struct image_t img_jpeg;
};

bool init_stream(struct stream_context_t *ctx, const char *ip, int port);
bool stream_frame(struct stream_context_t *ctx, struct image_t *frame);
void cleanup_stream(struct stream_context_t *ctx);

#endif