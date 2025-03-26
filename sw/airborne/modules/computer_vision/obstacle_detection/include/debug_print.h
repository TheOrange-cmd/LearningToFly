#ifndef DEBUG_PRINT_H
#define DEBUG_PRINT_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_LOG_LENGTH 256

// Generic debug print function that takes a tag
static inline void debug_print_tag(const char *tag, const char *format, ...) {
  va_list args;
  va_start(args, format);

#ifdef TARGET_AP
  char buffer[MAX_LOG_LENGTH];
  vsnprintf(buffer, sizeof(buffer), format, args);
  char command[MAX_LOG_LENGTH + 32];
  snprintf(command, sizeof(command), "ulogger -t %s '%s'", tag, buffer);
  system(command);
#else
  printf("[%s] ", tag);
  vprintf(format, args);
  printf("\n");
  fflush(stdout);
#endif

  va_end(args);
}

// Macro to create a debug_print function for a specific tag
#define DEFINE_DEBUG_PRINT(tag)                                                \
  static inline void debug_print(const char *format, ...) {                    \
    va_list args;                                                              \
    va_start(args, format);                                                    \
    char buffer[MAX_LOG_LENGTH];                                               \
    vsnprintf(buffer, sizeof(buffer), format, args);                           \
    va_end(args);                                                              \
    debug_print_tag(tag, "%s", buffer);                                        \
  }

#endif // DEBUG_PRINT_H