// border_model_wrapper.c
#include "border_model_wrapper.h"
#include <stdio.h>

#define entry entry_border
#include "models/bottom_model.c"
#undef entry
