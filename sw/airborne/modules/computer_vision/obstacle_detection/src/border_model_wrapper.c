// border_model_wrapper.c
#include "border_model_wrapper.h"
#include <stdio.h>

#define entry entry_border
#include "models/border_model_16.c"
#undef entry
