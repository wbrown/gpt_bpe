#ifndef GPT_BPE_LIB_H
#define GPT_BPE_LIB_H

#include <stdlib.h>
#include <stdint.h>

typedef struct {
	uint32_t *tokens;
	size_t len;
} Tokens;

#endif