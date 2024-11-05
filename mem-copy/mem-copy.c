#define AIL_ALL_IMPL
#define AIL_BENCH_IMPL
#define AIL_BENCH_PROFILE
#include "../util/ail/ail.h"       // For typedefs and some useful macros
#include "../util/ail/ail_alloc.h" // For allocation
#include "../util/ail/ail_bench.h" // For benchmarking
#include <stdio.h>                 // For printf
#include <string.h>                // For memset
#include <xmmintrin.h>             // For SIMD instructions

#if 0
#   define TEST
#else
#   define ALL
#endif

#if 0
#   define MIN_BUFFER_SIZE AIL_KB(512)
#   define MAX_BUFFER_SIZE AIL_MB(128)
#   define ITER_COUNT 4
#else
#   define MIN_BUFFER_SIZE 128
#   define MAX_BUFFER_SIZE AIL_MB(512)
#   define ITER_COUNT 12
#endif

#ifdef ALL
#   define BENCH
#   define TEST
#endif


#if !defined(__WIN32__) && !defined(_WIN32)
    internal inline void *__movsq(void *d, const void *s, size_t n) {
        asm volatile ("rep movsq"
                        : "=D" (d),
                        "=S" (s),
                        "=c" (n)
                        : "0" (d),
                        "1" (s),
                        "2" (n)
                        : "memory");
        return d;
    }
    internal inline void *__movsb(void *d, const void *s, size_t n) {
        asm volatile ("rep movsb"
                        : "=D" (d),
                        "=S" (s),
                        "=c" (n)
                        : "0" (d),
                        "1" (s),
                        "2" (n)
                        : "memory");
        return d;
    }
#endif

typedef struct {
    u64 size;
    u64 overlap_size;
    u8 *dst;
    u8 *src;
} Buffer;
AIL_SLICE_INIT(Buffer);

internal void print_buffer(Buffer buf)
{
    printf("Buffer (size=%zu): src (%p) = [", buf.size, (void*)buf.src);
    for (u64 i = 0; i < buf.size; i++) {
        if (i) printf(" ");
        printf("%d", buf.src[i]);
    }
    printf("], dst (%p) = [", (void*)buf.dst);
    for (u64 i = 0; i < buf.size; i++) {
        if (i) printf(" ");
        printf("%d", buf.dst[i]);
    }
    printf("]\n");
}

internal Buffer get_buffer(u64 size, u64 overlap_size, b32 overlap_left)
{
    Buffer buf = {0};
    buf.size = size;
    buf.overlap_size = overlap_size;
    if (!overlap_size) {
        buf.dst  = AIL_CALL_ALLOC(ail_alloc_pager, size);
        buf.src  = AIL_CALL_ALLOC(ail_alloc_pager, size);
    } else {
        u8 *left  = AIL_CALL_ALLOC(ail_alloc_pager, size*2 - overlap_size);
        u8 *right = left + size - overlap_size;
        if (overlap_left) {
            buf.dst = left;
            buf.src = right;
        } else {
            buf.dst = right;
            buf.src = left;
        }
    }
    AIL_ASSERT(buf.dst != 0);
    AIL_ASSERT(buf.src != 0);
    return buf;
}

internal void free_buffer(Buffer buf)
{
    if (buf.overlap_size) {
        AIL_CALL_FREE(ail_alloc_pager, AIL_MIN(buf.dst, buf.src));
    } else {
        AIL_CALL_FREE(ail_alloc_pager, buf.dst);
        AIL_CALL_FREE(ail_alloc_pager, buf.src);
    }
}

internal void fill_buffer(Buffer buf)
{
    u8 x = 0;
	for (u64 i = 0; i < buf.size; i++) {
		buf.src[i] = x++;
	}
}

internal b32 test_buffer(Buffer buf)
{
    u8 x = 0;
    for (u64 i = 0; i < buf.size; i++) {
		if (buf.dst[i] != x++) return false;
	}
    return true;
}


internal void copy_bytes(void* restrict dst, void* restrict src, u64 size)
{
    AIL_BENCH_PROFILE_START(copy_bytes);
    u8 *d = dst;
    u8 *s = src;
    for (u64 i = 0; i < size; i++) d[i] = s[i];
    AIL_BENCH_PROFILE_END(copy_bytes);
}

internal void move_bytes(void *dst, void *src, u64 size)
{
    AIL_BENCH_PROFILE_START(move_bytes);
    u8 *d = dst;
    u8 *s = src;
    if (s < d) {
        for (i64 i = size - 1; i >= 0; i--) d[i] = s[i];
    } else {
        for (u64 i = 0; i < size; i++) d[i] = s[i];
    }
    AIL_BENCH_PROFILE_END(move_bytes);
}

internal void copy_bytes_wide(void* restrict dst, void* restrict src, u64 size)
{
    AIL_BENCH_PROFILE_START(copy_bytes_wide);
    u8 *d = dst;
    u8 *s = src;
    u64 rem = size % 4;
    for (u64 i = 0; i < size - rem; i += 4) {
        d[i + 0] = s[i + 0];
        d[i + 1] = s[i + 1];
        d[i + 2] = s[i + 2];
        d[i + 3] = s[i + 3];
    }
    for (u64 i = 0; i < rem; i++) {
        d[size - i - 1] = s[size - i - 1];
    }
    AIL_BENCH_PROFILE_END(copy_bytes_wide);
}

internal void copy_bytes_wide_backwards(void* restrict dst, void* restrict src, u64 size)
{
    AIL_BENCH_PROFILE_START(copy_bytes_wide_backwards);
    u8 *d = dst;
    u8 *s = src;
    u64 rem = size % 4;
    for (i64 i = size - 1; i >= 0; i -= 4) {
        d[i - 0] = s[i - 0];
        d[i - 1] = s[i - 1];
        d[i - 2] = s[i - 2];
        d[i - 3] = s[i - 3];
    }
    for (i64 i = rem - 1; i >= 0; i--) {
        d[i] = s[i];
    }
    AIL_BENCH_PROFILE_END(copy_bytes_wide_backwards);
}

internal void move_bytes_wide(void *dst, void *src, u64 size)
{
    AIL_BENCH_PROFILE_START(move_bytes_wide);
    u8 *d = dst;
    u8 *s = src;
    if (s < d && d < s + size) {
        copy_bytes_wide_backwards(dst, src, size);
    } else {
        copy_bytes_wide(dst, src, size);
    }
    AIL_BENCH_PROFILE_END(move_bytes_wide);
}

internal void copy_quads(void* restrict dst, void* restrict src, u64 size)
{
    AIL_BENCH_PROFILE_START(copy_quads);
    u64 n   = size / sizeof(u64);
    u64 rem = size &  (sizeof(u64) - 1);
    u64 *d = dst;
    u64 *s = src;
    for (u64 i = 0; i < n; i++) d[i] = s[i];
    for (u64 i = 0; i < rem; i++) ((u8*)dst)[n*sizeof(u64) + i] = ((u8*)src )[n*sizeof(u64) + i];
    AIL_BENCH_PROFILE_END(copy_quads);
}

internal void copy_simd(void* restrict dst, void* restrict src, u64 size)
{
    AIL_BENCH_PROFILE_START(copy_simd);
    AIL_STATIC_ASSERT(AIL_IS_2POWER_POS(sizeof(__m128))); // Otherwise, fancy modulo trick below doesn't work
    u64 n   = size / sizeof(__m128);
    u64 rem = size & (sizeof(__m128) - 1);
    __m128i *s = (__m128i*)src;
    __m128i *d = (__m128i*)dst;
    for (u64 i = 0; i < n; i++) {
        _mm_storeu_si128(d + i, _mm_loadu_si128(s + i));
    }
    for (u64 i = 0; i < rem; i++) {
        ((u8*)dst)[size - i - 1] = ((u8*)src)[size - i - 1];
    }
    AIL_BENCH_PROFILE_END(copy_simd);
}

internal void copy_simd_backwards(void* restrict dst, void* restrict src, u64 size)
{
    AIL_BENCH_PROFILE_START(copy_simd_backwards);
    AIL_STATIC_ASSERT(AIL_IS_2POWER_POS(sizeof(__m128))); // Otherwise, fancy modulo trick below doesn't work
    u64 n   = size / sizeof(__m128);
    u64 rem = size & (sizeof(__m128) - 1);
    __m128i *s = (__m128i*)src;
    __m128i *d = (__m128i*)dst;
    for (i64 i = n - 1; i >= 0; i--) {
        _mm_storeu_si128(d + i, _mm_loadu_si128(s + i));
    }
    for (i64 i = rem - 1; i >= 0; i--) {
        ((u8*)dst)[i] = ((u8*)src)[i];
    }
    AIL_BENCH_PROFILE_END(copy_simd_backwards);
}

internal void move_simd(void *dst, void *src, u64 size)
{
    AIL_BENCH_PROFILE_START(move_simd);
    AIL_STATIC_ASSERT(AIL_IS_2POWER_POS(sizeof(__m128))); // Otherwise, fancy modulo trick below doesn't work
    u8 *d = (u8*)dst;
    u8 *s = (u8*)src;
    if (s < d && d < s + size) {
        copy_simd_backwards(d, s, size);
    } else if (d < s && s < d + size) {
        u64 overlapped = d + size - s;
        copy_simd(d, s, size - overlapped);
        copy_bytes(d + size - overlapped, s + size - overlapped, overlapped);
    } else {
        copy_simd(dst, src, size);
    }
    AIL_BENCH_PROFILE_END(move_simd);
}

internal void copy_rep_movs(void* restrict dst, void* restrict src, u64 size)
{
    AIL_BENCH_PROFILE_START(copy_rep_movs);
    u64 n   = size / sizeof(u64);
    u64 rem = size & (sizeof(u64) - 1);
    i64 *s = (i64*)src;
    i64 *d = (i64*)dst;
    if (n) __movsq(d, s, n);
    if (rem) __movsb((u8*)(d + n), (u8*)(s + n), rem);
    AIL_BENCH_PROFILE_END(copy_rep_movs);
}

internal void move_rep_movs(void *dst, void *src, u64 size)
{
    printf("move_rep_movs:\n");
    print_buffer((Buffer) { .dst = dst, .src = src, .size = size });
    AIL_BENCH_PROFILE_START(move_simd);
    u8 *d = (u8*)dst;
    u8 *s = (u8*)src;
    if (s < d && d < s + size) {
        u64 n = s + size - d;
        __movsq((i64*)(d + n), (i64*)(s + n), (size - n) / sizeof(u64));
        for (i64 i = ((size - n) % sizeof(u64)) - 1; i >= 0; i--) {
            d[n + i] = s[n + i];
        }
        copy_rep_movs(d, s, n);
    } else if (d < s && s < d + size) {
        u64 overlapped = d + size - s;
        copy_rep_movs(d, s, size - overlapped);
        copy_bytes(d + size - overlapped, s + size - overlapped, overlapped);
    } else {
        copy_rep_movs(dst, src, size);
    }
    AIL_BENCH_PROFILE_END(move_simd);
    print_buffer((Buffer) { .dst = dst, .src = src, .size = size });
}

internal void copy_builtin(void* restrict dst, void* restrict src, u64 size)
{
    AIL_BENCH_PROFILE_START(copy_builtin);
    memcpy(dst, src, size);
    AIL_BENCH_PROFILE_END(copy_builtin);
}

internal void move_builtin(void *dst, void *src, u64 size)
{
    AIL_BENCH_PROFILE_START(move_builtin);
    memmove(dst, src, size);
    AIL_BENCH_PROFILE_END(move_builtin);
}

#define COPY_FUNCS \
    X(copy_bytes) \
    X(copy_bytes_wide) \
    X(copy_bytes_wide_backwards) \
    X(copy_quads) \
    X(copy_simd) \
    X(copy_simd_backwards) \
    X(copy_rep_movs) \
    X(copy_builtin)

#define MOVE_FUNCS \
    X(move_bytes) \
    X(move_bytes_wide) \
    X(move_simd) \
    X(move_builtin)
    // X(move_rep_movs) \


typedef struct {
    u64 size;
    u64 overlap_size;
    b32 overlap_left;
} Input;

global Input test_inputs[] = {
    { .size = 1,              .overlap_size = 0 },
    { .size = 1,              .overlap_size = 1 },
    { .size = 15,             .overlap_size = 0 },
    { .size = 15,             .overlap_size = 3, .overlap_left = false },
    { .size = 15,             .overlap_size = 7, .overlap_left = true },
    { .size = 16,             .overlap_size = 0 },
    { .size = 16,             .overlap_size = 4, .overlap_left = false },
    { .size = 16,             .overlap_size = 9, .overlap_left = true },
    { .size = 17,             .overlap_size = 0 },
    { .size = 17,             .overlap_size = 17, .overlap_left = false },
    { .size = 17,             .overlap_size = 16, .overlap_left = true },
    { .size = 25,             .overlap_size = 0 },
    { .size = 25,             .overlap_size = 1, .overlap_left = false },
    { .size = 25,             .overlap_size = 2, .overlap_left = true },
    { .size = 31,             .overlap_size = 0 },
    { .size = 32,             .overlap_size = 0 },
    { .size = 33,             .overlap_size = 0 },
    { .size = 511,            .overlap_size = 0 },
    { .size = 511,            .overlap_size = 256, .overlap_left = false },
    { .size = 511,            .overlap_size = 1,   .overlap_left = true },
    { .size = 512,            .overlap_size = 0 },
    { .size = 512,            .overlap_size = 16,  .overlap_left = false },
    { .size = 512,            .overlap_size = 511, .overlap_left = true },
    { .size = 513,            .overlap_size = 0 },
    { .size = AIL_KB(1) + 15, .overlap_size = 0 },
    { .size = AIL_KB(1) + 15, .overlap_size = AIL_KB(1), .overlap_left = false },
    { .size = AIL_KB(1) + 15, .overlap_size = 420,       .overlap_left = true },
    { .size = AIL_KB(1) + 17, .overlap_size = 0 },
    { .size = AIL_KB(1) + 17, .overlap_size = 3,             .overlap_left = false },
    { .size = AIL_KB(1) + 17, .overlap_size = AIL_KB(1) + 2, .overlap_left = true },
    { .size = AIL_KB(1) + 17, .overlap_size = AIL_KB(1) + 2, .overlap_left = true },
};

typedef Buffer BufferList[AIL_ARRLEN(test_inputs)];
typedef void (FuncType)(void *dst, void *src, u64 size);

static void test(BufferList buffers, FuncType func, char *func_name, b32 is_copy_func)
{
	for (u64 i = 0; i < AIL_ARRLEN(test_inputs); i++) {
		Buffer buf = buffers[i];
        if (!is_copy_func || !buf.overlap_size) {
            fill_buffer(buf);
		    func(buf.dst, buf.src, buf.size);
            if (!test_buffer(buf)) {
		    	printf("\033[31m%s failed test for buffer-size %zu (with %zu %s-overlapped bytes) :(\033[0m\n", func_name, test_inputs[i].size, test_inputs[i].overlap_size, test_inputs[i].overlap_left ? "left" : "right");
		    	return;
		    }
        }
	}
	printf("\033[32m%s succeeded all tests :)\033[0m\n", func_name);
}

void get_printable_mem_size(char *str, u64 mem_size)
{
	if      (mem_size >= AIL_GB(1)) snprintf(str, 8, "%zuGB", mem_size/AIL_GB(1));
	else if (mem_size >= AIL_MB(1)) snprintf(str, 8, "%zuMB", mem_size/AIL_MB(1));
	else if (mem_size >= AIL_KB(1)) snprintf(str, 8, "%zuKB", mem_size/AIL_KB(1));
	else                            snprintf(str, 8, "%zuB", mem_size);
}

int main(void)
{
    u64 t0 = ail_bench_cpu_timer();
#ifdef TEST
    BufferList buffers;
    for (u64 i = 0; i < AIL_ARRLEN(test_inputs); i++) {
        Input in = test_inputs[i];
        buffers[i] = get_buffer(in.size, in.overlap_size, in.overlap_left);
    }
    #define X(func) test(buffers, func, AIL_STRINGIFY(func), true);
		COPY_FUNCS
	#undef X
    #define X(func) test(buffers, func, AIL_STRINGIFY(func), false);
		MOVE_FUNCS
	#undef X
	for (u64 i = 0; i < AIL_ARRLEN(test_inputs); i++) {
		free_buffer(buffers[i]);
	}
#endif

#ifdef BENCH
    ail_bench_clear_anchors();
    for (u64 i = 0, size = MIN_BUFFER_SIZE, overlap = 4; size <= MAX_BUFFER_SIZE; size <<= 2, overlap <<= 1, i++) {
        AIL_ASSERT(overlap < size);
        Buffer buffers[] = {
            get_buffer(size, 0, 0),
            get_buffer(size, overlap, i & 1),
            get_buffer(size, size - overlap, !(i & 1)),
        };
        char mem_size[12];
        get_printable_mem_size(mem_size, size);
        printf("Benchmark Results for Copying %s of memory\n", mem_size);
        for (u64 j = 0; j < AIL_ARRLEN(buffers); j++) {
            Buffer buf = buffers[j];
            if (!buf.overlap_size) {
                fill_buffer(buf);
                ail_bench_begin_profile();
                for (u64 k = 0; k < ITER_COUNT; k++) {
                    #define X(func) func(buf.dst, buf.src, buf.size);
                        COPY_FUNCS
                    #undef X
                }
                ail_bench_end_and_print_profile(1, true);
            }
        }
        printf("-----------\n");
        printf("Benchmark Results for Moving %s of memory\n", mem_size);
        for (u64 j = 0; j < AIL_ARRLEN(buffers); j++) {
            Buffer buf = buffers[j];
            fill_buffer(buf);
            printf("With %zu overlapped bytes:\n", buf.overlap_size);
            ail_bench_begin_profile();
            for (u64 k = 0; k < ITER_COUNT; k++) {
                #define X(func) func(buf.dst, buf.src, buf.size);
                    MOVE_FUNCS
                #undef X
            }
            ail_bench_end_and_print_profile(1, true);
        }
        printf("-----------\n");
        for (u64 j = 0; j < AIL_ARRLEN(buffers); j++) free_buffer(buffers[j]);
    }
#endif

    u64 t1 = ail_bench_cpu_timer();
    f64 elapsed_ms   = ail_bench_cpu_elapsed_to_ms(t1 - t0);
    f64 second_in_ms = 1000.0f;
    f64 minute_in_ms = 60000.0f;
	printf("Total time for running entire program: ~");
    if (elapsed_ms > minute_in_ms) printf("%fmin\n", elapsed_ms/minute_in_ms);
    else printf("%fsec\n", elapsed_ms/second_in_ms);
}
