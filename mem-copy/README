# Memory Copy

Copy a region of memory into another region.

Like in C's standard library, there are two different interfaces that were implemented:
- copy: Is only guarantueed to work with non-overlapping memory regions
- move: Works with any memory regions (Note that `src` gets overwritten if the memory regions overlap of course)

All code is contained within the `mem-copy.c` file.

There are a few ways to easily customize the program. All of these are done via macros, that are defined at the top of the file.

- `#define TEST` enables code to test all routines for correctness
- `#define BENCH` enables code to benchmark all routines
- `#define ALL` enables both testing and benchmarking
- `#define MIN_BUFFER_SIZE n` sets the minimum amount of memory to move/copy when benchmarking to `n`
- `#define MAX_BUFFER_SIZE n` sets the maximum amount of memory to move/copy when benchmarking to `n`
- `#define ITER_COUNT n` sets the amount of iterations done when benchmarking to `n`

When benchmarking, each routine is printed with the amount of times it was called.
Next to its name is the amount of time spent in the function in total (both in approx. clock cycles and milliseconds).
The percentage shows how much relative time of the program was spent in this function.
The `Min:` section shows how long the shortest run of the function took.

## Quickstart

Depending on your platform/compiler, run the following command to build and execute:

- `gcc -o mem-copy mem-copy.c -march=native && ./mem-copy`
- `clang -o mem-copy mem-copy.c -march=native && ./mem-copy`
- `cl mem-copy.c && mem-copy.exe`

## Procedures

(This is very much still work-in-progress)

The followign copy-procedures are currently implemented:
- `copy_bytes`: Naive byte-per-byte copy
- `copy_bytes_wide`: 4-times unrolled byte-per-byte copy
- `copy_bytes_wide_backwards`: 4-times unrolled byte-per-byte copy but from the back to the front of the buffer
- `copy_quads`: Copy individual quadwords (64 bits) at a time
- `copy_simd`: Uses SSE2 to copy 16 bytes at a time
- `copy_simd_backwards`: Same as copy_simd but going from the back to the front of the buffer
- `copy_rep_movs`: Uses the intrinsic `__movsq` (aka the `rep mov` assembly instruction) to copy n bytes without any loop
- `copy_builtin`: Uses the standard C library's memcpy - serves as a highly optimized reference implementation

The followign move-procedures are currently implemented:
- `move_bytes`: Naive byte-per-byte move
- `move_bytes_wide`: 4-times unrolled byte-per-byte move
- `move_simd`: Uses SSE2 to move 16 bytes at a time - certain overlapped memory regions are highly pessimized right now, falling back to byte-per-byte copy
- `move_builtin`: Uses the standard C library's memmove - serves as a highly optimized reference implementation

## Requirements

A CPU with SSE2 and SSSE3 extensions is required for the SIMD routines to work.

The code has only been tested on Windows and Linux.
