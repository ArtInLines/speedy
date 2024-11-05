# Memory Reverser

Reverse Memory Buffers. Reversal is on a byte-basis, such that the first byte becomes the last byte, second byte becomes second-to-last byte, etc.

There are two different interfaces, that were implemented:
- in_place: The buffer is reversed in place
- otherwise: The memory region is copied in reversed order into a new buffer

The idea for this came out of a discussion with someone claiming that this kind of task could not be substantially sped up with SIMD. This repo clearly disproves that claim.

All code is contained within the `mem-reverse.c` file.

There are a few ways to easily customize the program. All of these are done via macros, that are defined at the top of the file.

- `#define TEST` enables code to test all routines for correctness
- `#define BENCH` enables code to benchmark all routines
- `#define ALL` enables both testing and benchmarking
- `#define BUFFER_SIZE n` sets the amount of memory to reverse to `n`
- `#define ITER_COUNT n` sets the amount of iterations done when benchmarking to `n`

When benchmarking, each routine is printed with the amount of times it was called.
Next to its name is the amount of time spent in the function in total (both in approx. clock cycles and milliseconds).
The percentage shows how much relative time of the program was spent in this function.
The `Min:` section shows how long the shortest run of the function took.

## Quickstart

```
clang -o mem-reverse.exe mem-reverse.c -march=native -O1 && mem-reverse.exe
```

## Procedures

There are currently 4 implemenations:
1. `scalar`: A simple scalar loop, writing the result into a second buffer and going byte-by-byte through the buffer
2. `scalar_in_place`: A simple scalar loop, reversing the buffer in place and going byte-by-byte through the buffer
3. `scalar_wide`: An unrolled scalar loop, writing the result into a second buffer and going byte-by-byte through the buffer
4. `scalar_wide_in_place`: An unrolled scalar loop, reversing the buffer in place and going byte-by-byte through the buffer
5. `simd_shuffle`: A simple SIMD loop, using the SSSE3 shuffling instruction for reversing bytes and writing the result into a second buffer
6. `simd_shuffle_in_place`: A simple SIMD loop, using the same shuffling instruction, but reversing the buffer in place

## Requirements

Benchmarking is currently only implemented for x86-64 architectures.

A CPU with SSE2 and SSSE3 extensions is required for the SIMD routines to work.

For allocating memory, VirtualAlloc/mmap is currently used. An OS that doesn't support either of these thus requires minor changes.

The code has been tested on Windows and Linux.
