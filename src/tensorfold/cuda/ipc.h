// Shared by ipc.cpp (the region, IPC handles, launches) and ipc.cu (the one-shot all-gather kernel).
#pragma once

#include <ATen/ATen.h>

constexpr int IPC_ABI = 1;                      // ipc.py checks it; bump with any change to the layout below
constexpr int MAX_WORLD = 8;                    // ranks at most
constexpr int MAX_BLOCKS = 128;                 // blocks a launch at most (one flag a block, source and slot)
constexpr int MAX_THREADS = 1024;
constexpr int PACK_BYTES = 16;                  // the kernel moves 16-byte packs
constexpr long long FLAG_BYTES = 2LL * MAX_WORLD * MAX_BLOCKS * 8;     // u64 flag[2][MAX_WORLD][MAX_BLOCKS]
constexpr long long DATA_OFFSET = FLAG_BYTES;   // then data[2][world - 1][slot_bytes]

// every rank's region as mapped in this process (this rank's own entry: the local region)
struct Regions {
    char* base[MAX_WORLD];
};

void ipc_gather_cuda(const at::Tensor& in, at::Tensor& out, long long nbytes, long long chunk, int grid, int threads,
                     const Regions& regions, long long slot_bytes, unsigned long long* epoch, unsigned* arrive,
                     unsigned* poison, long long* err, long long timeout_ns, int world, int rank);
