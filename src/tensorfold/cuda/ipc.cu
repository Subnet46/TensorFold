// One-shot all-gather between the GPUs of one host over CUDA IPC (PCIe or NVLink peer to peer), for decode-sized
// gathers. Same protocol as Mia's one-shot RoCE kernel (roce.cu: device epoch, two slots, acquire-polled flags, a
// timeout that poisons), without a host proxy: each rank writes its shard straight into every peer's slot. The IPC
// buffer handling follows b12x's PCIe one-shot collectives (https://github.com/local-inference-lab/b12x,
// b12x/comm/pcie, Apache License 2.0); no b12x source is copied.
//
// Every rank exports one region (cudaMalloc, opened by the peers with cudaIpcOpenMemHandle):
//   flags  u64 flag[slot 2][source MAX_WORLD][block MAX_BLOCKS]   written by the peers, polled here
//   data   data[slot 2][source, peers only: world - 1][slot_bytes]  written by the peers, read here
// One launch (grid and partition chosen from the byte count alone, so every rank's launch matches):
//   0. every block reads the device epoch: seq = epoch + 1, slot = seq & 1 (a poisoned runtime returns at once);
//   1. block b pushes its partition of the local shard into slot `slot` of every peer (plain stores; peer to peer
//      writes, no atomics: P2P atomics need not be supported) and copies it to its own place in the output;
//   2. every thread fences at system scope, the block meets, then one thread a peer writes the block's flag there
//      with a system-scope release store: flag[slot][rank][b] = seq;
//   3. one thread a peer polls its own flag[slot][peer][b] (relaxed system loads, then an acquire fence); a wait
//      past the limit records the failure in mapped host memory and poisons the runtime;
//   4. the block copies partition b of every peer's shard out of its own slots, rank by rank (the output layout
//      is NCCL's: rank r's shard at r * nbytes), so the bytes equal NCCL's all-gather bit for bit;
//   5. the last block to finish (a local arrival counter) advances the epoch, so CUDA graphs replay the exchange.
// Why two slots and per-block flags are enough: a rank starts call n + 2 (slot n & 1 again) only after it has seen
// every peer's flags of call n + 1, which a peer writes only after its launch n finished reading slot n & 1 (stream
// order). Flags are 64-bit sequence numbers, so a stale flag never equals the awaited one.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "ipc.h"

namespace {

__device__ __forceinline__ unsigned long long ld_relaxed_gpu_u64(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned ld_relaxed_gpu_u32(const unsigned* p) {
    unsigned v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned long long ld_relaxed_sys_u64(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint4 ld_relaxed_sys_v4(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_sys_u64(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed_sys_s64(long long* p, long long v) {
    asm volatile("st.relaxed.sys.global.s64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed_gpu_u64(unsigned long long* p, unsigned long long v) {
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_release_gpu_u32(unsigned* p, unsigned v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_global_v4(void* p, uint4 v) {
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}
__device__ __forceinline__ void fence_acq_rel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// a 16-byte pack i of a byte range [base, base + nbytes): whole and aligned in one vector access, else byte by byte
// (a tail pack, or a buffer that is not 16-byte aligned; only small gathers meet either)
__device__ __forceinline__ uint4 load_pack(const unsigned char* base, long long i, long long nbytes, bool aligned) {
    const long long off = i * PACK_BYTES;
    if (aligned && off + PACK_BYTES <= nbytes) return *reinterpret_cast<const uint4*>(base + off);
    unsigned long long lo = 0ull, hi = 0ull;                            // no indexed array: nothing on the stack
    const int n = (int)min((long long)PACK_BYTES, nbytes - off);
    for (int k = 0; k < n; ++k) {
        const unsigned long long byte = base[off + k];
        if (k < 8) lo |= byte << (8 * k);
        else hi |= byte << (8 * (k - 8));
    }
    return make_uint4((unsigned)lo, (unsigned)(lo >> 32), (unsigned)hi, (unsigned)(hi >> 32));
}
__device__ __forceinline__ void store_pack(unsigned char* base, long long i, long long nbytes, bool aligned, uint4 v) {
    const long long off = i * PACK_BYTES;
    if (aligned && off + PACK_BYTES <= nbytes) {
        *reinterpret_cast<uint4*>(base + off) = v;
        return;
    }
    const unsigned long long lo = ((unsigned long long)v.y << 32) | v.x, hi = ((unsigned long long)v.w << 32) | v.z;
    const int n = (int)min((long long)PACK_BYTES, nbytes - off);
    for (int k = 0; k < n; ++k) base[off + k] = (unsigned char)(k < 8 ? lo >> (8 * k) : hi >> (8 * (k - 8)));
}

__device__ __forceinline__ long long data_offset(int slot, int source, int receiver, int world, long long slot_bytes) {
    const int index = source < receiver ? source : source - 1;          // the receiver keeps no slot for itself
    return DATA_OFFSET + ((long long)slot * (world - 1) + index) * slot_bytes;
}

__device__ __forceinline__ unsigned long long* flag_at(char* region, int slot, int source, int block) {
    return reinterpret_cast<unsigned long long*>(region) + ((long long)slot * MAX_WORLD + source) * MAX_BLOCKS + block;
}

__global__ void __launch_bounds__(MAX_THREADS) ipc_gather_kernel(
        const unsigned char* __restrict__ in, unsigned char* __restrict__ out, long long nbytes, long long chunk,
        const __grid_constant__ Regions regions, long long slot_bytes, unsigned long long* epoch, unsigned* arrive,
        unsigned* poison, long long* err, long long timeout_ns, int world, int rank) {
    __shared__ unsigned long long s_seq;
    __shared__ unsigned s_stop;
    const int tid = threadIdx.x, b = blockIdx.x, nt = blockDim.x;
    if (tid == 0) {
        s_seq = ld_relaxed_gpu_u64(epoch) + 1ull;
        s_stop = ld_relaxed_gpu_u32(poison);
    }
    __syncthreads();
    // 0. a recorded timeout poisons the runtime: later launches do nothing until the host raises
    if (s_stop != 0u) return;
    const unsigned long long seq = s_seq;
    const int slot = (int)(seq & 1ull);
    const long long packs = (nbytes + PACK_BYTES - 1) / PACK_BYTES;
    const long long lo = (long long)b * chunk, hi = min(packs, lo + chunk);
    const bool in_aligned = (reinterpret_cast<uintptr_t>(in) & (PACK_BYTES - 1)) == 0;
    unsigned char* own = out + (long long)rank * nbytes;
    const bool own_aligned = (reinterpret_cast<uintptr_t>(own) & (PACK_BYTES - 1)) == 0;
    const bool own_copy = own != in;                                     // in place: the shard is already there

    // 1. push this block's partition into every peer's slot (peers staggered by rank) and into the own output
    for (long long i = lo + tid; i < hi; i += nt) {
        const uint4 v = load_pack(in, i, nbytes, in_aligned);
        for (int k = 1; k < world; ++k) {
            const int p = (rank + k) % world;
            uint4* dst = reinterpret_cast<uint4*>(regions.base[p] + data_offset(slot, rank, p, world, slot_bytes));
            st_global_v4(dst + i, v);                                    // a plain peer-to-peer write
        }
        if (own_copy) store_pack(own, i, nbytes, own_aligned, v);
    }
    // 2. the block's stores before its flags: a system fence in every thread, the block meets, then the flags
    fence_acq_rel_sys();
    __syncthreads();
    if (tid < world && tid != rank) st_release_sys_u64(flag_at(regions.base[tid], slot, rank, b), seq);

    // 3. wait for every peer's matching block (one thread a peer)
    if (tid < world && tid != rank) {
        const unsigned long long* flag = flag_at(regions.base[rank], slot, tid, b);
        if (ld_relaxed_sys_u64(flag) != seq) {
            const unsigned long long start = globaltimer();
            unsigned polls = 0;
            while (ld_relaxed_sys_u64(flag) != seq) {
                if ((++polls & 255u) != 0u) continue;
                if (ld_relaxed_gpu_u32(poison) != 0u) break;                  // another wait gave up first
                const unsigned long long waited = globaltimer() - start;
                if ((long long)waited > timeout_ns) {
                    st_relaxed_sys_s64(err + 1, tid);
                    st_relaxed_sys_s64(err + 2, b);
                    st_relaxed_sys_s64(err + 3, (long long)waited);
                    fence_acq_rel_sys();
                    st_relaxed_sys_s64(err + 0, (long long)seq);
                    st_release_gpu_u32(poison, 1u);
                    break;
                }
            }
        }
        fence_acq_rel_sys();                                             // acquire: the shard before its flag
    }
    __syncthreads();

    // 4. copy every peer's partition out of the slots, in rank order
    if (ld_relaxed_gpu_u32(poison) == 0u) {
        const char* mine = regions.base[rank];
        for (int s = 0; s < world; ++s) {
            if (s == rank) continue;
            const uint4* src = reinterpret_cast<const uint4*>(mine + data_offset(slot, s, rank, world, slot_bytes));
            unsigned char* dst = out + (long long)s * nbytes;
            const bool dst_aligned = (reinterpret_cast<uintptr_t>(dst) & (PACK_BYTES - 1)) == 0;
            for (long long i = lo + tid; i < hi; i += nt) store_pack(dst, i, nbytes, dst_aligned, ld_relaxed_sys_v4(src + i));
        }
    }

    // 5. the last block advances the epoch (atomicInc wraps the arrival count to 0 for the next launch)
    __syncthreads();
    if (tid == 0) {
        const unsigned prior = atomicInc(arrive, gridDim.x - 1u);
        if (prior == gridDim.x - 1u && ld_relaxed_gpu_u32(poison) == 0u) st_relaxed_gpu_u64(epoch, seq);
    }
}

}  // namespace

void ipc_gather_cuda(const at::Tensor& in, at::Tensor& out, long long nbytes, long long chunk, int grid, int threads,
                     const Regions& regions, long long slot_bytes, unsigned long long* epoch, unsigned* arrive,
                     unsigned* poison, long long* err, long long timeout_ns, int world, int rank) {
    auto stream = at::cuda::getCurrentCUDAStream();
    ipc_gather_kernel<<<(unsigned)grid, (unsigned)threads, 0, stream>>>(
        reinterpret_cast<const unsigned char*>(in.data_ptr()), reinterpret_cast<unsigned char*>(out.data_ptr()),
        nbytes, chunk, regions, slot_bytes, epoch, arrive, poison, err, timeout_ns, world, rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
