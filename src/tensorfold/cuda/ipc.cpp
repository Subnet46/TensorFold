// One rank's side of the CUDA IPC one-shot all-gather (ipc.cu): the exported region, the peers' mapped regions,
// the local control words (epoch, arrival count, poison) and the mapped host error record.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "ipc.h"

namespace {

void cuda_ok(cudaError_t code, const char* what) {
    if (code != cudaSuccess) {
        cudaGetLastError();                     // an API error is not sticky: clear it before raising
        TORCH_CHECK(false, "CUDA IPC all-gather: ", what, ": ", cudaGetErrorString(code));
    }
}

}  // namespace

// The launch geometry from the byte count alone, so every rank's launch of a gather matches: one block for each
// block_bytes of the shard (at least one, at most MAX_BLOCKS), the shard's 16-byte packs split into equal
// contiguous runs, and no empty block.
std::vector<int64_t> plan(int64_t nbytes, int64_t block_bytes) {
    TORCH_CHECK(nbytes > 0 && block_bytes > 0, "CUDA IPC all-gather: plan needs positive sizes");
    const int64_t packs = (nbytes + PACK_BYTES - 1) / PACK_BYTES;
    int64_t grid = (nbytes + block_bytes - 1) / block_bytes;
    grid = std::max<int64_t>(1, std::min<int64_t>({grid, (int64_t)MAX_BLOCKS, packs}));
    const int64_t chunk = (packs + grid - 1) / grid;
    grid = (packs + chunk - 1) / chunk;
    return {grid, chunk};
}

class Gather {
  public:
    Gather(int64_t world, int64_t rank, int64_t slot_bytes, int64_t timeout_ns, int64_t block_bytes, int64_t threads)
        : world_(world), rank_(rank), slot_bytes_(slot_bytes), timeout_ns_(timeout_ns), block_bytes_(block_bytes),
          threads_(threads) {
        TORCH_CHECK(2 <= world && world <= MAX_WORLD, "CUDA IPC all-gather: 2 to ", MAX_WORLD, " ranks, not ", world);
        TORCH_CHECK(0 <= rank && rank < world, "CUDA IPC all-gather: rank ", rank, " of ", world);
        TORCH_CHECK(slot_bytes > 0 && slot_bytes % 256 == 0, "CUDA IPC all-gather: slot bytes a multiple of 256");
        TORCH_CHECK(block_bytes >= PACK_BYTES, "CUDA IPC all-gather: block bytes at least ", PACK_BYTES);
        check_threads(threads);
        TORCH_CHECK(timeout_ns > 0, "CUDA IPC all-gather: a positive wait limit");
        cuda_ok(cudaGetDevice(&device_), "cudaGetDevice");
        region_bytes_ = DATA_OFFSET + 2LL * (world - 1) * slot_bytes;
        std::memset(&regions_, 0, sizeof(regions_));
        cuda_ok(cudaMalloc(&region_, (size_t)region_bytes_), "cudaMalloc (the IPC region)");
        cuda_ok(cudaMalloc(&ctl_, 256), "cudaMalloc (control words)");
        cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&err_host_), 64, cudaHostAllocMapped | cudaHostAllocPortable),
                "cudaHostAlloc (error record)");
        std::memset(err_host_, 0, 64);
        cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void**>(&err_dev_), err_host_, 0),
                "cudaHostGetDevicePointer");
        cuda_ok(cudaMemset(region_, 0, (size_t)region_bytes_), "cudaMemset (the IPC region)");
        cuda_ok(cudaMemset(ctl_, 0, 256), "cudaMemset (control words)");
        cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize");   // zeroed before any peer can write a flag
        regions_.base[rank] = static_cast<char*>(region_);
    }

    ~Gather() { close(); }

    // this rank's region as a CUDA IPC handle (CPU uint8 tensor)
    at::Tensor handle() const {
        TORCH_CHECK(region_ != nullptr, "CUDA IPC all-gather: closed");
        const c10::cuda::CUDAGuard guard(device_);
        cudaIpcMemHandle_t h;
        cuda_ok(cudaIpcGetMemHandle(&h, region_), "cudaIpcGetMemHandle");
        at::Tensor out = at::empty({(int64_t)sizeof(h)}, at::TensorOptions().dtype(at::kByte));
        std::memcpy(out.data_ptr(), &h, sizeof(h));
        return out;
    }

    // every rank's handle, in rank order (this rank's own is skipped): map the peers' regions
    void connect(const std::vector<at::Tensor>& handles) {
        TORCH_CHECK((int64_t)handles.size() == world_, "CUDA IPC all-gather: one handle a rank");
        TORCH_CHECK(opened_.empty(), "CUDA IPC all-gather: already connected");
        const c10::cuda::CUDAGuard guard(device_);
        for (int64_t r = 0; r < world_; ++r) {
            if (r == rank_) continue;
            const at::Tensor h = handles[r].contiguous().to(at::kCPU);
            TORCH_CHECK(h.numel() * h.element_size() == (int64_t)sizeof(cudaIpcMemHandle_t),
                        "CUDA IPC all-gather: a handle is ", sizeof(cudaIpcMemHandle_t), " bytes");
            cudaIpcMemHandle_t handle;
            std::memcpy(&handle, h.data_ptr(), sizeof(handle));
            void* p = nullptr;
            const cudaError_t code = cudaIpcOpenMemHandle(&p, handle, cudaIpcMemLazyEnablePeerAccess);
            if (code != cudaSuccess) {
                cudaGetLastError();
                unmap();
                TORCH_CHECK(false, "CUDA IPC all-gather: cudaIpcOpenMemHandle (rank ", r, "'s region): ",
                            cudaGetErrorString(code));
            }
            opened_.push_back(p);
            regions_.base[r] = static_cast<char*>(p);
        }
    }

    // recv [world * n] <- every rank's send [n] in rank order; contiguous CUDA tensors on this device, n bytes at
    // most slot_bytes. block_bytes / threads: the defaults when negative (every rank must pass the same block_bytes)
    void run(const at::Tensor& send, at::Tensor recv, int64_t block_bytes, int64_t threads) {
        TORCH_CHECK(!opened_.empty(), "CUDA IPC all-gather: not connected");
        TORCH_CHECK(send.is_cuda() && recv.is_cuda() && send.is_contiguous() && recv.is_contiguous(),
                    "CUDA IPC all-gather: contiguous CUDA tensors");
        TORCH_CHECK(send.get_device() == device_ && recv.get_device() == device_,
                    "CUDA IPC all-gather: tensors on the region's device");
        const int64_t nbytes = send.numel() * (int64_t)send.element_size();
        TORCH_CHECK(nbytes > 0 && nbytes <= slot_bytes_, "CUDA IPC all-gather: 1 to ", slot_bytes_, " bytes a rank, not ",
                    nbytes);
        TORCH_CHECK(recv.numel() * (int64_t)recv.element_size() == world_ * nbytes,
                    "CUDA IPC all-gather: recv must hold world x send bytes");
        const int64_t t = threads < 0 ? threads_ : threads;
        check_threads(t);
        const std::vector<int64_t> geometry = plan(nbytes, block_bytes < 0 ? block_bytes_ : block_bytes);
        const c10::cuda::CUDAGuard guard(device_);
        auto* ctl = static_cast<char*>(ctl_);
        ipc_gather_cuda(send, recv, nbytes, geometry[1], (int)geometry[0], (int)t, regions_, slot_bytes_,
                        reinterpret_cast<unsigned long long*>(ctl), reinterpret_cast<unsigned*>(ctl + 8),
                        reinterpret_cast<unsigned*>(ctl + 16), err_dev_, timeout_ns_, (int)world_, (int)rank_);
    }

    // the mapped host error record's address: {failed sequence (0: none), peer, block, waited ns}
    int64_t error_address() const { return reinterpret_cast<int64_t>(err_host_); }

    std::vector<int64_t> error() const {
        if (err_host_ == nullptr) return {0, 0, 0, 0};
        const volatile long long* e = err_host_;
        return {e[0], e[1], e[2], e[3]};
    }

    int64_t region_bytes() const { return region_bytes_; }

    // the wait limit of later launches (a captured graph keeps the value it was captured with)
    void set_timeout(int64_t timeout_ns) {
        TORCH_CHECK(timeout_ns > 0, "CUDA IPC all-gather: a positive wait limit");
        timeout_ns_ = timeout_ns;
    }

    void close() {
        if (region_ == nullptr && opened_.empty() && ctl_ == nullptr && err_host_ == nullptr) return;
        cudaSetDevice(device_);
        unmap();
        if (region_ != nullptr) cudaFree(region_);
        if (ctl_ != nullptr) cudaFree(ctl_);
        if (err_host_ != nullptr) cudaFreeHost(err_host_);
        region_ = ctl_ = nullptr;
        err_host_ = err_dev_ = nullptr;
        cudaGetLastError();                     // teardown at exit: nothing to report
    }

  private:
    static void check_threads(int64_t threads) {
        TORCH_CHECK(threads >= 32 && threads <= MAX_THREADS && threads % 32 == 0,
                    "CUDA IPC all-gather: threads a block a multiple of 32 up to ", MAX_THREADS);
    }

    void unmap() {
        for (void* p : opened_) cudaIpcCloseMemHandle(p);
        opened_.clear();
        for (int64_t r = 0; r < world_; ++r)
            if (r != rank_) regions_.base[r] = nullptr;
    }

    int64_t world_, rank_, slot_bytes_, timeout_ns_, block_bytes_, threads_;
    int device_ = 0;
    int64_t region_bytes_ = 0;
    void* region_ = nullptr;
    void* ctl_ = nullptr;
    long long* err_host_ = nullptr;
    long long* err_dev_ = nullptr;
    std::vector<void*> opened_;
    Regions regions_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("plan", &plan, "(grid, packs a block) of a gather of nbytes a rank");
    m.def("layout", []() {
        return std::vector<int64_t>{IPC_ABI, MAX_WORLD, MAX_BLOCKS, MAX_THREADS, PACK_BYTES, FLAG_BYTES, DATA_OFFSET,
                                    (int64_t)sizeof(cudaIpcMemHandle_t)};
    });
    py::class_<Gather>(m, "Gather")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>(), py::arg("world"), py::arg("rank"),
             py::arg("slot_bytes"), py::arg("timeout_ns"), py::arg("block_bytes"), py::arg("threads"))
        .def("handle", &Gather::handle)
        .def("connect", &Gather::connect)
        .def("run", &Gather::run, py::arg("send"), py::arg("recv"), py::arg("block_bytes") = -1,
             py::arg("threads") = -1)
        .def("error_address", &Gather::error_address)
        .def("error", &Gather::error)
        .def("region_bytes", &Gather::region_bytes)
        .def("set_timeout", &Gather::set_timeout)
        .def("close", &Gather::close);
}
