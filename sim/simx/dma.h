#pragma once

#include <simobject.h>
#include <stdint.h>
#include <VX_config.h>
#include <VX_types.h>
#include <VX_types.h>
#include "arch.h"
#include <queue>
#include <vector>
#include <mem.h>
#include <bitset>
#include "types.h"


namespace vortex {

class Cluster;

class GlobalMemAdapter : public SimObject<GlobalMemAdapter> {
public:
    std::vector<SimPort<DmaReq>> DmaReqIn;
    std::vector<SimPort<DmaRsp>> DmaRspOut;
    SimPort<DmaReq> DmaReqOut;
    SimPort<DmaRsp> DmaRspIn;

    GlobalMemAdapter(const SimContext& ctx, const char* name, uint32_t num_inputs);
    ~GlobalMemAdapter();

    void reset();
    void tick();

    LsuMemAdapter::Ptr lsu_adapter_;
    DmaArbiter::Ptr arbiter_;

private:
    struct pending_req_t {
        bool last_req;
        uint32_t src_id;
        uint32_t input_port;
        bool write;
        std::vector<uint64_t> dst_addrs;
        uint32_t count;
        uint32_t orig_count;
        uint64_t tag;
    };
    HashTable<pending_req_t> pending_reqs;
};

class LocalMemAdapter : public SimObject<LocalMemAdapter> {
public:
    std::vector<SimPort<DmaReq>> DmaReqIn;
    std::vector<SimPort<DmaRsp>> DmaRspOut;
    SimPort<DmaReq> DmaReqOut;
    SimPort<DmaRsp> DmaRspIn;

    SimPort<LsuReq> LsuReqOut;
    SimPort<LsuRsp> LsuRspIn;

    LocalMemAdapter(const SimContext& ctx, const char* name, uint32_t num_inputs);
    ~LocalMemAdapter();

    void reset();
    void tick();

    DmaArbiter::Ptr arbiter_;

private:
    struct pending_req_t {
        bool last_req;
        uint32_t src_id;
        uint32_t input_port; // Added input_port to LocalMemAdapter pending_req_t
        bool write;          // Added write to LocalMemAdapter pending_req_t
        std::vector<uint64_t> dst_addrs; // Added dst_addrs to LocalMemAdapter pending_req_t
        uint32_t count;
        uint32_t orig_count;
        uint64_t tag;
    };
    HashTable<pending_req_t> pending_reqs;
};

class AccumMemAdapter : public SimObject<AccumMemAdapter> {
public:
    std::vector<SimPort<DmaReq>> DmaReqIn;   // From Virgo_DMA
    std::vector<SimPort<DmaRsp>> DmaRspOut;  // To Virgo_DMA
    std::vector<SimPort<DmaReq>> DmaReqOut;  // To Global/Local MemAdapter
    std::vector<SimPort<DmaRsp>> DmaRspIn;   // From Global/Local MemAdapter

    SimPort<LsuReq> LsuReqOut;               // To MatMul (Accumulator)
    SimPort<LsuRsp> LsuRspIn;                // From MatMul (Accumulator)

    AccumMemAdapter(const SimContext& ctx, const char* name, uint32_t num_inputs);
    ~AccumMemAdapter();

    void reset();
    void tick();

    DmaArbiter::Ptr arbiter_;

private:
    struct pending_req_t {
        bool last_req;
        uint32_t src_id;
        uint32_t input_port;
        bool write;
        std::vector<uint64_t> dst_addrs;
        uint32_t count;
        uint32_t orig_count;
        uint64_t tag;
    };
    HashTable<pending_req_t> pending_reqs;
};

class Virgo_DMA : public SimObject<Virgo_DMA> {
public:

    struct PerfStats {
        uint64_t reads;
        uint64_t writes;
        uint64_t transfers;
    };

    Virgo_DMA(const SimContext& ctx,
              const Arch &arch,
              Cluster* cluster);
    ~Virgo_DMA();
    
    SimPort<LsuReq> ReadReqIn;
    SimPort<LsuRsp> ReadRspIn;

    SimPort<LsuReq> WriteReqIn;
    SimPort<LsuRsp> WriteRspIn;

    SimPort<MatMulDmaReq> MatMulReqIn;
    SimPort<MatMulDmaRsp> MatMulRspOut;

    void read(void* data, uint64_t addr, uint32_t size);
    void write(const void* data, uint64_t addr, uint32_t size);

    void attach_ram(RAM* ram);

    void reset();
    void tick();

    const PerfStats& perf_stats() const;

    const GlobalMemAdapter::Ptr& global_mem_adapter() const { return global_mem_adapter_; }
    const LocalMemAdapter::Ptr& local_mem_adapter() const { return local_mem_adapter_; }
    const AccumMemAdapter::Ptr& accum_mem_adapter() const { return accum_mem_adapter_; }
private:
    struct dma_load_t {
        uint32_t src_addr;
        uint32_t dst_addr;
        uint32_t data_type_size;
        uint32_t num_rows;
        uint32_t num_cols;
        uint32_t src_stride;
        uint32_t dst_stride;
        bool is_accum;
        uint32_t tag;
    };

    SimPort<dma_load_t> DmaLoadReqIn;
    SimPort<dma_load_t> DmaLoadReqOut;

    void dma_transfer(const dma_load_t& dma_load);
    void data_transfer( uint64_t src_addr, 
                        uint64_t dst_addr, 
                        uint32_t data_type_size, 
                        AddrType src_addr_type, 
                        AddrType dst_addr_type);

    Cluster* cluster_;
    Arch arch_;
    std::vector<uint32_t> write_registers;
    std::vector<uint32_t> read_registers;
    std::queue<dma_load_t> dma_load_queue_;
    std::vector<std::bitset<32>> tag_table;
    std::vector<uint32_t> timing_mmio_queue;
    MemoryUnit mmu_;
    PerfStats perf_stats_;

    struct DmaState {
        uint32_t row;
        uint32_t col;
        bool busy;
        DmaState() : row(0), col(0), busy(false) {}
        void reset() { row = 0; col = 0; busy = false; }
    };
    DmaState dma_state_;
    GlobalMemAdapter::Ptr global_mem_adapter_;
    LocalMemAdapter::Ptr local_mem_adapter_;
    AccumMemAdapter::Ptr accum_mem_adapter_;

};

} // namespace vortex