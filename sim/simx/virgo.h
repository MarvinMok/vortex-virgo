#pragma once
#include <cstdint>
#include <mem.h>
#include <simobject.h>
#include "arch.h"
#include <queue>
#include <vector>
#include <mem.h>
#include <bitset>
#include "types.h"
#include <VX_config.h>
#include <VX_types.h>
namespace vortex {

class Cluster;

typedef struct {
    uint32_t src_addr_A;
    uint32_t src_addr_B;
    uint32_t dst_addr;
    uint32_t data_type;
    uint32_t num_rows_A; // [rows_A x cols_A] x [cols_A x cols_B] = [rows_A x cols_B]
    uint32_t num_cols_A; 
    uint32_t num_cols_B;
    uint32_t accum_addr;
    uint32_t  tag;
    bool store;
    bool accum;
} virgo_req_t;

typedef struct {
    uint64_t scratchpad_src_addr_A;
    uint64_t scratchpad_src_addr_B;
} virgo_rsp_t;

#define SCRATCHPAD_BANKS NUM_LSU_LANES

#define SCRATCHPAD_TILE_A_1 0
#define SCRATCHPAD_TILE_B_1 1024
#define SCRATCHPAD_TILE_A_2 2048
#define SCRATCHPAD_TILE_B_2 3072

class AccumulatorMem : public SimObject<AccumulatorMem> {
public:
    AccumulatorMem(const SimContext& ctx, const char* name, uint64_t capacity);
    ~AccumulatorMem();

    SimPort<LsuReq> ReadReqIn;
    SimPort<LsuRsp> ReadRspOut;
    SimPort<LsuReq> WriteReqIn;
    SimPort<LsuRsp> WriteRspOut;

    void read(void* data, uint64_t addr, uint32_t size);
    void write(const void* data, uint64_t addr, uint32_t size);
    void tick();
    void reset();
    uint64_t size() const;

private:
    RAM ram_;
    MemoryUnit mmu_;
    uint64_t capacity_;
};

class ScratchPadMem : public SimObject<ScratchPadMem> {
public:
    ScratchPadMem(const SimContext& ctx, const char* name, uint64_t capacity);
    ~ScratchPadMem();

    SimPort<LsuReq> ReadReqIn;
    SimPort<LsuRsp> ReadRspOut;
    SimPort<LsuReq> WriteReqIn;
    SimPort<LsuRsp> WriteRspOut;

    void read(void* data, uint64_t addr, uint32_t size);
    void write(const void* data, uint64_t addr, uint32_t size);
    void tick();
    void reset();
    uint64_t size() const;

private:
    std::vector<SimPort<MemReq>> BankWriteReqOut;
    std::vector<SimPort<MemRsp>> BankWriteRspIn;
    std::vector<SimPort<MemReq>> BankReadReqOut;
    std::vector<SimPort<MemRsp>> BankReadRspIn;
    RAM ram_;
    MemoryUnit mmu_;
    uint64_t capacity_;
};

class LocalMemReader : public SimObject<LocalMemReader> {
public:
    LocalMemReader(const SimContext& ctx, const char* name);
    ~LocalMemReader();
    void tick();
    void reset();

    SimPort<LsuReq> ReqIn; // from MatMul Virgo Controller
    SimPort<LsuRsp> RspIn; // make custom data structure for requests/responses from Virgo Controller

    SimPort<LsuReq> CoreArbReqOut; // to CoreArbiter
    SimPort<LsuRsp> CoreArbRspOut;

    SimPort<LsuReq> ScratchpadReqOut; // to scratchpad
    SimPort<LsuRsp> ScratchpadRspOut;

    SimPort<virgo_req_t> VirgoReqIn; // receive requests from Virgo MM controller
    SimPort<virgo_rsp_t> VirgoRspIn;

private:
    struct pending_req_t {
        bool is_last_req; // last row in the req
        bool is_A;
        bool workload; // 0 for first set of A/B tiles, 1 for 2nd set of A/B tiles
        std::vector<uint64_t> addrs;
        uint32_t row;
        uint32_t count;
        uint32_t orig_count;
    };

    struct LMemState {
        uint32_t row; // which row are we on?
        bool done;
        LMemState() : row(0), done(false) {}
        void reset() { row = 0; done = false; }
    };

    struct ScratchpadState {
        uint32_t is_last_req_count; // 2 for both A & B tiles
        // bool workload;
        uint64_t scratchpad_address_A_;
        uint64_t scratchpad_address_B_;
        ScratchpadState() : is_last_req_count(0) {}
        void reset() { 
            is_last_req_count = 0;
            // workload = !workload; 
            scratchpad_address_A_ = 0;
            scratchpad_address_B_ = 0;
        }
    };

    LMemState lmem_state_A_; // need one for A and B matrixes
    LMemState lmem_state_B_;
    bool lmem_workload_ = 0;
    ScratchpadState scratchpad_state_;

    HashTable<pending_req_t> lmem_pending_reqs_;
    HashTable<pending_req_t> scratchpad_pending_reqs_;

};

class SysSubArray : public SimObject<SysSubArray> {
public:
    SysSubArray(const SimContext& ctx, const char* name, uint32_t size);
    ~SysSubArray();
    void tick();
    void reset();

    std::vector<SimPort<uint32_t>> in_a;
    std::vector<SimPort<uint32_t>> out_a;
    std::vector<SimPort<uint32_t>> in_c;
    std::vector<SimPort<uint32_t>> out_c;
    std::vector<SimPort<uint32_t>> in_weight;
    std::vector<SimPort<uint32_t>> out_weight;
    std::vector<SimPort<uint32_t>> load_en_in;
    std::vector<SimPort<uint32_t>> load_en_out;
    std::vector<SimPort<uint32_t>> flip_reg_in;
    std::vector<SimPort<uint32_t>> flip_reg_out;
    std::vector<SimPort<uint32_t>> in_valid;
    std::vector<SimPort<uint32_t>> out_valid;

private:
    uint32_t size_;
};

class SystolicArray : public SimObject<SystolicArray> {
public:
    SystolicArray(const SimContext& ctx, const char* name, uint32_t tile_size, uint32_t array_size);
    ~SystolicArray();
    void tick();
    void reset();

    std::vector<SimPort<uint32_t>> in_a;
    std::vector<SimPort<uint32_t>> in_b;
    std::vector<SimPort<uint32_t>> in_c;
    std::vector<SimPort<uint32_t>> out_c;
    std::vector<SimPort<uint32_t>> load_en_in;
    std::vector<SimPort<uint32_t>> flip_reg_in;
    std::vector<SimPort<uint32_t>> out_a;
    std::vector<SimPort<uint32_t>> out_weight;
    std::vector<SimPort<uint32_t>> load_en_out;
    std::vector<SimPort<uint32_t>> flip_reg_out;
    std::vector<SimPort<uint32_t>> in_valid;
    std::vector<SimPort<uint32_t>> out_valid;

    // New Ports
    SimPort<SysArrReq> ReqIn;
    SimPort<SysArrRsp> RspOut;
    SimPort<LsuReq> ScratchReadReqOut;
    SimPort<LsuRsp> ScratchReadRspIn;
    SimPort<LsuReq> AccumWriteReqOut;
    SimPort<LsuRsp> AccumWriteRspIn;

private:
    uint32_t tile_size_;
    uint32_t array_size_;
    std::vector<SysSubArray::Ptr> sub_arrays_;

    // Queues
    SimPort<SysArrReq> preload_queueIn;
    SimPort<SysArrReq> preload_queueOut;
    SimPort<SysArrReq> compute_queueIn;
    SimPort<SysArrReq> compute_queueOut;

    // State Variables
    bool preload_active_;
    bool compute_active_;
    bool last_accum_;
    uint32_t accum_counter_;
    uint32_t preload_row_counter_;
    uint32_t compute_counter_;
    SysArrReq preload_req_;
    SysArrReq compute_req_;

    // Pending Requests Tracking
    struct pending_req_t {
        uint32_t sys_tag;
        bool is_weight;
        uint32_t row_num;
        uint32_t count;
    };
    HashTable<pending_req_t> pending_scratchpad_reqs_;
    HashTable<pending_req_t> pending_array_reqs_;
    HashTable<pending_req_t> pending_accumulator_reqs_;
};

class Virgo_MatMul : public SimObject<Virgo_MatMul> {
public:

Virgo_MatMul(const SimContext& ctx,
            const Arch &arch,               // architecture stats essentially
            Cluster* cluster);

~Virgo_MatMul();

SimPort<LsuReq> ReqIn; // MMIO req in
SimPort<LsuRsp> RspIn; // MMIO response

SimPort<MatMulDmaReq> DmaReqOut;
SimPort<MatMulDmaRsp> DmaRspIn;

SimPort<virgo_req_t> LMemReqOut;
SimPort<virgo_rsp_t> LMemRspOut; // might change the types

void read(const void* data, uint64_t addr, uint32_t size);
void write(const void* data, uint64_t addr, uint32_t size);
void MatMul(const virgo_req_t& virgo_queue_entry);
void attach_ram(RAM* ram);
void reset();
void tick();

// Exposed ports for AccumulatorMem
AccumulatorMem* accum_mem() {
    return &accum_mem_;
}

Cluster* cluster() const {
    return cluster_;
}

const LocalMemReader::Ptr& local_mem_reader() const {
    return local_mem_reader_;
}

const ScratchPadMem::Ptr& scratchpad_mem() const {
    return scratchpad_mem_;
}
struct PerfStats {
    uint64_t reads;
    uint64_t writes;
    uint64_t transfers;
};

const PerfStats& perf_stats() const;

private:
    Cluster* cluster_;
    Arch arch_;
    std::vector<uint32_t> write_registers;
    std::vector<uint32_t> read_registers;
    std::vector<std::bitset<32>> tag_table;
    std::queue<virgo_req_t> virgo_req_queue_;
    std::queue<virgo_rsp_t> virgo_rsp_queue_;
    MemoryUnit mmu_;
    AccumulatorMem accum_mem_;
    ScratchPadMem::Ptr scratchpad_mem_;
    LocalMemReader::Ptr local_mem_reader_;
    SystolicArray systolic_array_;
    PerfStats perf_stats_;
    bool inflight_inst;
};

}