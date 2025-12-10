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
} virgo_queue_t;

#define SCRATCHPAD_BANKS NUM_LSU_LANES

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
};

class SystolicArray : public SimObject<SystolicArray> {
public:
    SystolicArray(const SimContext& ctx, const char* name);
    ~SystolicArray();
    void tick();
    void reset();
};

class SysSubArray : public SimObject<SysSubArray> {
public:
    SysSubArray(const SimContext& ctx, const char* name);
    ~SysSubArray();
    void tick();
    void reset();
};

class Virgo_MatMul : public SimObject<Virgo_MatMul> {
public:

Virgo_MatMul(const SimContext& ctx,
            const Arch &arch,               // architecture stats essentially
            Cluster* cluster);

~Virgo_MatMul();

void read(const void* data, uint64_t addr, uint32_t size);
void write(const void* data, uint64_t addr, uint32_t size);
void MatMul();
void attach_ram(RAM* ram);
void reset();
void tick();

SimPort<MatMulDmaReq> DmaReqOut;
SimPort<MatMulDmaRsp> DmaRspIn;

// Exposed ports for AccumulatorMem
SimPort<LsuReq> AccumReadReqIn;
SimPort<LsuRsp> AccumReadRspOut;
SimPort<LsuReq> AccumWriteReqIn;
SimPort<LsuRsp> AccumWriteRspOut;

// Exposed ports for ScratchPadMem
SimPort<LsuReq> ScratchReadReqIn;
SimPort<LsuRsp> ScratchReadRspOut;
SimPort<LsuReq> ScratchWriteReqIn;
SimPort<LsuRsp> ScratchWriteRspOut;

Cluster* cluster() const {
    return cluster_;
}

private:
    Cluster* cluster_;
    Arch arch_;
    std::vector<uint32_t> write_registers;
    std::vector<uint32_t> read_registers;
    std::vector<std::bitset<32>> tag_table;
    std::queue<virgo_queue_t> virgo_compute_queue_;
    MemoryUnit mmu_;
    AccumulatorMem accum_mem_;
    ScratchPadMem scratchpad_mem_;
    LocalMemReader local_mem_reader_;
    SystolicArray systolic_array_;
};

}