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

class AccumulatorMem : public SimObject<AccumulatorMem> {
public:
    AccumulatorMem(const SimContext& ctx, const char* name, uint64_t capacity);
    ~AccumulatorMem();

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

class Virgo_MatMul : public SimObject<Virgo_MatMul> {
public:

Virgo_MatMul(const SimContext& ctx,
            const Arch &arch,               // architecture stats essentially
            Cluster* cluster);

~Virgo_MatMul();

SimPort<LsuReq> ReqIn;
SimPort<LsuRsp> RspIn;

void read(const void* data, uint64_t addr, uint32_t size);
void write(const void* data, uint64_t addr, uint32_t size);
void MatMul();
void attach_ram(RAM* ram);
void reset();
void tick();

Cluster* cluster() const {
    return cluster_;
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
    std::queue<virgo_queue_t> virgo_compute_queue_;
    MemoryUnit mmu_;
    AccumulatorMem accum_mem_;
    ScratchPadMem scratchpad_mem_;
    PerfStats perf_stats_;
};

}