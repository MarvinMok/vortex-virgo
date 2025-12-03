#pragma once
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
    uint32_t  tag;
} virgo_queue_t;

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
    
};

}