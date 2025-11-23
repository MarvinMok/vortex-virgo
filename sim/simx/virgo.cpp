#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "types.h"
#include "arch.h"
#include "mem.h"
#include "cluster.h"
#include "virgo.h"

using namespace vortex;

Virgo_MatMul::Virgo_MatMul(const SimContext& ctx,
                     const Arch &arch,
                     Cluster* cluster)
    : SimObject(ctx, StrFormat("virgo_matmul%d", cluster->id()))
    , cluster_(cluster)
    , arch_(arch)
    , write_registers(arch.num_cores()*arch.num_warps()*11)
    , read_registers(1, 0)
    , tag_table(4) // number of max in-flight matmul unit instructions
{
    
}

Virgo_MatMul::~Virgo_MatMul() {}

void Virgo_MatMul::read(const void* data, uint64_t addr, uint32_t addr) {

}

void Virgo_MatMul::write(const void* data, uint64_t addr, uint32_t addr) {
    uint32_t* d = (uint32_t*) data;
    uint32_t super_index = (static_cast<uint32_t>(addr) - MMIO_VIRGO_WRITE_ADDR) / 4;
    uint32_t index = super_index % 11;
    uint32_t warp_id = (super_index / 11) % arch.num_warps();
    uint32_t core_id = (super_index / 11) / arch.num_warps();

    write_registers.at(warp_id*core_id + index) = *d;
    
    if (index == 10) { // 10 index == 11th thing (commit address)
        tag_table.at(write_registers.at(warp_id*core_id + 10))++;
        if (tag_table.at(write_registers.at(warp_id*core_id + 10) == arch.num_warps()*arch.num_cores())) {
            virgo_compute_t virgo_compute = {
                .src_addr_A = write_registers.at(warp_id*core_id + 0),
                .src_addr_B = write_registers.at(warp_id*core_id + 1),
                .dst_addr = write_registers.at(warp_id*core_id + 2),
                .data_type_size = write_registers.at(warp_id*core_id + 3),
                .num_rows_A = write_registers.at(warp_id*core_id + 4),
                .num_cols_A = write_registers.at(warp_id*core_id + 5),
                .num_cols_B = write_registers.at(warp_id*core_id + 6),
                .core_id = write_registers.at(warp_id*core_id + 7),
                .wid = write_registers.at(warp_id*core_id + 8),
                .tag = write_registers.at(warp_id*core_id + 9),
            };

            // add to queue
            virgo_compute_queue_.push(virgo_compute);

            tag_table.at(write_registers.at(warp_id*core_id + 10)) = 0;
        }
    }   
}