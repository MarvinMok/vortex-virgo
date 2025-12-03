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

class Virgo_DMA : public SimObject<Virgo_DMA> {
public:

    Virgo_DMA(const SimContext& ctx,
              const Arch &arch,
              Cluster* cluster);
    ~Virgo_DMA();

    void read(const void* data, uint64_t addr, uint32_t size);
    void write(const void* data, uint64_t addr, uint32_t size);

    void attach_ram(RAM* ram);

    void reset();
    void tick();

private:
    typedef struct {
        uint32_t src_addr;
        uint32_t dst_addr;
        uint32_t data_type_size;
        uint32_t num_rows;
        uint32_t num_cols;
        uint32_t row_stride;
    } dma_load_t;

    void dma_transfer();
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
    MemoryUnit mmu_;
};

} // namespace vortex