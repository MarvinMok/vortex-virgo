
#include "dma.h"
#include "cluster.h"
#include <iostream>
namespace vortex {
Virgo_DMA::Virgo_DMA(const SimContext& ctx,
                     const Arch &arch,
                     Cluster* cluster)
    : SimObject(ctx, StrFormat("virgo_dma%d", cluster->id()))
    , cluster_(cluster)
    , warp_counter(0)
    , arch_(arch)
    , write_registers(256, 0)  
    , read_registers(1, 0) 
{
    
}

Virgo_DMA::~Virgo_DMA() {}

void Virgo_DMA::read(const void* data,  uint64_t /*addr*/, uint32_t /*size*/) {
    uint32_t* d = (uint32_t*)data;
    *d = read_registers.at(0);
}

void Virgo_DMA::write(const void* data, uint64_t addr, uint32_t /*size*/) {
    uint32_t* d = (uint32_t*)data;
    uint32_t index = ((static_cast<uint32_t>(addr) - MMIO_WRITE_ADDR) & 0xFF ) >> 2;
    std::cout << "DMA write to index " << index << ", addr  " << std::hex << addr << ", MMIO " << std::hex << MMIO_WRITE_ADDR << ", data " << *d << std::endl;
    write_registers.at(index) = *d;

    if (index == 0x10) {
        warp_counter++;
        std::cout << "warp counter: " << warp_counter << std::endl;
        if (warp_counter == arch_.num_warps() * arch_.num_cores()) {
            warp_counter = 0;
            std::cout << "Begin DMA transfer" << std::endl;
            dma_load_t dma_load = {
                .src_addr = write_registers.at(0),
                .dst_addr = write_registers.at(1),
                .data_type_size = write_registers.at(2),
                .num_rows = write_registers.at(3),
                .num_cols = write_registers.at(4),
                .row_stride = write_registers.at(5),
                .core_id = write_registers.at(6),
                .wid = write_registers.at(7)
            };
            dma_load_queue_.push(dma_load);
            read_registers.at(0)++;
            dma_transfer();
        }
    }
}

void Virgo_DMA::dma_transfer(){
    
    auto dma_load = dma_load_queue_.front();
    dma_load_queue_.pop();
    std::cout << "DMA transfer from " << dma_load.src_addr << " to " << dma_load.dst_addr << std::endl;
    std::cout << "Data type size: " << dma_load.data_type_size << std::endl;
    std::cout << "Num rows: " << dma_load.num_rows << std::endl;
    std::cout << "Num cols: " << dma_load.num_cols << std::endl;
    std::cout << "Row stride: " << dma_load.row_stride << std::endl;
    auto src_addr_type = get_addr_type(dma_load.src_addr);
    auto dst_addr_type = get_addr_type(dma_load.dst_addr);

    for (uint32_t row = 0; row < dma_load.num_rows; row++) {
        for (uint32_t col = 0; col < dma_load.num_cols; col++) {
            auto index = row * dma_load.row_stride + col;
            auto src_addr = dma_load.src_addr + index * dma_load.data_type_size;
            auto dst_addr = dma_load.dst_addr + index * dma_load.data_type_size;
            data_transfer(src_addr, dst_addr, dma_load.data_type_size, src_addr_type, dst_addr_type);
        }
    }
    read_registers.at(0)--;
}

void Virgo_DMA::data_transfer(uint64_t src_addr, uint64_t dst_addr, uint32_t data_type_size, AddrType src_addr_type, AddrType dst_addr_type) {
    uint64_t data = 0;

    auto true_src_type = get_addr_type(src_addr);
    auto true_dst_type = get_addr_type(dst_addr);
    if (true_src_type != src_addr_type || true_dst_type != dst_addr_type) {
        std::cout << "Error: src_addr_type and dst_addr_type do not match for DMA transfer" << std::endl;
        std::abort();
    }

    if (src_addr_type == AddrType::Shared) {
        mmu_.read(static_cast<void*>(&data), static_cast<uint64_t>(src_addr), data_type_size, 0);
    } else {
        mmu_.read(static_cast<void*>(&data), static_cast<uint64_t>(src_addr), data_type_size, 0);
    }

    std::cout << "Data transfer from " << src_addr << " to " << dst_addr << " with data " << data << std::endl;

    if (dst_addr_type == AddrType::Shared) {
        mmu_.write(static_cast<void*>(&data), static_cast<uint64_t>(dst_addr), data_type_size, 0);
    } else {
        mmu_.write(static_cast<void*>(&data), static_cast<uint64_t>(dst_addr), data_type_size, 0);
    }

}

void Virgo_DMA::attach_ram(RAM* ram) {
#if (XLEN == 64)
    mmu_.attach(*ram, 0, 0x7FFFFFFFFF); //39bit SV39
#else
    mmu_.attach(*ram, 0, 0xFFFFFFFF);
#endif
}

void Virgo_DMA::reset() {
    warp_counter = 0;
    read_registers.at(0) = 0;
}

void Virgo_DMA::tick() {
}



} // namespace vortex