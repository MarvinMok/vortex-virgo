
#include "dma.h"
#include "cluster.h"
#include <iostream>
namespace vortex {
Virgo_DMA::Virgo_DMA(const SimContext& ctx,
                     const Arch &arch,
                     Cluster* cluster)
    : SimObject(ctx, StrFormat("virgo_dma%d", cluster->id()))
    , cluster_(cluster)
    , arch_(arch)
    , write_registers(256, 0)  
    , read_registers(1, 0) 
    , tag_table(4)
{
    
}

Virgo_DMA::~Virgo_DMA() {}

void Virgo_DMA::read(const void* data,  uint64_t /* addr */, uint32_t /* size */) {
    uint32_t* d = (uint32_t*)data;
    *d = read_registers.at(0);
}

void Virgo_DMA::write(const void* data, uint64_t addr, uint32_t /*size*/) {
    uint32_t* d = (uint32_t*)data;
    uint32_t super_index = ((static_cast<uint32_t>(addr) - MMIO_WRITE_ADDR) & 0xFFF ) >> 2;
    uint32_t index = super_index % 8;
    uint32_t warp_id = (super_index / 8) % arch_.num_warps();
    uint32_t core_id = (super_index / 8) / arch_.num_warps();
    uint32_t global_warp_id = core_id * arch_.num_warps() + warp_id;
    std::cout << "DMA write to index " << index << ", addr  " << std::hex << addr << ", MMIO " << std::hex << MMIO_WRITE_ADDR << ", data " << *d << std::endl;
    write_registers.at(global_warp_id * 8 + index) = *d;
    if (index == 7) {
        if (tag_table.at(*d).count() == 0) {
            read_registers.at(0)++;
        }
        tag_table.at(*d).set(global_warp_id);
        if (tag_table.at(*d).count() == arch_.num_warps() * arch_.num_cores()) {
            std::cout << "Begin DMA transfer" << std::endl;
            dma_load_t dma_load = {
                .src_addr = write_registers.at(global_warp_id * 8 + 0),
                .dst_addr = write_registers.at(global_warp_id * 8 + 1),
                .data_type_size = write_registers.at(global_warp_id * 8 + 2),
                .num_rows = write_registers.at(global_warp_id * 8 + 3),
                .num_cols = write_registers.at(global_warp_id * 8 + 4),
                .src_stride = write_registers.at(global_warp_id * 8 + 5),
                .dst_stride = write_registers.at(global_warp_id * 8 + 6),
            };
            tag_table.at(*d).reset();
            dma_load_queue_.push(dma_load);
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
    std::cout << "Src stride: " << dma_load.src_stride << std::endl;
    std::cout << "Dst stride: " << dma_load.dst_stride << std::endl;
    auto src_addr_type = get_addr_type(dma_load.src_addr);
    auto dst_addr_type = get_addr_type(dma_load.dst_addr);

    for (uint32_t row = 0; row < dma_load.num_rows; row++) {
        for (uint32_t col = 0; col < dma_load.num_cols; col++) {
            auto src_index = row * dma_load.src_stride + col;
            auto dst_index = row * dma_load.dst_stride + col;
            auto src_addr = dma_load.src_addr + src_index * dma_load.data_type_size;
            auto dst_addr = dma_load.dst_addr + dst_index * dma_load.data_type_size;
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
        cluster_->local_mem()->read(static_cast<void*>(&data), static_cast<uint64_t>(src_addr), data_type_size);
    } else {
        mmu_.read(static_cast<void*>(&data), static_cast<uint64_t>(src_addr), data_type_size, 0);
    }

    std::cout << "Data transfer from " << src_addr << " to " << dst_addr << " with data " << data << std::endl;

    if (dst_addr_type == AddrType::Shared) {
        cluster_->local_mem()->write(static_cast<void*>(&data), static_cast<uint64_t>(dst_addr), data_type_size);
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
    read_registers.at(0) = 0;
    for (uint32_t i = 0; i < write_registers.size(); i++) {
        write_registers.at(i) = 0;
    }
    for (uint32_t i = 0; i < tag_table.size(); i++) {
        tag_table.at(i).reset();
    }
}

void Virgo_DMA::tick() {
}



} // namespace vortex