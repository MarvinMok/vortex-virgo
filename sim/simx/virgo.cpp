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
#include <bitset>
#include "tensor_cfg.h"

#define FP32 0
#define FP16 1

using namespace vortex;

Virgo_MatMul::Virgo_MatMul(const SimContext& ctx,
                     const Arch &arch,
                     Cluster* cluster)
    : SimObject(ctx, StrFormat("virgo_matmul%d", cluster->id()))
    , cluster_(cluster)
    , arch_(arch)
    , write_registers(arch.num_cores()*arch.num_warps()*8)
    , tag_table(4) // number of max in-flight matmul unit instructions
    , read_register(0)
{
    
}

Virgo_MatMul::~Virgo_MatMul() {}

void Virgo_MatMul::read(const void* data, uint64_t addr, uint32_t size) {
    uint32_t* d = (uint32_t*) data;
    *d = read_register;
}

void Virgo_MatMul::write(const void* data, uint64_t addr, uint32_t size) {
    
    uint32_t* d = (uint32_t*) data;
    uint32_t super_index =((static_cast<uint32_t>(addr) - MMIO_VIRGO_WRITE_ADDR) & 0xFF ) >> 2;
    uint32_t index = super_index % 8;
    
    uint32_t warp_id = (super_index / 8) % arch_.num_warps();
    uint32_t core_id = (super_index / 8) / arch_.num_warps();
    uint32_t global_warp_id = core_id * arch_.num_warps() + warp_id;
    std::cout << "writing super_index " << super_index << " index " << index << " warp id " << warp_id << " core id " << core_id << std::endl;
    write_registers.at(global_warp_id * 8 + index) = *d;
    
    if (index == 7) { // 7 index == 8th thing (commit address, tag)
        uint32_t tag = write_registers.at(global_warp_id * 8 + 7);
        tag_table[tag][global_warp_id] = 1; // set to 1

        if (tag_table[tag].count() == 1) {
            read_register++;
        }
        
        if (tag_table[tag].count() == arch_.num_warps()*arch_.num_cores()) {
            virgo_queue_t virgo_compute = {
                .src_addr_A = write_registers.at(global_warp_id * 8 + 0),
                .src_addr_B = write_registers.at(global_warp_id * 8 + 1),
                .dst_addr = write_registers.at(global_warp_id * 8 + 2),
                .data_type = write_registers.at(global_warp_id * 8 + 3),
                .num_rows_A = write_registers.at(global_warp_id * 8 + 4),
                .num_cols_A = write_registers.at(global_warp_id * 8 + 5),
                .num_cols_B = write_registers.at(global_warp_id * 8 + 6),
                .tag = tag,
            };

            tag_table.at(tag).reset();
            virgo_compute_queue_.push(virgo_compute); // add to queue
            MatMul();
        }
    }   
}

void Virgo_MatMul::MatMul() {
    auto virgo_compute = virgo_compute_queue_.front();
    virgo_compute_queue_.pop();

    uint64_t src_addr_A = static_cast<uint64_t>(virgo_compute.src_addr_A);
    uint64_t src_addr_B = static_cast<uint64_t>(virgo_compute.src_addr_B);
    uint64_t dst_addr = static_cast<uint64_t>(virgo_compute.dst_addr);
    uint32_t data_type = virgo_compute.data_type;
    uint32_t num_rows_A = virgo_compute.num_rows_A;
    uint32_t num_cols_A = virgo_compute.num_cols_A;
    uint32_t num_cols_B = virgo_compute.num_cols_B;

    auto src_addr_A_type = get_addr_type(src_addr_A);
    auto src_addr_B_type = get_addr_type(src_addr_B);
    auto dst_addr_type = get_addr_type(dst_addr);
    std::cout <<"matmuling" << std::endl;
    if (data_type == FP32) {
        for (int i = 0; i < num_rows_A; i++) { // loop over rows of matrix A
            for (int j = 0; j < num_cols_B; j++) { // Loop over columns of matrix B
                float sum;

                if (dst_addr_type == AddrType::Shared) {
                    cluster_->local_mem()->read(static_cast<void*>(&sum), static_cast<uint64_t>(dst_addr + (i * num_cols_B + j) * 4), 4);
                } else {
                    mmu_.read(static_cast<void*>(&sum), static_cast<uint64_t>(dst_addr + (i * num_cols_B + j) * 4), 4, 0);
                }
                // inner loop: Calculate dot product of row i from A and column j from B
                for (int k = 0; k < num_cols_A; k++) {
                    // array[row_index * width + column_index]
                    float aVal;
                    float bVal;
                    if (src_addr_A_type == AddrType::Shared) {
                        cluster_->local_mem()->read(static_cast<void*>(&aVal), static_cast<uint64_t>(src_addr_A + (i * num_cols_A + k) * 4), 4);
                    } else {
                        mmu_.read(static_cast<void*>(&aVal), static_cast<uint64_t>(src_addr_A + (i * num_cols_A + k) * 4), 4, 0);
                    }

                    if (src_addr_B_type == AddrType::Shared) {
                        cluster_->local_mem()->read(static_cast<void*>(&bVal), static_cast<uint64_t>(src_addr_B + (k * num_cols_B + j) * 4), 4);
                    } else {
                        mmu_.read(static_cast<void*>(&bVal), static_cast<uint64_t>(src_addr_B + (k * num_cols_B + j) * 4), 4, 0);
                    }
                    //  = src_addr_A[i * num_cols_A + k]; 
                    //  = src_addr_B[k * num_cols_B + j];   
                    sum += aVal * bVal;
                    std::cout << "matmul vals: " << aVal << " " << bVal << " " << sum << std::endl;
                }
                // Store result in destination matrix C
                // dest_addr[i * num_cols_B + j] = sum;
                std::cout << "matmul res: " << sum << std::endl;
                if (dst_addr_type == AddrType::Shared) {
                    cluster_->local_mem()->write(static_cast<void*>(&sum), static_cast<uint64_t>(dst_addr + (i * num_cols_B + j) * 4), 4);
                } else {
                    mmu_.write(static_cast<void*>(&sum), static_cast<uint64_t>(dst_addr + (i * num_cols_B + j) * 4), 4, 0);
                }
            }
        }
    } else if (data_type == FP16) {
        // for (int i = 0; i < num_rows_A; i++) { // loop over rows of matrix A
        //     for (int j = 0; j < num_cols_B; j++) { // Loop over columns of matrix B
        //         vortex::tensor::fp16 sum = 0;
        //         // inner loop: Calculate dot product of row i from A and column j from B
        //         for (int k = 0; k < num_cols_A; k++) {
        //             // array[row_index * width + column_index]
        //             vortex::tensor::fp16 aVal = src_addr_A[i * num_cols_A + k]; 
        //             vortex::tensor::fp16 bVal = src_addr_B[k * num_cols_B + j];   
        //             sum += aVal * bVal;
        //         }
        //         // Store result in destination matrix C
        //         C[i * num_cols_B + j] = sum;
        //     }
        // }
    }

    read_register--;
}

void Virgo_MatMul::attach_ram(RAM* ram) {
#if (XLEN == 64)
    mmu_.attach(*ram, 0, 0x7FFFFFFFFF); //39bit SV39
#else
    mmu_.attach(*ram, 0, 0xFFFFFFFF);
#endif
}

void Virgo_MatMul::reset() {
}

void Virgo_MatMul::tick() {
    // for TIMING simulation, where we'll actually make the matrix multiply a systolic array
}